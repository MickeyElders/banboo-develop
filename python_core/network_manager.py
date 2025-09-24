"""
网络通信管理器 - Python实现
负责TCP Socket通信和Modbus服务器
"""

import logging
import socket
import threading
import time
import json
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, asdict
from queue import Queue, Empty
import struct

logger = logging.getLogger(__name__)

# 尝试导入Modbus库
try:
    from pymodbus.server.sync import StartTcpServer
    from pymodbus.device import ModbusDeviceIdentification
    from pymodbus.datastore import ModbusSequentialDataBlock, ModbusSlaveContext, ModbusServerContext
    from pymodbus.transaction import ModbusRtuFramer, ModbusSocketFramer
    MODBUS_AVAILABLE = True
except ImportError:
    logger.warning("Modbus库不可用，将禁用Modbus功能")
    MODBUS_AVAILABLE = False


@dataclass
class NetworkConfig:
    """网络配置"""
    tcp_enabled: bool = True              # 启用TCP服务器
    tcp_host: str = "0.0.0.0"            # TCP服务器地址
    tcp_port: int = 8888                  # TCP服务器端口
    tcp_max_clients: int = 5              # 最大客户端连接数
    
    modbus_enabled: bool = True           # 启用Modbus服务器
    modbus_host: str = "0.0.0.0"         # Modbus服务器地址
    modbus_port: int = 502                # Modbus服务器端口
    modbus_slave_id: int = 1              # Modbus从机ID
    
    heartbeat_interval: float = 5.0       # 心跳间隔(秒)
    connection_timeout: float = 30.0      # 连接超时(秒)


@dataclass
class CuttingCommand:
    """切割命令数据结构"""
    x: float                              # X坐标 (mm)
    y: float                              # Y坐标 (mm)
    z: float = 0.0                        # Z坐标 (mm)
    speed: float = 100.0                  # 切割速度 (mm/min)
    command_id: int = 0                   # 命令ID
    timestamp: float = 0.0                # 时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
        
    def to_bytes(self) -> bytes:
        """转换为字节数据"""
        return json.dumps(self.to_dict()).encode('utf-8')
        
    @classmethod
    def from_bytes(cls, data: bytes) -> 'CuttingCommand':
        """从字节数据创建"""
        try:
            json_data = json.loads(data.decode('utf-8'))
            return cls(**json_data)
        except Exception as e:
            logger.error(f"解析切割命令失败: {e}")
            return cls(0.0, 0.0)


class TCPServer:
    """TCP服务器实现"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.server_socket = None
        self.running = False
        self.server_thread = None
        self.client_threads = []
        self.clients = {}  # 客户端连接字典
        self.message_callback = None
        
    def start(self) -> bool:
        """启动TCP服务器"""
        try:
            if self.running:
                logger.warning("TCP服务器已在运行")
                return True
                
            # 创建服务器套接字
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.config.tcp_host, self.config.tcp_port))
            self.server_socket.listen(self.config.tcp_max_clients)
            
            self.running = True
            
            # 启动服务器线程
            self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
            self.server_thread.start()
            
            logger.info(f"TCP服务器已启动: {self.config.tcp_host}:{self.config.tcp_port}")
            return True
            
        except Exception as e:
            logger.error(f"TCP服务器启动失败: {e}")
            return False
            
    def stop(self):
        """停止TCP服务器"""
        try:
            self.running = False
            
            # 关闭所有客户端连接
            for client_id, client_info in list(self.clients.items()):
                try:
                    client_info['socket'].close()
                except:
                    pass
                    
            self.clients.clear()
            
            # 关闭服务器套接字
            if self.server_socket:
                self.server_socket.close()
                self.server_socket = None
                
            # 等待线程退出
            if self.server_thread and self.server_thread.is_alive():
                self.server_thread.join(timeout=2.0)
                
            logger.info("TCP服务器已停止")
            
        except Exception as e:
            logger.error(f"TCP服务器停止异常: {e}")
            
    def _server_loop(self):
        """服务器主循环"""
        logger.info("TCP服务器监听线程已启动")
        
        while self.running:
            try:
                # 设置超时以便检查running状态
                self.server_socket.settimeout(1.0)
                client_socket, address = self.server_socket.accept()
                
                if not self.running:
                    client_socket.close()
                    break
                    
                # 为新客户端创建处理线程
                client_id = f"{address[0]}:{address[1]}_{int(time.time())}"
                client_info = {
                    'socket': client_socket,
                    'address': address,
                    'connected_time': time.time(),
                    'last_activity': time.time()
                }
                
                self.clients[client_id] = client_info
                
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_id, client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
                logger.info(f"新客户端连接: {client_id}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"TCP服务器接受连接异常: {e}")
                    
        logger.info("TCP服务器监听线程已退出")
        
    def _handle_client(self, client_id: str, client_socket: socket.socket, address: tuple):
        """处理客户端连接"""
        logger.info(f"客户端处理线程已启动: {client_id}")
        
        try:
            client_socket.settimeout(self.config.connection_timeout)
            
            while self.running and client_id in self.clients:
                try:
                    # 接收数据
                    data = client_socket.recv(1024)
                    if not data:
                        logger.info(f"客户端断开连接: {client_id}")
                        break
                        
                    # 更新活动时间
                    self.clients[client_id]['last_activity'] = time.time()
                    
                    # 处理消息
                    if self.message_callback:
                        try:
                            response = self.message_callback(client_id, data)
                            if response:
                                client_socket.send(response)
                        except Exception as e:
                            logger.error(f"消息回调异常: {e}")
                            
                except socket.timeout:
                    # 检查连接是否超时
                    current_time = time.time()
                    last_activity = self.clients[client_id]['last_activity']
                    if current_time - last_activity > self.config.connection_timeout:
                        logger.warning(f"客户端连接超时: {client_id}")
                        break
                        
                except Exception as e:
                    logger.error(f"处理客户端数据异常: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"客户端处理异常: {e}")
        finally:
            # 清理客户端连接
            try:
                client_socket.close()
                if client_id in self.clients:
                    del self.clients[client_id]
                logger.info(f"客户端连接已清理: {client_id}")
            except:
                pass
                
    def broadcast_message(self, message: bytes) -> int:
        """广播消息到所有客户端"""
        sent_count = 0
        for client_id, client_info in list(self.clients.items()):
            try:
                client_info['socket'].send(message)
                sent_count += 1
            except Exception as e:
                logger.error(f"向客户端 {client_id} 发送消息失败: {e}")
                
        return sent_count
        
    def send_to_client(self, client_id: str, message: bytes) -> bool:
        """发送消息到指定客户端"""
        try:
            if client_id in self.clients:
                self.clients[client_id]['socket'].send(message)
                return True
            else:
                logger.warning(f"客户端不存在: {client_id}")
                return False
        except Exception as e:
            logger.error(f"发送消息到客户端 {client_id} 失败: {e}")
            return False
            
    def set_message_callback(self, callback: Callable[[str, bytes], Optional[bytes]]):
        """设置消息回调函数"""
        self.message_callback = callback
        
    def get_clients_info(self) -> Dict[str, Dict[str, Any]]:
        """获取客户端信息"""
        info = {}
        current_time = time.time()
        for client_id, client_info in self.clients.items():
            info[client_id] = {
                'address': client_info['address'],
                'connected_time': client_info['connected_time'],
                'last_activity': client_info['last_activity'],
                'connection_duration': current_time - client_info['connected_time']
            }
        return info


class ModbusServer:
    """Modbus服务器实现"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.server_thread = None
        self.running = False
        self.context = None
        
    def start(self) -> bool:
        """启动Modbus服务器"""
        try:
            if not MODBUS_AVAILABLE:
                logger.error("Modbus库不可用")
                return False
                
            if self.running:
                logger.warning("Modbus服务器已在运行")
                return True
                
            # 创建数据存储
            store = ModbusSlaveContext(
                di=ModbusSequentialDataBlock(0, [0] * 100),    # 离散输入
                co=ModbusSequentialDataBlock(0, [0] * 100),    # 线圈
                hr=ModbusSequentialDataBlock(0, [0] * 100),    # 保持寄存器
                ir=ModbusSequentialDataBlock(0, [0] * 100)     # 输入寄存器
            )
            
            self.context = ModbusServerContext(slaves=store, single=True)
            
            # 设置设备信息
            identity = ModbusDeviceIdentification()
            identity.VendorName = 'Bamboo Cutting System'
            identity.ProductCode = 'BCS'
            identity.VendorUrl = 'http://github.com/bamboo-cutting'
            identity.ProductName = 'Bamboo Detection Controller'
            identity.ModelName = 'BDC-v1.0'
            identity.MajorMinorRevision = '1.0.0'
            
            self.running = True
            
            # 启动Modbus服务器线程
            self.server_thread = threading.Thread(
                target=self._start_modbus_server,
                args=(identity,),
                daemon=True
            )
            self.server_thread.start()
            
            logger.info(f"Modbus服务器已启动: {self.config.modbus_host}:{self.config.modbus_port}")
            return True
            
        except Exception as e:
            logger.error(f"Modbus服务器启动失败: {e}")
            return False
            
    def _start_modbus_server(self, identity):
        """启动Modbus服务器线程"""
        try:
            StartTcpServer(
                context=self.context,
                identity=identity,
                address=(self.config.modbus_host, self.config.modbus_port),
                framer=ModbusSocketFramer
            )
        except Exception as e:
            logger.error(f"Modbus服务器线程异常: {e}")
            
    def stop(self):
        """停止Modbus服务器"""
        try:
            self.running = False
            # Modbus服务器停止需要特殊处理
            logger.info("Modbus服务器已停止")
        except Exception as e:
            logger.error(f"Modbus服务器停止异常: {e}")
            
    def write_register(self, address: int, value: int) -> bool:
        """写入保持寄存器"""
        try:
            if self.context:
                slave_context = self.context[self.config.modbus_slave_id]
                slave_context.setValues(3, address, [value])  # 功能码3 = 保持寄存器
                return True
            return False
        except Exception as e:
            logger.error(f"写入Modbus寄存器失败: {e}")
            return False
            
    def read_register(self, address: int) -> Optional[int]:
        """读取保持寄存器"""
        try:
            if self.context:
                slave_context = self.context[self.config.modbus_slave_id]
                values = slave_context.getValues(3, address, 1)  # 功能码3 = 保持寄存器
                return values[0] if values else None
            return None
        except Exception as e:
            logger.error(f"读取Modbus寄存器失败: {e}")
            return None


class NetworkManager:
    """网络通信管理器"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.tcp_server = None
        self.modbus_server = None
        self.command_queue = Queue()
        self.statistics = {
            'tcp_connections': 0,
            'messages_received': 0,
            'messages_sent': 0,
            'commands_processed': 0,
            'start_time': time.time()
        }
        
    def initialize(self) -> bool:
        """初始化网络管理器"""
        try:
            logger.info("初始化网络管理器...")
            
            success = True
            
            # 初始化TCP服务器
            if self.config.tcp_enabled:
                self.tcp_server = TCPServer(self.config)
                self.tcp_server.set_message_callback(self._handle_tcp_message)
                if not self.tcp_server.start():
                    logger.error("TCP服务器启动失败")
                    success = False
                    
            # 初始化Modbus服务器  
            if self.config.modbus_enabled:
                self.modbus_server = ModbusServer(self.config)
                if not self.modbus_server.start():
                    logger.error("Modbus服务器启动失败")
                    success = False
                    
            if success:
                logger.info("网络管理器初始化成功")
            else:
                logger.error("网络管理器初始化失败")
                
            return success
            
        except Exception as e:
            logger.error(f"网络管理器初始化异常: {e}")
            return False
            
    def _handle_tcp_message(self, client_id: str, data: bytes) -> Optional[bytes]:
        """处理TCP消息"""
        try:
            self.statistics['messages_received'] += 1
            
            # 尝试解析为切割命令
            try:
                command = CuttingCommand.from_bytes(data)
                command.timestamp = time.time()
                
                # 添加到命令队列
                self.command_queue.put(command)
                self.statistics['commands_processed'] += 1
                
                # 返回确认消息
                response = {
                    'status': 'success',
                    'command_id': command.command_id,
                    'timestamp': command.timestamp
                }
                
                response_data = json.dumps(response).encode('utf-8')
                self.statistics['messages_sent'] += 1
                
                return response_data
                
            except Exception as e:
                # 如果不是切割命令，返回错误
                error_response = {
                    'status': 'error',
                    'message': f'Invalid command format: {e}'
                }
                return json.dumps(error_response).encode('utf-8')
                
        except Exception as e:
            logger.error(f"处理TCP消息异常: {e}")
            return None
            
    def send_detection_result(self, detection_result) -> int:
        """发送检测结果到所有客户端"""
        try:
            # 转换检测结果为JSON
            result_data = {
                'type': 'detection_result',
                'data': detection_result.to_dict(),
                'timestamp': time.time()
            }
            
            message = json.dumps(result_data).encode('utf-8')
            
            # TCP广播
            sent_count = 0
            if self.tcp_server:
                sent_count = self.tcp_server.broadcast_message(message)
                
            # Modbus更新（如果需要）
            if self.modbus_server and detection_result.points:
                # 将第一个检测点写入Modbus寄存器
                first_point = detection_result.points[0]
                self.modbus_server.write_register(0, int(first_point.x))  # X坐标
                self.modbus_server.write_register(1, int(first_point.y))  # Y坐标
                self.modbus_server.write_register(2, int(first_point.confidence * 100))  # 置信度
                
            self.statistics['messages_sent'] += sent_count
            return sent_count
            
        except Exception as e:
            logger.error(f"发送检测结果异常: {e}")
            return 0
            
    def get_cutting_command(self, timeout: float = 0.1) -> Optional[CuttingCommand]:
        """获取切割命令"""
        try:
            return self.command_queue.get(timeout=timeout)
        except Empty:
            return None
        except Exception as e:
            logger.error(f"获取切割命令异常: {e}")
            return None
            
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        current_time = time.time()
        runtime = current_time - self.statistics['start_time']
        
        stats = self.statistics.copy()
        stats['runtime_seconds'] = runtime
        
        # TCP服务器统计
        if self.tcp_server:
            stats['tcp_clients'] = len(self.tcp_server.clients)
            stats['tcp_clients_info'] = self.tcp_server.get_clients_info()
        else:
            stats['tcp_clients'] = 0
            stats['tcp_clients_info'] = {}
            
        # Modbus统计
        stats['modbus_enabled'] = self.config.modbus_enabled and self.modbus_server is not None
        
        return stats
        
    def cleanup(self):
        """清理网络资源"""
        try:
            logger.info("清理网络资源...")
            
            # 停止TCP服务器
            if self.tcp_server:
                self.tcp_server.stop()
                self.tcp_server = None
                
            # 停止Modbus服务器
            if self.modbus_server:
                self.modbus_server.stop()
                self.modbus_server = None
                
            # 清空命令队列
            while not self.command_queue.empty():
                try:
                    self.command_queue.get_nowait()
                except:
                    break
                    
            logger.info("网络资源清理完成")
            
        except Exception as e:
            logger.error(f"网络资源清理异常: {e}")