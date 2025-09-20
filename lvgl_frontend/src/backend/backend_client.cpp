#include "backend/backend_client.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <sys/wait.h>

// 获取当前时间戳（毫秒）
static uint64_t get_timestamp_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// 读取系统硬件信息
static void read_system_health(system_health_t* health) {
    if (!health) return;
    
    // 初始化默认值
    health->cpu_usage = 0.0f;
    health->memory_usage = 0.0f;
    health->gpu_usage = 0.0f;
    health->gpu_memory_usage = 0.0f;
    health->temperature = 0.0f;
    health->ai_fps = 0;
    health->camera_fps = 30; // 默认摄像头帧率
    health->heartbeat_count = (uint32_t)(get_timestamp_ms() / 1000);
    
    // 读取CPU使用率
    FILE* fp = fopen("/proc/loadavg", "r");
    if (fp) {
        float load1, load5, load15;
        if (fscanf(fp, "%f %f %f", &load1, &load5, &load15) == 3) {
            health->cpu_usage = load1 * 25.0f; // 简化计算，假设4核
            if (health->cpu_usage > 100.0f) health->cpu_usage = 100.0f;
        }
        fclose(fp);
    }
    
    // 读取内存使用率
    fp = fopen("/proc/meminfo", "r");
    if (fp) {
        char line[256];
        unsigned long total_mem = 0, free_mem = 0, available_mem = 0;
        
        while (fgets(line, sizeof(line), fp)) {
            if (sscanf(line, "MemTotal: %lu kB", &total_mem) == 1) continue;
            if (sscanf(line, "MemFree: %lu kB", &free_mem) == 1) continue;
            if (sscanf(line, "MemAvailable: %lu kB", &available_mem) == 1) break;
        }
        
        if (total_mem > 0 && available_mem > 0) {
            health->memory_usage = (float)(total_mem - available_mem) * 100.0f / total_mem;
        }
        fclose(fp);
    }
    
    // 读取GPU信息（nvidia-smi）
    fp = popen("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits 2>/dev/null", "r");
    if (fp) {
        float gpu_util, gpu_mem_util, gpu_temp;
        if (fscanf(fp, "%f, %f, %f", &gpu_util, &gpu_mem_util, &gpu_temp) == 3) {
            health->gpu_usage = gpu_util;
            health->gpu_memory_usage = gpu_mem_util;
            health->temperature = gpu_temp;
        }
        pclose(fp);
    }
    
    // 如果没有GPU，读取CPU温度
    if (health->temperature == 0.0f) {
        fp = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
        if (fp) {
            int temp_millicelsius;
            if (fscanf(fp, "%d", &temp_millicelsius) == 1) {
                health->temperature = temp_millicelsius / 1000.0f;
            }
            fclose(fp);
        }
    }
}

// 后端通信线程
static void* backend_communication_thread(void* arg) {
    backend_client_t* client = (backend_client_t*)arg;
    printf("后端通信线程启动\n");
    
    while (client->thread_running) {
        // 模拟与后端通信
        pthread_mutex_lock(&client->data_mutex);
        
        // 更新系统健康信息
        read_system_health(&client->system_health);
        
        // 模拟坐标数据更新（实际应该从后端获取）
        static int coord_counter = 0;
        if (coord_counter++ % 100 == 0) { // 每3秒更新一次坐标
            client->current_coordinate.coordinate_ready = true;
            client->current_coordinate.x_coordinate = 2500 + (coord_counter % 1000);
            client->current_coordinate.blade_number = (coord_counter % 3) + 1;
            client->current_coordinate.cutting_quality = 0;
            client->current_coordinate.timestamp = get_timestamp_ms();
        }
        
        // 更新心跳
        client->last_heartbeat_time = get_timestamp_ms();
        
        pthread_mutex_unlock(&client->data_mutex);
        
        usleep(30000); // 30ms更新间隔
    }
    
    printf("后端通信线程退出\n");
    return nullptr;
}

backend_client_t* backend_client_create(const char* host, int port) {
    if (!host || port <= 0) {
        printf("错误：后端客户端参数无效\n");
        return nullptr;
    }
    
    backend_client_t* client = (backend_client_t*)calloc(1, sizeof(backend_client_t));
    if (!client) {
        printf("错误：分配后端客户端内存失败\n");
        return nullptr;
    }
    
    // 初始化基本参数
    strncpy(client->backend_host, host, sizeof(client->backend_host) - 1);
    client->backend_port = port;
    client->socket_fd = -1;
    
    // 初始化状态
    client->backend_status = BACKEND_DISCONNECTED;
    client->plc_status = PLC_DISCONNECTED;
    client->system_status = SYSTEM_STOPPED;
    
    // 初始化数据
    memset(&client->current_coordinate, 0, sizeof(client->current_coordinate));
    memset(&client->system_health, 0, sizeof(client->system_health));
    
    // 初始化线程同步
    pthread_mutex_init(&client->data_mutex, nullptr);
    client->thread_running = false;
    
    // 初始化统计信息
    client->messages_sent = 0;
    client->messages_received = 0;
    client->last_heartbeat_time = 0;
    
    printf("创建后端客户端: %s:%d\n", host, port);
    return client;
}

void backend_client_destroy(backend_client_t* client) {
    if (!client) return;
    
    printf("销毁后端客户端\n");
    
    // 停止通信
    backend_client_stop_communication(client);
    
    // 断开连接
    backend_client_disconnect(client);
    
    // 销毁互斥锁
    pthread_mutex_destroy(&client->data_mutex);
    
    free(client);
}

bool backend_client_connect(backend_client_t* client) {
    if (!client) return false;
    
    printf("连接到后端: %s:%d\n", client->backend_host, client->backend_port);
    
    // 模拟连接（实际应该建立TCP连接）
    client->backend_status = BACKEND_CONNECTED;
    client->plc_status = PLC_CONNECTED; // 模拟PLC也连接
    client->system_status = SYSTEM_RUNNING;
    
    printf("后端连接成功\n");
    return true;
}

void backend_client_disconnect(backend_client_t* client) {
    if (!client) return;
    
    printf("断开后端连接\n");
    
    if (client->socket_fd >= 0) {
        close(client->socket_fd);
        client->socket_fd = -1;
    }
    
    client->backend_status = BACKEND_DISCONNECTED;
    client->plc_status = PLC_DISCONNECTED;
    client->system_status = SYSTEM_STOPPED;
}

bool backend_client_start_communication(backend_client_t* client) {
    if (!client) return false;
    
    if (client->thread_running) {
        printf("警告：通信线程已经在运行\n");
        return true;
    }
    
    printf("启动后端通信线程\n");
    
    client->thread_running = true;
    if (pthread_create(&client->communication_thread, nullptr, backend_communication_thread, client) != 0) {
        printf("错误：创建通信线程失败: %s\n", strerror(errno));
        client->thread_running = false;
        return false;
    }
    
    printf("后端通信线程启动成功\n");
    return true;
}

void backend_client_stop_communication(backend_client_t* client) {
    if (!client || !client->thread_running) return;
    
    printf("停止后端通信线程\n");
    
    client->thread_running = false;
    
    // 等待线程退出
    pthread_join(client->communication_thread, nullptr);
    
    printf("后端通信线程已停止\n");
}

bool backend_client_send_plc_command(backend_client_t* client, int16_t command) {
    if (!client || client->backend_status != BACKEND_CONNECTED) {
        return false;
    }
    
    printf("发送PLC命令: %d\n", command);
    
    pthread_mutex_lock(&client->data_mutex);
    client->messages_sent++;
    pthread_mutex_unlock(&client->data_mutex);
    
    return true;
}

bool backend_client_get_coordinate(backend_client_t* client, cutting_coordinate_t* coordinate) {
    if (!client || !coordinate) return false;
    
    pthread_mutex_lock(&client->data_mutex);
    *coordinate = client->current_coordinate;
    pthread_mutex_unlock(&client->data_mutex);
    
    return true;
}

bool backend_client_get_system_health(backend_client_t* client, system_health_t* health) {
    if (!client || !health) return false;
    
    pthread_mutex_lock(&client->data_mutex);
    *health = client->system_health;
    pthread_mutex_unlock(&client->data_mutex);
    
    return true;
}

backend_status_t backend_client_get_backend_status(backend_client_t* client) {
    return client ? client->backend_status : BACKEND_DISCONNECTED;
}

plc_status_t backend_client_get_plc_status(backend_client_t* client) {
    return client ? client->plc_status : PLC_DISCONNECTED;
}

system_status_t backend_client_get_system_status(backend_client_t* client) {
    return client ? client->system_status : SYSTEM_STOPPED;
}

bool backend_client_is_backend_running(void) {
    // 检查cpp_backend进程是否运行
    int ret = system("pgrep -f bamboo_cut_backend > /dev/null 2>&1");
    return WEXITSTATUS(ret) == 0;
}

bool backend_client_start_backend_process(void) {
    printf("启动cpp_backend后端进程\n");
    
    // 检查后端可执行文件是否存在
    if (access("../cpp_backend/build/bamboo_cut_backend", F_OK) != 0) {
        printf("错误：找不到后端可执行文件\n");
        return false;
    }
    
    // 启动后端进程
    pid_t pid = fork();
    if (pid == 0) {
        // 子进程：启动后端
        execl("../cpp_backend/build/bamboo_cut_backend", "bamboo_cut_backend", (char*)nullptr);
        printf("错误：启动后端进程失败\n");
        exit(1);
    } else if (pid > 0) {
        // 父进程：等待后端启动
        printf("后端进程已启动，PID: %d\n", pid);
        sleep(2); // 给后端时间启动
        return true;
    } else {
        printf("错误：fork失败: %s\n", strerror(errno));
        return false;
    }
}

bool backend_client_stop_backend_process(void) {
    printf("停止cpp_backend后端进程\n");
    
    // 发送SIGTERM信号
    int ret = system("pkill -TERM -f bamboo_cut_backend");
    if (WEXITSTATUS(ret) == 0) {
        printf("后端进程停止信号已发送\n");
        sleep(2); // 等待进程退出
        
        // 检查是否还在运行
        if (!backend_client_is_backend_running()) {
            printf("后端进程已正常退出\n");
            return true;
        } else {
            // 强制终止
            printf("强制终止后端进程\n");
            system("pkill -KILL -f bamboo_cut_backend");
            return true;
        }
    } else {
        printf("警告：停止后端进程失败\n");
        return false;
    }
}