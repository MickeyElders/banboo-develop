/**
 * 共享内存架构测试程序
 * 验证后端和前端之间的共享内存通信
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <signal.h>
#include <opencv2/opencv.hpp>

// 后端头文件
#include "cpp_backend/include/bamboo_cut/vision/shared_memory_manager.h"
#include "cpp_backend/include/bamboo_cut/vision/camera_manager.h"

// 前端头文件  
#include "lvgl_frontend/include/camera/shared_memory_reader.h"

using namespace bamboo_cut::vision;

std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "接收到信号 " << signal << "，停止测试..." << std::endl;
    g_running = false;
}

// 模拟后端生产者
void backend_producer_test() {
    std::cout << "=== 后端生产者测试 ===" << std::endl;
    
    // 创建共享内存管理器（生产者）
    auto producer = SharedMemoryFactory::createProducer("/tmp/bamboo_camera_shm", 640, 480, 3);
    if (!producer) {
        std::cerr << "创建共享内存生产者失败" << std::endl;
        return;
    }
    
    std::cout << "共享内存生产者创建成功" << std::endl;
    
    // 生成测试图像
    cv::Mat test_image(480, 640, CV_8UC3);
    uint32_t frame_id = 0;
    
    while (g_running) {
        // 生成渐变测试图像
        for (int y = 0; y < test_image.rows; ++y) {
            for (int x = 0; x < test_image.cols; ++x) {
                auto& pixel = test_image.at<cv::Vec3b>(y, x);
                pixel[0] = (frame_id + x) % 256;      // Blue
                pixel[1] = (frame_id + y) % 256;      // Green  
                pixel[2] = (frame_id + x + y) % 256;  // Red
            }
        }
        
        // 添加帧ID文本
        std::string frame_text = "Frame: " + std::to_string(frame_id);
        cv::putText(test_image, frame_text, cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
        
        // 写入共享内存
        if (producer->writeFrame(test_image)) {
            std::cout << "写入帧 " << frame_id << " 成功" << std::endl;
        } else {
            std::cerr << "写入帧 " << frame_id << " 失败" << std::endl;
        }
        
        frame_id++;
        
        // 控制帧率（30fps）
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
    
    std::cout << "后端生产者测试结束" << std::endl;
}

// 模拟前端消费者
void frontend_consumer_test() {
    std::cout << "=== 前端消费者测试 ===" << std::endl;
    
    // 等待共享内存可用
    if (!shared_memory_reader_wait_for_availability("/tmp/bamboo_camera_shm", 10000)) {
        std::cerr << "等待共享内存超时" << std::endl;
        return;
    }
    
    // 创建共享内存读取器
    auto reader = shared_memory_reader_create("/tmp/bamboo_camera_shm");
    if (!reader) {
        std::cerr << "创建共享内存读取器失败" << std::endl;
        return;
    }
    
    // 连接到共享内存
    if (!shared_memory_reader_connect(reader)) {
        std::cerr << "连接到共享内存失败" << std::endl;
        shared_memory_reader_destroy(reader);
        return;
    }
    
    std::cout << "共享内存读取器连接成功" << std::endl;
    
    cv::Mat received_frame;
    uint32_t last_frame_id = 0;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (g_running) {
        // 读取帧
        if (shared_memory_reader_read_frame(reader, received_frame, 100)) {
            uint32_t current_frame_id = shared_memory_reader_get_current_frame_id(reader);
            
            if (current_frame_id > last_frame_id) {
                frame_count++;
                last_frame_id = current_frame_id;
                
                std::cout << "接收帧 " << current_frame_id 
                         << ", 尺寸: " << received_frame.cols << "x" << received_frame.rows
                         << ", 通道: " << received_frame.channels() << std::endl;
                
                // 每5秒输出一次统计信息
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
                if (elapsed.count() >= 5) {
                    shared_memory_stats_t stats;
                    shared_memory_reader_get_stats(reader, &stats);
                    
                    std::cout << "=== 5秒统计 ===" << std::endl;
                    std::cout << "接收帧数: " << frame_count << std::endl;
                    std::cout << "平均FPS: " << (frame_count / 5.0) << std::endl;
                    std::cout << "共享内存FPS: " << stats.fps << std::endl;
                    std::cout << "连接状态: " << (stats.connected ? "已连接" : "已断开") << std::endl;
                    
                    frame_count = 0;
                    start_time = now;
                }
            }
        } else {
            // 读取失败，检查连接状态
            if (!shared_memory_reader_is_connected(reader)) {
                std::cerr << "共享内存连接丢失" << std::endl;
                break;
            }
        }
        
        // 短暂休眠
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // 清理
    shared_memory_reader_destroy(reader);
    std::cout << "前端消费者测试结束" << std::endl;
}

int main() {
    std::cout << "=== 共享内存架构测试程序 ===" << std::endl;
    std::cout << "测试后端和前端之间的共享内存通信" << std::endl;
    
    // 设置信号处理
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // 启动后端生产者线程
    std::thread producer_thread(backend_producer_test);
    
    // 等待一秒钟让生产者启动
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // 启动前端消费者线程
    std::thread consumer_thread(frontend_consumer_test);
    
    // 等待线程结束
    producer_thread.join();
    consumer_thread.join();
    
    // 清理共享内存
    SharedMemoryFactory::cleanup("/tmp/bamboo_camera_shm");
    
    std::cout << "测试程序结束" << std::endl;
    return 0;
}