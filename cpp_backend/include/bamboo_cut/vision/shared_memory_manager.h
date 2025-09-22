#ifndef BAMBOO_CUT_SHARED_MEMORY_MANAGER_H
#define BAMBOO_CUT_SHARED_MEMORY_MANAGER_H

#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <opencv2/opencv.hpp>

namespace bamboo_cut {
namespace vision {

// 共享内存帧头结构
struct SharedFrameHeader {
    uint32_t frame_id;          // 帧ID
    uint64_t timestamp;         // 时间戳（毫秒）
    uint32_t width;             // 图像宽度
    uint32_t height;            // 图像高度
    uint32_t channels;          // 通道数
    uint32_t data_size;         // 数据大小
    uint8_t  format;            // 图像格式 (0=BGR, 1=RGB, 2=YUV)
    uint8_t  ready;             // 数据就绪标志 (0=写入中, 1=就绪)
    uint8_t  reserved[2];       // 预留字节对齐
    // 实际图像数据紧跟在头部后面
};

// 共享内存状态统计
struct SharedMemoryStats {
    uint64_t total_frames_written;
    uint64_t total_frames_read;
    uint64_t frames_dropped;
    double   fps;
    uint32_t current_readers;
    uint32_t max_readers;
};

class SharedMemoryManager {
public:
    explicit SharedMemoryManager(const std::string& shared_mem_key);
    virtual ~SharedMemoryManager();

    // 初始化共享内存（生产者调用）
    bool initialize(int width, int height, int channels = 3, size_t buffer_count = 2);
    
    // 连接到已存在的共享内存（消费者调用）
    bool connect();
    
    // 断开连接
    void disconnect();
    
    // 写入帧数据（生产者调用）
    bool writeFrame(const cv::Mat& frame);
    
    // 读取帧数据（消费者调用）
    bool readFrame(cv::Mat& frame, int timeout_ms = -1);
    
    // 检查是否有新帧可读
    bool hasNewFrame() const;
    
    // 获取当前帧ID
    uint32_t getCurrentFrameId() const;
    
    // 获取共享内存状态统计
    SharedMemoryStats getStats() const;
    
    // 重置统计信息
    void resetStats();
    
    // 检查共享内存是否就绪
    bool isReady() const { return initialized_; }
    
    // 获取共享内存大小
    size_t getSharedMemorySize() const;

private:
    std::string shared_mem_key_;
    key_t shm_key_;
    int shm_id_;
    void* shm_ptr_;
    size_t shm_size_;
    bool initialized_;
    bool is_producer_;
    
    // 帧缓冲区管理
    size_t frame_size_;
    size_t buffer_count_;
    uint32_t write_index_;
    uint32_t last_read_frame_id_;
    
    // 线程同步
    mutable std::mutex mutex_;
    std::condition_variable frame_ready_cv_;
    
    // 性能统计
    mutable SharedMemoryStats stats_;
    mutable std::mutex stats_mutex_;
    std::chrono::steady_clock::time_point last_stats_time_;
    
    // 内部方法
    SharedFrameHeader* getFrameHeader(size_t buffer_index) const;
    uint8_t* getFrameData(size_t buffer_index) const;
    size_t calculateSharedMemorySize(int width, int height, int channels, size_t buffer_count);
    void updateStats(bool is_write_operation);
    bool waitForFrameReady(uint32_t target_frame_id, int timeout_ms);
};

// 共享内存工厂类
class SharedMemoryFactory {
public:
    // 创建生产者（后端使用）
    static std::unique_ptr<SharedMemoryManager> createProducer(
        const std::string& key, int width, int height, int channels = 3);
        
    // 创建消费者（前端使用）
    static std::unique_ptr<SharedMemoryManager> createConsumer(const std::string& key);
    
    // 清理共享内存段
    static bool cleanup(const std::string& key);
    
    // 检查共享内存段是否存在
    static bool exists(const std::string& key);
};

} // namespace vision
} // namespace bamboo_cut

#endif // BAMBOO_CUT_SHARED_MEMORY_MANAGER_H