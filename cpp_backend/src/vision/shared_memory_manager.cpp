#include "bamboo_cut/vision/shared_memory_manager.h"
#include "bamboo_cut/core/logger.h"
#include <cstring>
#include <chrono>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

namespace bamboo_cut {
namespace vision {

SharedMemoryManager::SharedMemoryManager(const std::string& shared_mem_key)
    : shared_mem_key_(shared_mem_key)
    , shm_key_(0)
    , shm_id_(-1)
    , shm_ptr_(nullptr)
    , shm_size_(0)
    , initialized_(false)
    , is_producer_(false)
    , frame_size_(0)
    , buffer_count_(0)
    , write_index_(0)
    , last_read_frame_id_(0)
    , last_stats_time_(std::chrono::steady_clock::now()) {
    
    // 生成IPC密钥
    shm_key_ = ftok(shared_mem_key_.c_str(), 'B');
    if (shm_key_ == -1) {
        // 如果文件不存在，创建一个临时文件
        int fd = creat(shared_mem_key_.c_str(), 0666);
        if (fd != -1) {
            close(fd);
            shm_key_ = ftok(shared_mem_key_.c_str(), 'B');
        }
    }
    
    // 初始化统计信息
    memset(&stats_, 0, sizeof(stats_));
}

SharedMemoryManager::~SharedMemoryManager() {
    disconnect();
}

bool SharedMemoryManager::initialize(int width, int height, int channels, size_t buffer_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        LOG_WARN("共享内存已经初始化");
        return true;
    }
    
    if (shm_key_ == -1) {
        LOG_ERROR("无法生成IPC密钥");
        return false;
    }
    
    buffer_count_ = buffer_count;
    frame_size_ = width * height * channels;
    shm_size_ = calculateSharedMemorySize(width, height, channels, buffer_count);
    
    LOG_INFO("初始化共享内存: key={}, size={}MB", shm_key_, shm_size_ / (1024 * 1024));
    
    // 创建共享内存段
    shm_id_ = shmget(shm_key_, shm_size_, IPC_CREAT | IPC_EXCL | 0666);
    if (shm_id_ == -1) {
        if (errno == EEXIST) {
            // 共享内存已存在，尝试连接
            LOG_WARN("共享内存段已存在，尝试重新连接");
            shm_id_ = shmget(shm_key_, shm_size_, 0666);
            if (shm_id_ == -1) {
                LOG_ERROR("连接到现有共享内存段失败: {}", strerror(errno));
                return false;
            }
        } else {
            LOG_ERROR("创建共享内存段失败: {}", strerror(errno));
            return false;
        }
    }
    
    // 附加到共享内存
    shm_ptr_ = shmat(shm_id_, nullptr, 0);
    if (shm_ptr_ == (void*)-1) {
        LOG_ERROR("附加到共享内存失败: {}", strerror(errno));
        shmctl(shm_id_, IPC_RMID, nullptr);
        return false;
    }
    
    // 初始化共享内存内容
    memset(shm_ptr_, 0, shm_size_);
    
    // 初始化每个缓冲区的头部
    for (size_t i = 0; i < buffer_count_; ++i) {
        SharedFrameHeader* header = getFrameHeader(i);
        header->width = width;
        header->height = height;
        header->channels = channels;
        header->data_size = frame_size_;
        header->format = 0; // BGR
        header->ready = 0;
        header->frame_id = 0;
        header->timestamp = 0;
    }
    
    is_producer_ = true;
    initialized_ = true;
    
    LOG_INFO("共享内存初始化成功: {}x{}x{}, {} buffers", width, height, channels, buffer_count);
    return true;
}

bool SharedMemoryManager::connect() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (initialized_) {
        LOG_WARN("共享内存已经连接");
        return true;
    }
    
    if (shm_key_ == -1) {
        LOG_ERROR("无法生成IPC密钥");
        return false;
    }
    
    // 连接到现有共享内存段
    shm_id_ = shmget(shm_key_, 0, 0666);
    if (shm_id_ == -1) {
        LOG_ERROR("连接到共享内存段失败: {}", strerror(errno));
        return false;
    }
    
    // 获取共享内存段信息
    struct shmid_ds shm_info;
    if (shmctl(shm_id_, IPC_STAT, &shm_info) == -1) {
        LOG_ERROR("获取共享内存信息失败: {}", strerror(errno));
        return false;
    }
    
    shm_size_ = shm_info.shm_segsz;
    
    // 附加到共享内存
    shm_ptr_ = shmat(shm_id_, nullptr, 0);
    if (shm_ptr_ == (void*)-1) {
        LOG_ERROR("附加到共享内存失败: {}", strerror(errno));
        return false;
    }
    
    // 读取第一个缓冲区的头部信息
    SharedFrameHeader* header = getFrameHeader(0);
    if (header->width > 0 && header->height > 0) {
        frame_size_ = header->data_size;
        buffer_count_ = shm_size_ / (sizeof(SharedFrameHeader) + frame_size_);
        
        LOG_INFO("连接到共享内存: {}x{}x{}, {} buffers, size={}MB", 
                header->width, header->height, header->channels, 
                buffer_count_, shm_size_ / (1024 * 1024));
    }
    
    is_producer_ = false;
    initialized_ = true;
    
    return true;
}

void SharedMemoryManager::disconnect() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (shm_ptr_ && shm_ptr_ != (void*)-1) {
        shmdt(shm_ptr_);
        shm_ptr_ = nullptr;
    }
    
    if (is_producer_ && shm_id_ != -1) {
        // 只有生产者负责删除共享内存段
        shmctl(shm_id_, IPC_RMID, nullptr);
        LOG_INFO("共享内存段已删除");
    }
    
    initialized_ = false;
    shm_id_ = -1;
}

bool SharedMemoryManager::writeFrame(const cv::Mat& frame) {
    if (!initialized_ || !is_producer_) {
        return false;
    }
    
    if (frame.empty() || 
        static_cast<size_t>(frame.total() * frame.elemSize()) != frame_size_) {
        LOG_WARN("帧尺寸不匹配: expected={}, actual={}", 
                frame_size_, frame.total() * frame.elemSize());
        return false;
    }
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 获取当前写入缓冲区
    size_t buffer_index = write_index_ % buffer_count_;
    SharedFrameHeader* header = getFrameHeader(buffer_index);
    uint8_t* data = getFrameData(buffer_index);
    
    // 标记正在写入
    header->ready = 0;
    
    // 写入帧数据
    memcpy(data, frame.data, frame_size_);
    
    // 更新头部信息
    header->frame_id = ++write_index_;
    header->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    
    // 根据OpenCV Mat类型设置格式
    if (frame.channels() == 3) {
        header->format = 0; // BGR
    } else if (frame.channels() == 1) {
        header->format = 2; // Grayscale
    }
    
    // 标记数据就绪
    header->ready = 1;
    
    // 通知等待的读取线程
    frame_ready_cv_.notify_all();
    
    // 更新统计
    updateStats(true);
    
    return true;
}

bool SharedMemoryManager::readFrame(cv::Mat& frame, int timeout_ms) {
    if (!initialized_) {
        return false;
    }
    
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 查找最新的就绪帧
    uint32_t latest_frame_id = 0;
    size_t latest_buffer_index = 0;
    bool found_new_frame = false;
    
    for (size_t i = 0; i < buffer_count_; ++i) {
        SharedFrameHeader* header = getFrameHeader(i);
        if (header->ready && header->frame_id > last_read_frame_id_ && 
            header->frame_id > latest_frame_id) {
            latest_frame_id = header->frame_id;
            latest_buffer_index = i;
            found_new_frame = true;
        }
    }
    
    // 如果没有新帧，等待
    if (!found_new_frame && timeout_ms != 0) {
        if (timeout_ms > 0) {
            auto timeout = std::chrono::milliseconds(timeout_ms);
            frame_ready_cv_.wait_for(lock, timeout, [this]() {
                return hasNewFrame();
            });
        } else {
            frame_ready_cv_.wait(lock, [this]() {
                return hasNewFrame();
            });
        }
        
        // 重新查找最新帧
        latest_frame_id = 0;
        found_new_frame = false;
        for (size_t i = 0; i < buffer_count_; ++i) {
            SharedFrameHeader* header = getFrameHeader(i);
            if (header->ready && header->frame_id > last_read_frame_id_ && 
                header->frame_id > latest_frame_id) {
                latest_frame_id = header->frame_id;
                latest_buffer_index = i;
                found_new_frame = true;
            }
        }
    }
    
    if (!found_new_frame) {
        return false;
    }
    
    // 读取帧数据
    SharedFrameHeader* header = getFrameHeader(latest_buffer_index);
    uint8_t* data = getFrameData(latest_buffer_index);
    
    // 创建OpenCV Mat
    int cv_type = (header->channels == 3) ? CV_8UC3 : CV_8UC1;
    frame = cv::Mat(header->height, header->width, cv_type);
    memcpy(frame.data, data, header->data_size);
    
    last_read_frame_id_ = latest_frame_id;
    
    // 更新统计
    updateStats(false);
    
    return true;
}

bool SharedMemoryManager::hasNewFrame() const {
    for (size_t i = 0; i < buffer_count_; ++i) {
        SharedFrameHeader* header = getFrameHeader(i);
        if (header->ready && header->frame_id > last_read_frame_id_) {
            return true;
        }
    }
    return false;
}

uint32_t SharedMemoryManager::getCurrentFrameId() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint32_t max_frame_id = 0;
    for (size_t i = 0; i < buffer_count_; ++i) {
        SharedFrameHeader* header = getFrameHeader(i);
        if (header->ready && header->frame_id > max_frame_id) {
            max_frame_id = header->frame_id;
        }
    }
    return max_frame_id;
}

SharedMemoryStats SharedMemoryManager::getStats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void SharedMemoryManager::resetStats() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    memset(&stats_, 0, sizeof(stats_));
    last_stats_time_ = std::chrono::steady_clock::now();
}

size_t SharedMemoryManager::getSharedMemorySize() const {
    return shm_size_;
}

SharedFrameHeader* SharedMemoryManager::getFrameHeader(size_t buffer_index) const {
    if (!shm_ptr_ || buffer_index >= buffer_count_) {
        return nullptr;
    }
    
    size_t offset = buffer_index * (sizeof(SharedFrameHeader) + frame_size_);
    return reinterpret_cast<SharedFrameHeader*>(
        static_cast<uint8_t*>(shm_ptr_) + offset);
}

uint8_t* SharedMemoryManager::getFrameData(size_t buffer_index) const {
    SharedFrameHeader* header = getFrameHeader(buffer_index);
    if (!header) {
        return nullptr;
    }
    
    return reinterpret_cast<uint8_t*>(header + 1);
}

size_t SharedMemoryManager::calculateSharedMemorySize(int width, int height, int channels, size_t buffer_count) {
    size_t frame_size = width * height * channels;
    size_t single_buffer_size = sizeof(SharedFrameHeader) + frame_size;
    return single_buffer_size * buffer_count;
}

void SharedMemoryManager::updateStats(bool is_write_operation) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_stats_time_);
    
    if (is_write_operation) {
        stats_.total_frames_written++;
    } else {
        stats_.total_frames_read++;
    }
    
    // 每秒更新一次FPS
    if (elapsed.count() >= 1000) {
        if (is_write_operation) {
            stats_.fps = stats_.total_frames_written * 1000.0 / elapsed.count();
        }
        last_stats_time_ = now;
    }
}

bool SharedMemoryManager::waitForFrameReady(uint32_t target_frame_id, int timeout_ms) {
    if (timeout_ms <= 0) {
        return hasNewFrame();
    }
    
    auto start = std::chrono::steady_clock::now();
    while (true) {
        if (getCurrentFrameId() >= target_frame_id) {
            return true;
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        if (elapsed.count() >= timeout_ms) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// SharedMemoryFactory implementation
std::unique_ptr<SharedMemoryManager> SharedMemoryFactory::createProducer(
    const std::string& key, int width, int height, int channels) {
    
    auto manager = std::make_unique<SharedMemoryManager>(key);
    if (manager->initialize(width, height, channels)) {
        LOG_INFO("共享内存生产者创建成功: {}", key);
        return manager;
    }
    
    LOG_ERROR("共享内存生产者创建失败: {}", key);
    return nullptr;
}

std::unique_ptr<SharedMemoryManager> SharedMemoryFactory::createConsumer(const std::string& key) {
    auto manager = std::make_unique<SharedMemoryManager>(key);
    if (manager->connect()) {
        LOG_INFO("共享内存消费者连接成功: {}", key);
        return manager;
    }
    
    LOG_ERROR("共享内存消费者连接失败: {}", key);
    return nullptr;
}

bool SharedMemoryFactory::cleanup(const std::string& key) {
    key_t shm_key = ftok(key.c_str(), 'B');
    if (shm_key == -1) {
        return false;
    }
    
    int shm_id = shmget(shm_key, 0, 0666);
    if (shm_id == -1) {
        return true; // 不存在，认为清理成功
    }
    
    return shmctl(shm_id, IPC_RMID, nullptr) == 0;
}

bool SharedMemoryFactory::exists(const std::string& key) {
    key_t shm_key = ftok(key.c_str(), 'B');
    if (shm_key == -1) {
        return false;
    }
    
    int shm_id = shmget(shm_key, 0, 0666);
    return shm_id != -1;
}

} // namespace vision
} // namespace bamboo_cut