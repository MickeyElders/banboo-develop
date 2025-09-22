#include "camera/shared_memory_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <sys/stat.h>
#include <errno.h>
#include <time.h>

// 获取当前时间戳（毫秒）
static uint64_t get_timestamp_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// 获取共享内存中的帧头
static shared_frame_header_t* get_frame_header(shared_memory_reader_t* reader, size_t buffer_index) {
    if (!reader->shm_ptr || buffer_index >= reader->buffer_count) {
        return nullptr;
    }
    
    size_t offset = buffer_index * (sizeof(shared_frame_header_t) + reader->frame_size);
    return (shared_frame_header_t*)((uint8_t*)reader->shm_ptr + offset);
}

// 获取共享内存中的帧数据
static uint8_t* get_frame_data(shared_memory_reader_t* reader, size_t buffer_index) {
    shared_frame_header_t* header = get_frame_header(reader, buffer_index);
    if (!header) {
        return nullptr;
    }
    
    return (uint8_t*)(header + 1);
}

// 更新统计信息
static void update_stats(shared_memory_reader_t* reader) {
    static uint64_t last_stats_time = 0;
    uint64_t current_time = get_timestamp_ms();
    
    reader->stats.total_frames_read++;
    
    // 每秒更新一次FPS
    if (last_stats_time == 0) {
        last_stats_time = current_time;
    } else if (current_time - last_stats_time >= 1000) {
        reader->stats.fps = reader->stats.total_frames_read * 1000.0 / (current_time - last_stats_time);
        last_stats_time = current_time;
    }
}

shared_memory_reader_t* shared_memory_reader_create(const char* key_path) {
    if (!key_path) {
        printf("错误：共享内存key路径为空\n");
        return nullptr;
    }
    
    shared_memory_reader_t* reader = (shared_memory_reader_t*)calloc(1, sizeof(shared_memory_reader_t));
    if (!reader) {
        printf("错误：分配共享内存读取器内存失败\n");
        return nullptr;
    }
    
    // 初始化基本参数
    strncpy(reader->key_path, key_path, sizeof(reader->key_path) - 1);
    reader->shm_key = 0;
    reader->shm_id = -1;
    reader->shm_ptr = nullptr;
    reader->shm_size = 0;
    reader->frame_size = 0;
    reader->buffer_count = 0;
    reader->last_read_frame_id = 0;
    reader->connected = false;
    reader->frame_cache_valid = false;
    
    // 初始化统计信息
    memset(&reader->stats, 0, sizeof(reader->stats));
    
    printf("创建共享内存读取器: %s\n", key_path);
    return reader;
}

void shared_memory_reader_destroy(shared_memory_reader_t* reader) {
    if (!reader) return;
    
    printf("销毁共享内存读取器\n");
    
    // 断开连接
    shared_memory_reader_disconnect(reader);
    
    free(reader);
}

bool shared_memory_reader_connect(shared_memory_reader_t* reader) {
    if (!reader) return false;
    
    if (reader->connected) {
        printf("警告：共享内存已连接\n");
        return true;
    }
    
    printf("连接到共享内存: %s\n", reader->key_path);
    
    // 生成IPC密钥
    reader->shm_key = ftok(reader->key_path, 'B');
    if (reader->shm_key == -1) {
        printf("错误：无法生成IPC密钥: %s\n", strerror(errno));
        return false;
    }
    
    // 连接到现有共享内存段
    reader->shm_id = shmget(reader->shm_key, 0, 0666);
    if (reader->shm_id == -1) {
        printf("错误：连接到共享内存段失败: %s\n", strerror(errno));
        return false;
    }
    
    // 获取共享内存段信息
    struct shmid_ds shm_info;
    if (shmctl(reader->shm_id, IPC_STAT, &shm_info) == -1) {
        printf("错误：获取共享内存信息失败: %s\n", strerror(errno));
        return false;
    }
    
    reader->shm_size = shm_info.shm_segsz;
    
    // 附加到共享内存
    reader->shm_ptr = shmat(reader->shm_id, nullptr, 0);
    if (reader->shm_ptr == (void*)-1) {
        printf("错误：附加到共享内存失败: %s\n", strerror(errno));
        reader->shm_ptr = nullptr;
        return false;
    }
    
    // 读取第一个缓冲区的头部信息
    shared_frame_header_t* header = get_frame_header(reader, 0);
    if (header && header->width > 0 && header->height > 0) {
        reader->frame_size = header->data_size;
        reader->buffer_count = reader->shm_size / (sizeof(shared_frame_header_t) + reader->frame_size);
        
        printf("连接到共享内存成功: %dx%dx%d, %zu buffers, size=%zuMB\n", 
               header->width, header->height, header->channels, 
               reader->buffer_count, reader->shm_size / (1024 * 1024));
    } else {
        printf("错误：共享内存头部信息无效\n");
        shmdt(reader->shm_ptr);
        reader->shm_ptr = nullptr;
        return false;
    }
    
    reader->connected = true;
    reader->stats.connected = true;
    
    return true;
}

void shared_memory_reader_disconnect(shared_memory_reader_t* reader) {
    if (!reader) return;
    
    if (reader->shm_ptr && reader->shm_ptr != (void*)-1) {
        shmdt(reader->shm_ptr);
        reader->shm_ptr = nullptr;
    }
    
    reader->connected = false;
    reader->stats.connected = false;
    reader->shm_id = -1;
    reader->frame_cache_valid = false;
    
    printf("共享内存连接已断开\n");
}

bool shared_memory_reader_has_new_frame(shared_memory_reader_t* reader) {
    if (!reader || !reader->connected) {
        return false;
    }
    
    for (size_t i = 0; i < reader->buffer_count; ++i) {
        shared_frame_header_t* header = get_frame_header(reader, i);
        if (header && header->ready && header->frame_id > reader->last_read_frame_id) {
            return true;
        }
    }
    
    return false;
}

bool shared_memory_reader_read_frame(shared_memory_reader_t* reader, cv::Mat& frame, int timeout_ms) {
    if (!reader || !reader->connected) {
        return false;
    }
    
    uint64_t start_time = get_timestamp_ms();
    
    while (true) {
        // 查找最新的就绪帧
        uint32_t latest_frame_id = 0;
        size_t latest_buffer_index = 0;
        bool found_new_frame = false;
        
        for (size_t i = 0; i < reader->buffer_count; ++i) {
            shared_frame_header_t* header = get_frame_header(reader, i);
            if (header && header->ready && 
                header->frame_id > reader->last_read_frame_id && 
                header->frame_id > latest_frame_id) {
                latest_frame_id = header->frame_id;
                latest_buffer_index = i;
                found_new_frame = true;
            }
        }
        
        if (found_new_frame) {
            // 读取帧数据
            shared_frame_header_t* header = get_frame_header(reader, latest_buffer_index);
            uint8_t* data = get_frame_data(reader, latest_buffer_index);
            
            if (header && data) {
                // 创建OpenCV Mat
                int cv_type = (header->channels == 3) ? CV_8UC3 : CV_8UC1;
                frame = cv::Mat(header->height, header->width, cv_type);
                memcpy(frame.data, data, header->data_size);
                
                reader->last_read_frame_id = latest_frame_id;
                reader->stats.current_frame_id = latest_frame_id;
                reader->stats.last_read_frame_id = latest_frame_id;
                
                // 更新统计
                update_stats(reader);
                
                return true;
            }
        }
        
        // 检查超时
        if (timeout_ms == 0) {
            return false; // 非阻塞模式
        }
        
        if (timeout_ms > 0) {
            uint64_t elapsed = get_timestamp_ms() - start_time;
            if (elapsed >= (uint64_t)timeout_ms) {
                return false; // 超时
            }
        }
        
        // 短暂休眠后重试
        usleep(1000); // 1ms
    }
}

bool shared_memory_reader_read_frame_to_buffer(
    shared_memory_reader_t* reader, 
    uint8_t* buffer, 
    size_t buffer_size,
    uint32_t* width,
    uint32_t* height, 
    uint32_t* channels,
    int timeout_ms) {
    
    if (!reader || !reader->connected || !buffer) {
        return false;
    }
    
    cv::Mat frame;
    if (!shared_memory_reader_read_frame(reader, frame, timeout_ms)) {
        return false;
    }
    
    if (frame.empty()) {
        return false;
    }
    
    // 检查缓冲区大小
    size_t required_size = frame.total() * frame.elemSize();
    if (buffer_size < required_size) {
        printf("错误：缓冲区大小不足: required=%zu, available=%zu\n", 
               required_size, buffer_size);
        return false;
    }
    
    // 复制数据到缓冲区
    memcpy(buffer, frame.data, required_size);
    
    // 输出尺寸信息
    if (width) *width = frame.cols;
    if (height) *height = frame.rows;
    if (channels) *channels = frame.channels();
    
    return true;
}

uint32_t shared_memory_reader_get_current_frame_id(shared_memory_reader_t* reader) {
    if (!reader || !reader->connected) {
        return 0;
    }
    
    uint32_t max_frame_id = 0;
    for (size_t i = 0; i < reader->buffer_count; ++i) {
        shared_frame_header_t* header = get_frame_header(reader, i);
        if (header && header->ready && header->frame_id > max_frame_id) {
            max_frame_id = header->frame_id;
        }
    }
    
    return max_frame_id;
}

bool shared_memory_reader_get_frame_info(
    shared_memory_reader_t* reader,
    uint32_t* width,
    uint32_t* height,
    uint32_t* channels) {
    
    if (!reader || !reader->connected) {
        return false;
    }
    
    shared_frame_header_t* header = get_frame_header(reader, 0);
    if (!header) {
        return false;
    }
    
    if (width) *width = header->width;
    if (height) *height = header->height;
    if (channels) *channels = header->channels;
    
    return true;
}

void shared_memory_reader_get_stats(shared_memory_reader_t* reader, shared_memory_stats_t* stats) {
    if (!reader || !stats) return;
    
    *stats = reader->stats;
    stats->current_frame_id = shared_memory_reader_get_current_frame_id(reader);
}

void shared_memory_reader_reset_stats(shared_memory_reader_t* reader) {
    if (!reader) return;
    
    memset(&reader->stats, 0, sizeof(reader->stats));
    reader->stats.connected = reader->connected;
}

bool shared_memory_reader_is_connected(shared_memory_reader_t* reader) {
    return reader && reader->connected;
}

bool shared_memory_reader_wait_for_availability(const char* key_path, int timeout_ms) {
    if (!key_path) return false;
    
    printf("检查共享内存可用性（非阻塞模式）: %s\n", key_path);
    
    // 立即检查一次，不等待
    key_t shm_key = ftok(key_path, 'B');
    if (shm_key != -1) {
        int shm_id = shmget(shm_key, 0, 0666);
        if (shm_id != -1) {
            void* shm_ptr = shmat(shm_id, nullptr, 0);
            if (shm_ptr != (void*)-1) {
                shared_frame_header_t* header = (shared_frame_header_t*)shm_ptr;
                if (header->width > 0 && header->height > 0) {
                    printf("共享内存立即可用: %dx%d\n", header->width, header->height);
                    shmdt(shm_ptr);
                    return true;
                }
                shmdt(shm_ptr);
            }
        }
    }
    
    printf("共享内存当前不可用，将在后台线程中继续重试\n");
    return false; // 立即返回，不阻塞UI
}