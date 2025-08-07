#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace bamboo_cut {
namespace core {

// 基础类型定义
using Timestamp = std::chrono::steady_clock::time_point;
using Duration = std::chrono::milliseconds;

// 错误码定义
enum class ErrorCode : int32_t {
    SUCCESS = 0,
    INVALID_PARAMETER = -1,
    INITIALIZATION_FAILED = -2,
    RESOURCE_NOT_FOUND = -3,
    OPERATION_TIMEOUT = -4,
    COMMUNICATION_ERROR = -5,
    HARDWARE_ERROR = -6,
    SYSTEM_ERROR = -7
};

// 系统状态枚举
enum class SystemState : uint16_t {
    UNINITIALIZED = 0,
    INITIALIZING = 1,
    READY = 2,
    RUNNING = 3,
    PAUSED = 4,
    ERROR = 5,
    EMERGENCY_STOP = 6,
    SHUTDOWN = 7
};

// 坐标点结构
struct Point2D {
    float x{0.0f};
    float y{0.0f};
    
    Point2D() = default;
    Point2D(float x, float y) : x(x), y(y) {}
};

struct Point3D {
    float x{0.0f};
    float y{0.0f};
    float z{0.0f};
    float confidence{0.0f};
    
    Point3D() = default;
    Point3D(float x, float y, float z, float conf = 0.0f) 
        : x(x), y(y), z(z), confidence(conf) {}
};

// 矩形区域结构
struct Rectangle {
    float x{0.0f};
    float y{0.0f};
    float width{0.0f};
    float height{0.0f};
    
    Rectangle() = default;
    Rectangle(float x, float y, float w, float h) 
        : x(x), y(y), width(w), height(h) {}
};

// 检测结果结构
struct DetectionResult {
    Point2D center;
    Rectangle bounding_box;
    float confidence{0.0f};
    std::string label;
    uint32_t class_id{0};
    
    DetectionResult() = default;
};

// 性能统计结构
struct PerformanceStats {
    uint64_t total_frames{0};
    uint64_t processed_frames{0};
    uint64_t detection_count{0};
    double avg_processing_time_ms{0.0};
    double fps{0.0};
    Timestamp last_update;
    
    PerformanceStats() = default;
};

// 配置基类
class ConfigBase {
public:
    virtual ~ConfigBase() = default;
    virtual bool validate() const = 0;
    virtual std::string to_string() const = 0;
};

// 智能指针类型别名
template<typename T>
using UniquePtr = std::unique_ptr<T>;

template<typename T>
using SharedPtr = std::shared_ptr<T>;

template<typename T>
using WeakPtr = std::weak_ptr<T>;

} // namespace core
} // namespace bamboo_cut 