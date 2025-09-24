/**
 * 工具函数实现
 */

#include "inference_core.h"
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>

namespace BambooUtils {

CudaDeviceInfo getCudaDeviceInfo() {
    CudaDeviceInfo info;
    
    // 获取CUDA设备数量
    cudaGetDeviceCount(&info.device_count);
    
    if (info.device_count > 0) {
        // 获取当前设备
        cudaGetDevice(&info.current_device);
        
        // 获取设备属性
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, info.current_device);
        
        info.device_name = prop.name;
        info.total_memory = prop.totalGlobalMem;
        info.compute_capability_major = prop.major;
        info.compute_capability_minor = prop.minor;
        
        // 获取可用内存
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory = free_mem;
    }
    
    return info;
}

std::string getTensorRTVersion() {
    std::stringstream ss;
    ss << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH;
    return ss.str();
}

std::string getOpenCVVersion() {
    return CV_VERSION;
}

bool setupLogging(const std::string& log_level) {
    // 简单的日志设置实现
    if (log_level == "DEBUG" || log_level == "INFO" || log_level == "WARNING" || log_level == "ERROR") {
        std::cout << "[Logger] 设置日志级别为: " << log_level << std::endl;
        return true;
    }
    return false;
}

} // namespace BambooUtils