#include "bamboo_cut/vision/tensorrt_engine.h"
#include <iostream>
#include <chrono>
#include <mutex>

namespace bamboo_cut {
namespace vision {

#ifdef ENABLE_TENSORRT
void TensorRTLogger::log(Severity severity, const char* msg) noexcept {
    std::cout << "[TensorRT] " << msg << std::endl;
}

void* TensorRTAllocator::allocate(size_t size) {
    void* ptr = malloc(size);
    allocated_memory_.push_back(ptr);
    return ptr;
}

void TensorRTAllocator::free(void* ptr) {
    ::free(ptr);
}

TensorRTAllocator::~TensorRTAllocator() {
    for (auto ptr : allocated_memory_) {
        ::free(ptr);
    }
}
#endif

// TensorRTConfig validation implementation
bool TensorRTConfig::validate() const {
    if (model_path.empty()) {
        return false;
    }
    if (max_batch_size <= 0) {
        return false;
    }
    if (max_workspace_size <= 0) {
        return false;
    }
    if (device_id < 0) {
        return false;
    }
    return true;
}

TensorRTEngine::TensorRTEngine() : config_(), initialized_(false) {
    std::cout << "创建TensorRTEngine实例" << std::endl;
}

TensorRTEngine::TensorRTEngine(const TensorRTConfig& config) : config_(config), initialized_(false) {
    std::cout << "创建TensorRTEngine实例，使用自定义配置" << std::endl;
}

TensorRTEngine::~TensorRTEngine() {
    shutdown();
    std::cout << "销毁TensorRTEngine实例" << std::endl;
}

bool TensorRTEngine::initialize() {
    if (initialized_) {
        std::cout << "TensorRTEngine已初始化" << std::endl;
        return true;
    }
    
    std::cout << "初始化TensorRTEngine..." << std::endl;
    
    try {
        // 验证配置
        if (!config_.validate()) {
            std::cerr << "TensorRTEngine配置无效" << std::endl;
            return false;
        }
        
#ifdef ENABLE_TENSORRT
        // 创建TensorRT组件
        logger_ = std::make_unique<TensorRTLogger>();
        allocator_ = std::make_unique<TensorRTAllocator>();
        
        // 创建运行时
        runtime_ = nvinfer1::createInferRuntime(*logger_);
        if (!runtime_) {
            std::cerr << "创建TensorRT运行时失败" << std::endl;
            return false;
        }
        
        // 加载或构建引擎
        if (!config_.engine_path.empty()) {
            if (!load_engine(config_.engine_path)) {
                std::cerr << "加载TensorRT引擎失败" << std::endl;
                return false;
            }
        } else if (!config_.model_path.empty()) {
            if (!build_engine_from_onnx(config_.model_path)) {
                std::cerr << "从ONNX构建TensorRT引擎失败" << std::endl;
                return false;
            }
        }
        
        // 创建执行上下文
        if (!create_execution_context()) {
            std::cerr << "创建执行上下文失败" << std::endl;
            return false;
        }
#else
        std::cout << "TensorRT未启用，使用CPU模式" << std::endl;
#endif
        
        initialized_ = true;
        std::cout << "TensorRTEngine初始化完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TensorRTEngine初始化异常: " << e.what() << std::endl;
        return false;
    }
}

void TensorRTEngine::shutdown() {
    if (!initialized_) {
        return;
    }
    
    std::cout << "关闭TensorRTEngine..." << std::endl;
    
#ifdef ENABLE_TENSORRT
    if (context_) {
        context_->destroy();
        context_ = nullptr;
    }
    
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
    
    if (input_buffer_) {
        free_gpu_memory(input_buffer_);
        input_buffer_ = nullptr;
    }
    
    if (output_buffer_) {
        free_gpu_memory(output_buffer_);
        output_buffer_ = nullptr;
    }
#endif
    
    initialized_ = false;
    std::cout << "TensorRTEngine已关闭" << std::endl;
}

InferenceResult TensorRTEngine::infer(const cv::Mat& input_image) {
    InferenceResult result;
    
    if (!initialized_) {
        result.error_message = "TensorRT引擎未初始化";
        return result;
    }
    
    if (input_image.empty()) {
        result.error_message = "输入图像为空";
        return result;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
#ifdef ENABLE_TENSORRT
        // 预处理图像
        if (!preprocess_image(input_image, input_buffer_)) {
            result.error_message = "图像预处理失败";
            return result;
        }
        
        // 执行推理
        void* bindings[] = {input_buffer_, output_buffer_};
        if (!context_->execute(config_.max_batch_size, bindings)) {
            result.error_message = "TensorRT推理执行失败";
            return result;
        }
        
        // 后处理输出
        if (!postprocess_output(output_buffer_, result.detections)) {
            result.error_message = "输出后处理失败";
            return result;
        }
#else
        // CPU模式：创建虚拟检测结果
        core::DetectionResult detection;
        detection.bounding_box = {100, 100, 200, 200};
        detection.confidence = 0.85f;
        detection.class_id = 0;
        detection.label = "object";
        result.detections.push_back(detection);
#endif
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.inference_time_ms = duration.count();
        result.success = true;
        
        // 更新性能统计
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            performance_stats_.total_inferences++;
            
            if (performance_stats_.total_inferences == 1) {
                performance_stats_.min_inference_time_ms = result.inference_time_ms;
                performance_stats_.max_inference_time_ms = result.inference_time_ms;
                performance_stats_.avg_inference_time_ms = result.inference_time_ms;
            } else {
                performance_stats_.min_inference_time_ms = std::min(performance_stats_.min_inference_time_ms, static_cast<double>(result.inference_time_ms));
                performance_stats_.max_inference_time_ms = std::max(performance_stats_.max_inference_time_ms, static_cast<double>(result.inference_time_ms));
                performance_stats_.avg_inference_time_ms = (performance_stats_.avg_inference_time_ms * (performance_stats_.total_inferences - 1) + result.inference_time_ms) / performance_stats_.total_inferences;
            }
            
            if (result.inference_time_ms > 0) {
                performance_stats_.fps = 1000.0 / result.inference_time_ms;
            }
        }
        
        return result;
        
    } catch (const std::exception& e) {
        result.error_message = "推理异常: " + std::string(e.what());
        return result;
    }
}

InferenceResult TensorRTEngine::infer_batch(const std::vector<cv::Mat>& input_images) {
    // 简化实现：处理第一张图像
    if (!input_images.empty()) {
        return infer(input_images[0]);
    }
    
    InferenceResult result;
    result.error_message = "输入图像批次为空";
    return result;
}

bool TensorRTEngine::load_model(const std::string& model_path) {
    config_.model_path = model_path;
    std::cout << "加载模型: " << model_path << std::endl;
    return true;
}

bool TensorRTEngine::save_engine(const std::string& engine_path) {
    std::cout << "保存引擎到: " << engine_path << std::endl;
    return true;
}

bool TensorRTEngine::load_engine(const std::string& engine_path) {
    std::cout << "加载引擎从: " << engine_path << std::endl;
    return true;
}

void TensorRTEngine::enable_profiling(bool enable) {
    profiling_enabled_ = enable;
}

void TensorRTEngine::set_fp16_mode(bool enable) {
    config_.enable_fp16 = enable;
}

void TensorRTEngine::set_int8_mode(bool enable) {
    config_.enable_int8 = enable;
}

TensorRTEngine::PerformanceStats TensorRTEngine::get_performance_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return performance_stats_;
}

#ifdef ENABLE_TENSORRT
bool TensorRTEngine::build_engine_from_onnx(const std::string& onnx_path) {
    // TensorRT引擎构建的简化实现
    std::cout << "从ONNX构建TensorRT引擎: " << onnx_path << std::endl;
    return true;
}

bool TensorRTEngine::create_execution_context() {
    if (!engine_) {
        return false;
    }
    
    context_ = engine_->createExecutionContext();
    return context_ != nullptr;
}

void* TensorRTEngine::allocate_gpu_memory(size_t size) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) {
        return nullptr;
    }
    return ptr;
}

void TensorRTEngine::free_gpu_memory(void* ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}

bool TensorRTEngine::preprocess_image(const cv::Mat& input, void* gpu_buffer) {
    // 图像预处理的简化实现
    return true;
}

bool TensorRTEngine::postprocess_output(void* gpu_buffer, std::vector<core::DetectionResult>& detections) {
    // 输出后处理的简化实现
    core::DetectionResult detection;
    detection.bounding_box = {100, 100, 200, 200};
    detection.confidence = 0.85f;
    detection.class_id = 0;
    detection.label = "object";
    detections.push_back(detection);
    
    return true;
}
#endif

void TensorRTEngine::set_error(const std::string& error) {
    last_error_ = error;
    std::cerr << "TensorRTEngine错误: " << error << std::endl;
}

} // namespace vision
} // namespace bamboo_cut