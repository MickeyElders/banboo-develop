/**
 * C++推理核心模块
 * 高性能TensorRT推理引擎实现
 */

#include "inference_core.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <fstream>
#include <memory>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

using namespace nvinfer1;
using namespace cv;

// Logger for TensorRT
class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// 构造函数
BambooDetector::BambooDetector() 
    : engine_(nullptr), context_(nullptr), stream_(nullptr), 
      input_binding_index_(-1), output_binding_index_(-1),
      initialized_(false) {
    
    // 初始化CUDA流
    cudaStreamCreate(&stream_);
}

// 析构函数
BambooDetector::~BambooDetector() {
    cleanup();
}

// 初始化检测器
bool BambooDetector::initialize(const DetectorConfig& config) {
    config_ = config;
    
    try {
        std::cout << "[BambooDetector] 初始化检测器..." << std::endl;
        
        // 如果引擎文件存在，直接加载
        if (std::filesystem::exists(config.engine_path)) {
            std::cout << "[BambooDetector] 加载现有引擎文件: " << config.engine_path << std::endl;
            if (!loadEngine(config.engine_path)) {
                return false;
            }
        } else {
            // 从ONNX模型构建引擎
            std::cout << "[BambooDetector] 从ONNX模型构建引擎: " << config.model_path << std::endl;
            if (!buildEngineFromOnnx(config.model_path, config.engine_path)) {
                return false;
            }
        }
        
        // 创建执行上下文
        context_ = engine_->createExecutionContext();
        if (!context_) {
            std::cerr << "[BambooDetector] 创建执行上下文失败" << std::endl;
            return false;
        }
        
        // 获取绑定索引 - TensorRT 10.x 兼容
        const char* input_name = "images";
        const char* output_name = "output";
        
        input_binding_index_ = -1;
        output_binding_index_ = -1;
        
        // 尝试新API
        int32_t nbIOTensors = engine_->getNbIOTensors();
        for (int32_t i = 0; i < nbIOTensors; ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (name && strcmp(name, input_name) == 0) {
                input_binding_index_ = i;
            } else if (name && strcmp(name, output_name) == 0) {
                output_binding_index_ = i;
            }
        }
        
        if (input_binding_index_ == -1 || output_binding_index_ == -1) {
            std::cerr << "[BambooDetector] 获取绑定索引失败" << std::endl;
            return false;
        }
        
        // 分配GPU内存
        if (!allocateBuffers()) {
            return false;
        }
        
        // 预热模型
        if (!warmupModel()) {
            return false;
        }
        
        initialized_ = true;
        std::cout << "[BambooDetector] 检测器初始化成功" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 初始化异常: " << e.what() << std::endl;
        return false;
    }
}

// 从ONNX构建引擎
bool BambooDetector::buildEngineFromOnnx(const std::string& onnx_path, const std::string& engine_path) {
    try {
        auto builder = std::unique_ptr<IBuilder>(createInferBuilder(gLogger));
        if (!builder) {
            std::cerr << "[BambooDetector] 创建构建器失败" << std::endl;
            return false;
        }
        
        // 创建网络 - 使用新API
        auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(0U));
        if (!network) {
            std::cerr << "[BambooDetector] 创建网络失败" << std::endl;
            return false;
        }
        
        // 创建ONNX解析器
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            std::cerr << "[BambooDetector] 创建ONNX解析器失败" << std::endl;
            return false;
        }
        
        // 解析ONNX文件
        if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
            std::cerr << "[BambooDetector] ONNX解析失败" << std::endl;
            return false;
        }
        
        // 创建构建配置
        auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            std::cerr << "[BambooDetector] 创建构建配置失败" << std::endl;
            return false;
        }
        
        // 设置内存池大小 - TensorRT 10.x 新API
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30); // 1GB
        
        // 启用FP16精度
        if (config_.use_fp16 && builder->platformHasFastFp16()) {
            config->setFlag(BuilderFlag::kFP16);
            std::cout << "[BambooDetector] 启用FP16精度" << std::endl;
        }
        
        // 构建引擎
        auto serializedEngine = std::unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!serializedEngine) {
            std::cerr << "[BambooDetector] 构建引擎失败" << std::endl;
            return false;
        }
        
        // 保存引擎文件
        std::ofstream engineFile(engine_path, std::ios::binary);
        if (!engineFile) {
            std::cerr << "[BambooDetector] 无法创建引擎文件: " << engine_path << std::endl;
            return false;
        }
        
        engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
        engineFile.close();
        
        // 创建运行时并反序列化引擎
        auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        engine_ = runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size());
        
        if (!engine_) {
            std::cerr << "[BambooDetector] 反序列化引擎失败" << std::endl;
            return false;
        }
        
        std::cout << "[BambooDetector] 引擎构建并保存成功: " << engine_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 构建引擎异常: " << e.what() << std::endl;
        return false;
    }
}

// 加载引擎
bool BambooDetector::loadEngine(const std::string& engine_path) {
    try {
        // 读取引擎文件
        std::ifstream engineFile(engine_path, std::ios::binary);
        if (!engineFile) {
            std::cerr << "[BambooDetector] 无法打开引擎文件: " << engine_path << std::endl;
            return false;
        }
        
        engineFile.seekg(0, std::ios::end);
        size_t engineSize = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        
        std::vector<char> engineData(engineSize);
        engineFile.read(engineData.data(), engineSize);
        engineFile.close();
        
        // 创建运行时并反序列化引擎
        auto runtime = std::unique_ptr<IRuntime>(createInferRuntime(gLogger));
        engine_ = runtime->deserializeCudaEngine(engineData.data(), engineSize);
        
        if (!engine_) {
            std::cerr << "[BambooDetector] 反序列化引擎失败" << std::endl;
            return false;
        }
        
        std::cout << "[BambooDetector] 引擎加载成功: " << engine_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 加载引擎异常: " << e.what() << std::endl;
        return false;
    }
}

// 分配缓冲区
bool BambooDetector::allocateBuffers() {
    try {
        // 获取输入输出尺寸 - TensorRT 10.x 兼容
        nvinfer1::Dims input_dims, output_dims;
        
        // 使用新API获取张量形状
        const char* input_name = engine_->getIOTensorName(input_binding_index_);
        const char* output_name = engine_->getIOTensorName(output_binding_index_);
        
        input_dims = engine_->getTensorShape(input_name);
        output_dims = engine_->getTensorShape(output_name);
        
        // 计算输入输出大小
        size_t input_size = 1;
        size_t output_size = 1;
        
        for (int i = 0; i < input_dims.nbDims; ++i) {
            input_size *= input_dims.d[i];
        }
        
        for (int i = 0; i < output_dims.nbDims; ++i) {
            output_size *= output_dims.d[i];
        }
        
        input_size_ = input_size * sizeof(float);
        output_size_ = output_size * sizeof(float);
        
        // 分配GPU内存
        cudaMalloc(&input_device_buffer_, input_size_);
        cudaMalloc(&output_device_buffer_, output_size_);
        
        // 分配CPU内存
        input_host_buffer_.resize(input_size);
        output_host_buffer_.resize(output_size);
        
        std::cout << "[BambooDetector] 缓冲区分配成功" << std::endl;
        std::cout << "  输入尺寸: " << input_size << " floats (" << input_size_ << " bytes)" << std::endl;
        std::cout << "  输出尺寸: " << output_size << " floats (" << output_size_ << " bytes)" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 分配缓冲区异常: " << e.what() << std::endl;
        return false;
    }
}

// 预热模型
bool BambooDetector::warmupModel() {
    try {
        std::cout << "[BambooDetector] 预热模型..." << std::endl;
        
        // 创建虚拟输入数据
        std::fill(input_host_buffer_.begin(), input_host_buffer_.end(), 0.0f);
        
        // 执行几次推理预热
        for (int i = 0; i < 3; ++i) {
            // 复制输入数据到GPU
            cudaMemcpyAsync(input_device_buffer_, input_host_buffer_.data(), input_size_, 
                           cudaMemcpyHostToDevice, stream_);
            
            // 执行推理 - TensorRT 10.x
            const char* input_name = engine_->getIOTensorName(input_binding_index_);
            const char* output_name = engine_->getIOTensorName(output_binding_index_);
            
            context_->setTensorAddress(input_name, input_device_buffer_);
            context_->setTensorAddress(output_name, output_device_buffer_);
            context_->enqueueV3(stream_);
            
            // 复制输出数据到CPU
            cudaMemcpyAsync(output_host_buffer_.data(), output_device_buffer_, output_size_,
                           cudaMemcpyDeviceToHost, stream_);
            
            // 同步流
            cudaStreamSynchronize(stream_);
        }
        
        std::cout << "[BambooDetector] 模型预热完成" << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 模型预热异常: " << e.what() << std::endl;
        return false;
    }
}

// 执行检测
DetectionResult BambooDetector::detect(const cv::Mat& image) {
    DetectionResult result;
    
    if (!initialized_) {
        result.success = false;
        result.error_message = "检测器未初始化";
        return result;
    }
    
    try {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 预处理图像
        cv::Mat processed_image;
        if (!preprocessImage(image, processed_image)) {
            result.success = false;
            result.error_message = "图像预处理失败";
            return result;
        }
        
        // 执行推理
        if (!runInference(processed_image)) {
            result.success = false;
            result.error_message = "推理执行失败";
            return result;
        }
        
        // 后处理结果
        if (!postprocessResults(image, result)) {
            result.success = false;
            result.error_message = "结果后处理失败";
            return result;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        result.processing_time_ms = duration.count() / 1000.0f;
        result.success = true;
        
        // 更新性能统计
        updatePerformanceStats(result.processing_time_ms);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error_message = std::string("检测异常: ") + e.what();
    }
    
    return result;
}

// 预处理图像
bool BambooDetector::preprocessImage(const cv::Mat& image, cv::Mat& processed) {
    try {
        // 调整大小到模型输入尺寸
        int input_h = 640;  // YOLO输入高度
        int input_w = 640;  // YOLO输入宽度
        
        cv::resize(image, processed, cv::Size(input_w, input_h));
        
        // 转换为RGB
        if (processed.channels() == 3) {
            cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
        }
        
        // 归一化到[0,1]并转换为CHW格式
        processed.convertTo(processed, CV_32F, 1.0/255.0);
        
        // 转换为CHW格式并复制到输入缓冲区
        std::vector<cv::Mat> channels(3);
        cv::split(processed, channels);
        
        float* input_ptr = input_host_buffer_.data();
        for (int c = 0; c < 3; ++c) {
            memcpy(input_ptr + c * input_w * input_h, 
                   channels[c].data, 
                   input_w * input_h * sizeof(float));
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 图像预处理异常: " << e.what() << std::endl;
        return false;
    }
}

// 执行推理
bool BambooDetector::runInference(const cv::Mat& preprocessed_image) {
    try {
        // 复制输入数据到GPU
        cudaMemcpyAsync(input_device_buffer_, input_host_buffer_.data(), input_size_,
                       cudaMemcpyHostToDevice, stream_);
        
        // 执行推理 - TensorRT 10.x
        const char* input_name = engine_->getIOTensorName(input_binding_index_);
        const char* output_name = engine_->getIOTensorName(output_binding_index_);
        
        context_->setTensorAddress(input_name, input_device_buffer_);
        context_->setTensorAddress(output_name, output_device_buffer_);
        bool success = context_->enqueueV3(stream_);
        
        if (!success) {
            std::cerr << "[BambooDetector] 推理执行失败" << std::endl;
            return false;
        }
        
        // 复制输出数据到CPU
        cudaMemcpyAsync(output_host_buffer_.data(), output_device_buffer_, output_size_,
                       cudaMemcpyDeviceToHost, stream_);
        
        // 同步流
        cudaStreamSynchronize(stream_);
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 推理执行异常: " << e.what() << std::endl;
        return false;
    }
}

// 后处理结果
bool BambooDetector::postprocessResults(const cv::Mat& original_image, DetectionResult& result) {
    try {
        result.points.clear();
        
        // YOLO输出格式解析
        const int num_classes = 2;  // 切点和节点两类
        const int box_elements = 5 + num_classes;  // x,y,w,h,conf,class0,class1
        const int num_proposals = output_host_buffer_.size() / box_elements;
        
        float scale_x = static_cast<float>(original_image.cols) / 640.0f;
        float scale_y = static_cast<float>(original_image.rows) / 640.0f;
        
        // 解析检测结果
        for (int i = 0; i < num_proposals; ++i) {
            float* proposal = output_host_buffer_.data() + i * box_elements;
            
            float cx = proposal[0] * scale_x;
            float cy = proposal[1] * scale_y;
            float w = proposal[2] * scale_x;
            float h = proposal[3] * scale_y;
            float obj_conf = proposal[4];
            
            if (obj_conf < config_.confidence_threshold) {
                continue;
            }
            
            // 找到最高置信度的类别
            int best_class = 0;
            float best_class_conf = proposal[5];
            
            for (int c = 1; c < num_classes; ++c) {
                if (proposal[5 + c] > best_class_conf) {
                    best_class_conf = proposal[5 + c];
                    best_class = c;
                }
            }
            
            float final_conf = obj_conf * best_class_conf;
            if (final_conf < config_.confidence_threshold) {
                continue;
            }
            
            // 创建检测点
            DetectionPoint point;
            point.x = cx;
            point.y = cy;
            point.confidence = final_conf;
            point.class_id = best_class;
            
            result.points.push_back(point);
        }
        
        // 按置信度排序
        std::sort(result.points.begin(), result.points.end(),
                 [](const DetectionPoint& a, const DetectionPoint& b) {
                     return a.confidence > b.confidence;
                 });
        
        // 限制最大检测数量
        if (result.points.size() > static_cast<size_t>(config_.max_detections)) {
            result.points.resize(config_.max_detections);
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 结果后处理异常: " << e.what() << std::endl;
        return false;
    }
}

// 清理资源
void BambooDetector::cleanup() {
    try {
        if (input_device_buffer_) {
            cudaFree(input_device_buffer_);
            input_device_buffer_ = nullptr;
        }
        
        if (output_device_buffer_) {
            cudaFree(output_device_buffer_);
            output_device_buffer_ = nullptr;
        }
        
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        
        // TensorRT 10.x 使用智能指针，不需要手动调用destroy()
        if (context_) {
            delete context_;
            context_ = nullptr;
        }
        
        if (engine_) {
            delete engine_;
            engine_ = nullptr;
        }
        
        initialized_ = false;
        
    } catch (const std::exception& e) {
        std::cerr << "[BambooDetector] 清理资源异常: " << e.what() << std::endl;
    }
}

// 更新性能统计
void BambooDetector::updatePerformanceStats(float processing_time_ms) {
    perf_stats_.total_detections++;
    perf_stats_.total_time_ms += processing_time_ms;
    perf_stats_.avg_time_ms = perf_stats_.total_time_ms / perf_stats_.total_detections;
    
    if (perf_stats_.total_detections == 1) {
        perf_stats_.min_time_ms = processing_time_ms;
        perf_stats_.max_time_ms = processing_time_ms;
    } else {
        if (processing_time_ms < perf_stats_.min_time_ms) {
            perf_stats_.min_time_ms = processing_time_ms;
        }
        if (processing_time_ms > perf_stats_.max_time_ms) {
            perf_stats_.max_time_ms = processing_time_ms;
        }
    }
    
    perf_stats_.fps = perf_stats_.avg_time_ms > 0.0f ? 1000.0f / perf_stats_.avg_time_ms : 0.0f;
}

// 重置性能统计
void BambooDetector::resetPerformanceStats() {
    perf_stats_ = PerformanceStats();
}

// 批量检测
std::vector<DetectionResult> BambooDetector::detectBatch(const std::vector<cv::Mat>& images) {
    std::vector<DetectionResult> results;
    results.reserve(images.size());
    
    for (const auto& image : images) {
        results.push_back(detect(image));
    }
    
    return results;
}