/**
 * Python绑定模块 - pybind11接口层
 * 连接Python和C++推理引擎
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/chrono.h>
#include "inference_core.h"

namespace py = pybind11;

/**
 * OpenCV Mat与NumPy数组的转换工具
 */
cv::Mat numpy_to_mat(py::array_t<uint8_t> input) {
    py::buffer_info buf_info = input.request();
    
    // 检查维度
    if (buf_info.ndim != 3) {
        throw std::runtime_error("输入数组必须是3维 (height, width, channels)");
    }
    
    int height = buf_info.shape[0];
    int width = buf_info.shape[1];
    int channels = buf_info.shape[2];
    
    if (channels != 3 && channels != 1) {
        throw std::runtime_error("不支持的通道数，仅支持1或3通道");
    }
    
    // 创建OpenCV Mat
    cv::Mat mat;
    if (channels == 3) {
        mat = cv::Mat(height, width, CV_8UC3, (unsigned char*)buf_info.ptr);
    } else {
        mat = cv::Mat(height, width, CV_8UC1, (unsigned char*)buf_info.ptr);
    }
    
    return mat.clone();  // 克隆以确保内存安全
}

py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        throw std::runtime_error("输入Mat为空");
    }
    
    std::vector<py::ssize_t> shape;
    shape.push_back(mat.rows);
    shape.push_back(mat.cols);
    
    if (mat.channels() > 1) {
        shape.push_back(mat.channels());
    }
    
    std::vector<py::ssize_t> strides;
    strides.push_back(mat.step[0]);
    strides.push_back(mat.step[1]);
    
    if (mat.channels() > 1) {
        strides.push_back(sizeof(uint8_t));
    }
    
    return py::array_t<uint8_t>(
        shape,
        strides,
        mat.data,
        py::cast(mat)  // 保持Mat对象的生命周期
    );
}

/**
 * 增强的BambooDetector包装类
 */
class PyBambooDetector {
public:
    PyBambooDetector() : detector_() {}
    
    bool initialize(const DetectorConfig& config) {
        try {
            std::cout << "[PyBambooDetector] 初始化Python绑定检测器..." << std::endl;
            bool success = detector_.initialize(config);
            if (success) {
                std::cout << "[PyBambooDetector] 初始化成功" << std::endl;
            } else {
                std::cout << "[PyBambooDetector] 初始化失败" << std::endl;
            }
            return success;
        } catch (const std::exception& e) {
            std::cerr << "[PyBambooDetector] 初始化异常: " << e.what() << std::endl;
            return false;
        }
    }
    
    DetectionResult detect(py::array_t<uint8_t> input_array) {
        try {
            // 转换NumPy数组为OpenCV Mat
            cv::Mat image = numpy_to_mat(input_array);
            
            // 执行检测
            return detector_.detect(image);
            
        } catch (const std::exception& e) {
            DetectionResult error_result;
            error_result.success = false;
            error_result.error_message = std::string("检测异常: ") + e.what();
            return error_result;
        }
    }
    
    std::vector<DetectionResult> detect_batch(const std::vector<py::array_t<uint8_t>>& input_arrays) {
        try {
            // 转换NumPy数组列表为OpenCV Mat列表
            std::vector<cv::Mat> images;
            images.reserve(input_arrays.size());
            
            for (const auto& array : input_arrays) {
                images.push_back(numpy_to_mat(array));
            }
            
            // 执行批量检测
            return detector_.detectBatch(images);
            
        } catch (const std::exception& e) {
            std::vector<DetectionResult> error_results;
            DetectionResult error_result;
            error_result.success = false;
            error_result.error_message = std::string("批量检测异常: ") + e.what();
            error_results.push_back(error_result);
            return error_results;
        }
    }
    
    bool is_initialized() const {
        return detector_.isInitialized();
    }
    
    DetectorConfig get_config() const {
        return detector_.getConfig();
    }
    
    BambooDetector::PerformanceStats get_performance_stats() const {
        return detector_.getPerformanceStats();
    }
    
    void reset_performance_stats() {
        detector_.resetPerformanceStats();
    }
    
    void cleanup() {
        detector_.cleanup();
    }
    
    // 便利方法：从文件路径检测
    DetectionResult detect_from_file(const std::string& image_path) {
        try {
            cv::Mat image = cv::imread(image_path);
            if (image.empty()) {
                DetectionResult error_result;
                error_result.success = false;
                error_result.error_message = "无法加载图像文件: " + image_path;
                return error_result;
            }
            
            return detector_.detect(image);
            
        } catch (const std::exception& e) {
            DetectionResult error_result;
            error_result.success = false;
            error_result.error_message = std::string("文件检测异常: ") + e.what();
            return error_result;
        }
    }
    
    // 获取模型信息
    py::dict get_model_info() const {
        py::dict info;
        const auto& config = detector_.getConfig();
        
        info["model_path"] = config.model_path;
        info["engine_path"] = config.engine_path;
        info["confidence_threshold"] = config.confidence_threshold;
        info["nms_threshold"] = config.nms_threshold;
        info["max_detections"] = config.max_detections;
        info["use_tensorrt"] = config.use_tensorrt;
        info["use_fp16"] = config.use_fp16;
        info["batch_size"] = config.batch_size;
        info["input_width"] = config.input_width;
        info["input_height"] = config.input_height;
        info["initialized"] = detector_.isInitialized();
        
        return info;
    }

private:
    BambooDetector detector_;
};

/**
 * 系统信息获取函数
 */
py::dict get_system_info() {
    py::dict info;
    
    // CUDA设备信息
    auto cuda_info = BambooUtils::getCudaDeviceInfo();
    py::dict cuda_dict;
    cuda_dict["device_count"] = cuda_info.device_count;
    cuda_dict["current_device"] = cuda_info.current_device;
    cuda_dict["device_name"] = cuda_info.device_name;
    cuda_dict["total_memory"] = cuda_info.total_memory;
    cuda_dict["free_memory"] = cuda_info.free_memory;
    cuda_dict["compute_capability"] = std::to_string(cuda_info.compute_capability_major) + "." + std::to_string(cuda_info.compute_capability_minor);
    
    info["cuda"] = cuda_dict;
    info["tensorrt_version"] = BambooUtils::getTensorRTVersion();
    info["opencv_version"] = BambooUtils::getOpenCVVersion();
    
    return info;
}

/**
 * 设置日志级别
 */
bool setup_logging(const std::string& log_level) {
    return BambooUtils::setupLogging(log_level);
}

/**
 * 图像处理工具函数
 */
py::array_t<uint8_t> resize_image(py::array_t<uint8_t> input_array, int width, int height) {
    try {
        cv::Mat image = numpy_to_mat(input_array);
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(width, height));
        return mat_to_numpy(resized);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("图像缩放异常: ") + e.what());
    }
}

py::array_t<uint8_t> convert_color(py::array_t<uint8_t> input_array, const std::string& conversion) {
    try {
        cv::Mat image = numpy_to_mat(input_array);
        cv::Mat converted;
        
        if (conversion == "BGR2RGB") {
            cv::cvtColor(image, converted, cv::COLOR_BGR2RGB);
        } else if (conversion == "RGB2BGR") {
            cv::cvtColor(image, converted, cv::COLOR_RGB2BGR);
        } else if (conversion == "BGR2GRAY") {
            cv::cvtColor(image, converted, cv::COLOR_BGR2GRAY);
        } else if (conversion == "RGB2GRAY") {
            cv::cvtColor(image, converted, cv::COLOR_RGB2GRAY);
        } else {
            throw std::runtime_error("不支持的颜色转换: " + conversion);
        }
        
        return mat_to_numpy(converted);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("颜色转换异常: ") + e.what());
    }
}

/**
 * 性能测试函数
 */
py::dict benchmark_detector(PyBambooDetector& detector, py::array_t<uint8_t> test_image, int num_iterations) {
    try {
        if (!detector.is_initialized()) {
            throw std::runtime_error("检测器未初始化");
        }
        
        std::vector<float> times;
        times.reserve(num_iterations);
        
        // 预热
        for (int i = 0; i < 3; ++i) {
            detector.detect(test_image);
        }
        
        // 重置性能统计
        detector.reset_performance_stats();
        
        // 执行基准测试
        auto start_total = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            auto result = detector.detect(test_image);
            auto end = std::chrono::high_resolution_clock::now();
            
            if (result.success) {
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                times.push_back(duration.count() / 1000.0f);
            }
        }\n        \n        auto end_total = std::chrono::high_resolution_clock::now();\n        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);\n        \n        // 计算统计信息\n        if (times.empty()) {\n            throw std::runtime_error(\"没有成功的检测结果\");\n        }\n        \n        float min_time = *std::min_element(times.begin(), times.end());\n        float max_time = *std::max_element(times.begin(), times.end());\n        float avg_time = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();\n        float total_time = total_duration.count();\n        \n        // 计算标准差\n        float variance = 0.0f;\n        for (float time : times) {\n            variance += (time - avg_time) * (time - avg_time);\n        }\n        variance /= times.size();\n        float std_dev = std::sqrt(variance);\n        \n        py::dict benchmark_result;\n        benchmark_result[\"iterations\"] = num_iterations;\n        benchmark_result[\"successful_iterations\"] = static_cast<int>(times.size());\n        benchmark_result[\"min_time_ms\"] = min_time;\n        benchmark_result[\"max_time_ms\"] = max_time;\n        benchmark_result[\"avg_time_ms\"] = avg_time;\n        benchmark_result[\"std_dev_ms\"] = std_dev;\n        benchmark_result[\"total_time_ms\"] = total_time;\n        benchmark_result[\"fps\"] = 1000.0f / avg_time;\n        benchmark_result[\"throughput_fps\"] = (times.size() * 1000.0f) / total_time;\n        \n        return benchmark_result;\n        \n    } catch (const std::exception& e) {\n        throw std::runtime_error(std::string(\"基准测试异常: \") + e.what());\n    }\n}\n\n/**\n * pybind11模块定义\n */\nPYBIND11_MODULE(cpp_inference_core, m) {\n    m.doc() = \"竹子检测系统C++推理核心模块\";\n    \n    // 检测点结构\n    py::class_<DetectionPoint>(m, \"DetectionPoint\")\n        .def(py::init<>())\n        .def(py::init<float, float, float, int>())\n        .def_readwrite(\"x\", &DetectionPoint::x)\n        .def_readwrite(\"y\", &DetectionPoint::y)\n        .def_readwrite(\"confidence\", &DetectionPoint::confidence)\n        .def_readwrite(\"class_id\", &DetectionPoint::class_id)\n        .def(\"to_string\", &DetectionPoint::toString)\n        .def(\"__repr__\", &DetectionPoint::toString);\n    \n    // 检测结果结构\n    py::class_<DetectionResult>(m, \"DetectionResult\")\n        .def(py::init<>())\n        .def_readwrite(\"points\", &DetectionResult::points)\n        .def_readwrite(\"processing_time_ms\", &DetectionResult::processing_time_ms)\n        .def_readwrite(\"success\", &DetectionResult::success)\n        .def_readwrite(\"error_message\", &DetectionResult::error_message)\n        .def(\"add_point\", &DetectionResult::addPoint)\n        .def(\"get_point_count\", &DetectionResult::getPointCount)\n        .def(\"clear\", &DetectionResult::clear)\n        .def(\"to_string\", &DetectionResult::toString)\n        .def(\"__repr__\", &DetectionResult::toString);\n    \n    // 检测器配置结构\n    py::class_<DetectorConfig>(m, \"DetectorConfig\")\n        .def(py::init<>())\n        .def_readwrite(\"model_path\", &DetectorConfig::model_path)\n        .def_readwrite(\"engine_path\", &DetectorConfig::engine_path)\n        .def_readwrite(\"confidence_threshold\", &DetectorConfig::confidence_threshold)\n        .def_readwrite(\"nms_threshold\", &DetectorConfig::nms_threshold)\n        .def_readwrite(\"max_detections\", &DetectorConfig::max_detections)\n        .def_readwrite(\"use_tensorrt\", &DetectorConfig::use_tensorrt)\n        .def_readwrite(\"use_fp16\", &DetectorConfig::use_fp16)\n        .def_readwrite(\"batch_size\", &DetectorConfig::batch_size)\n        .def_readwrite(\"input_width\", &DetectorConfig::input_width)\n        .def_readwrite(\"input_height\", &DetectorConfig::input_height)\n        .def(\"validate\", &DetectorConfig::validate);\n    \n    // 性能统计结构\n    py::class_<BambooDetector::PerformanceStats>(m, \"PerformanceStats\")\n        .def(py::init<>())\n        .def_readonly(\"total_detections\", &BambooDetector::PerformanceStats::total_detections)\n        .def_readonly(\"total_time_ms\", &BambooDetector::PerformanceStats::total_time_ms)\n        .def_readonly(\"avg_time_ms\", &BambooDetector::PerformanceStats::avg_time_ms)\n        .def_readonly(\"min_time_ms\", &BambooDetector::PerformanceStats::min_time_ms)\n        .def_readonly(\"max_time_ms\", &BambooDetector::PerformanceStats::max_time_ms)\n        .def_readonly(\"fps\", &BambooDetector::PerformanceStats::fps);\n    \n    // Python包装的检测器类\n    py::class_<PyBambooDetector>(m, \"BambooDetector\")\n        .def(py::init<>())\n        .def(\"initialize\", &PyBambooDetector::initialize)\n        .def(\"detect\", &PyBambooDetector::detect)\n        .def(\"detect_batch\", &PyBambooDetector::detect_batch)\n        .def(\"detect_from_file\", &PyBambooDetector::detect_from_file)\n        .def(\"is_initialized\", &PyBambooDetector::is_initialized)\n        .def(\"get_config\", &PyBambooDetector::get_config)\n        .def(\"get_performance_stats\", &PyBambooDetector::get_performance_stats)\n        .def(\"reset_performance_stats\", &PyBambooDetector::reset_performance_stats)\n        .def(\"get_model_info\", &PyBambooDetector::get_model_info)\n        .def(\"cleanup\", &PyBambooDetector::cleanup);\n    \n    // 工具函数\n    m.def(\"get_system_info\", &get_system_info, \"获取系统信息\");\n    m.def(\"setup_logging\", &setup_logging, \"设置日志级别\");\n    m.def(\"resize_image\", &resize_image, \"调整图像大小\");\n    m.def(\"convert_color\", &convert_color, \"转换图像颜色空间\");\n    m.def(\"benchmark_detector\", &benchmark_detector, \"基准测试检测器性能\");\n    \n    // 版本信息\n    m.attr(\"__version__\") = \"1.0.0\";\n    m.attr(\"__author__\") = \"Bamboo Detection System\";\n}\n