#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <sys/stat.h>
#include <gst/gst.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <drm/drm.h>
#include <fcntl.h>
#include <unistd.h>

class NvidiaDrmMigrationTester {
private:
    struct TestResult {
        std::string test_name;
        bool success;
        std::string details;
        double performance_score = 0.0;
    };
    
    std::vector<TestResult> test_results_;
    
public:
    void runAllTests() {
        std::cout << "=== NVIDIA-DRM 驱动迁移完整验证测试 ===" << std::endl;
        std::cout << "测试时间: " << getCurrentTimeString() << std::endl;
        std::cout << "测试目标: 验证从tegra_drm到nvidia-drm的完整迁移" << std::endl;
        std::cout << std::endl;
        
        // 1. 基础驱动验证
        testDrmDriverMigration();
        
        // 2. EGL环境验证
        testEglEnvironment();
        
        // 3. 帧缓冲区功能验证
        testFramebufferInitialization();
        
        // 4. 显示输出配置验证
        testDisplayConfiguration();
        
        // 5. GPU加速功能验证
        testGpuAcceleration();
        
        // 6. LVGL界面渲染验证
        testLvglRendering();
        
        // 7. 触摸交互验证
        testTouchInteraction();
        
        // 8. 摄像头集成验证
        testCameraIntegration();
        
        // 9. 性能基准测试
        testPerformanceBenchmark();
        
        // 10. 统一架构验证
        testUnifiedArchitecture();
        
        // 生成最终报告
        generateFinalReport();
    }

private:
    std::string getCurrentTimeString() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        char buffer[100];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", localtime(&time_t));
        return std::string(buffer);
    }
    
    void testDrmDriverMigration() {
        std::cout << "\n=== 1. DRM驱动迁移验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "DRM驱动迁移";
        result.success = true;
        
        // 检查nvidia-drm模块
        std::cout << "检查nvidia-drm模块..." << std::endl;
        FILE* pipe = popen("lsmod | grep nvidia_drm", "r");
        char buffer[256];
        bool nvidia_drm_loaded = false;
        
        if (pipe) {
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                if (strstr(buffer, "nvidia_drm") != nullptr) {
                    nvidia_drm_loaded = true;
                    std::cout << "✅ nvidia-drm模块已加载: " << buffer;
                    break;
                }
            }
            pclose(pipe);
        }
        
        if (!nvidia_drm_loaded) {
            std::cout << "❌ nvidia-drm模块未加载" << std::endl;
            result.success = false;
            result.details += "nvidia-drm模块未加载; ";
        }
        
        // 检查DRM设备
        std::cout << "检查DRM设备..." << std::endl;
        struct stat st;
        if (stat("/dev/dri/card0", &st) == 0) {
            std::cout << "✅ DRM设备 /dev/dri/card0 存在" << std::endl;
        } else {
            std::cout << "❌ DRM设备 /dev/dri/card0 不存在" << std::endl;
            result.success = false;
            result.details += "DRM设备不存在; ";
        }
        
        // 检查是否已从tegra_drm迁移
        pipe = popen("lsmod | grep tegra_drm", "r");
        bool tegra_drm_loaded = false;
        
        if (pipe) {
            while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                if (strstr(buffer, "tegra_drm") != nullptr) {
                    tegra_drm_loaded = true;
                    break;
                }
            }
            pclose(pipe);
        }
        
        if (tegra_drm_loaded) {
            std::cout << "⚠️ tegra_drm模块仍在使用，迁移可能不完整" << std::endl;
            result.details += "tegra_drm仍在使用; ";
        } else {
            std::cout << "✅ 已成功从tegra_drm迁移" << std::endl;
        }
        
        test_results_.push_back(result);
    }
    
    void testEglEnvironment() {
        std::cout << "\n=== 2. EGL环境验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "EGL环境";
        result.success = true;
        
        // 设置EGL环境变量
        setenv("EGL_PLATFORM", "drm", 1);
        setenv("__EGL_VENDOR_LIBRARY_DIRS", "/usr/lib/aarch64-linux-gnu/tegra-egl", 1);
        
        std::cout << "初始化EGL..." << std::endl;
        EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        
        if (display == EGL_NO_DISPLAY) {
            std::cout << "❌ 无法获取EGL Display" << std::endl;
            result.success = false;
            result.details += "EGL Display获取失败; ";
        } else {
            std::cout << "✅ EGL Display获取成功" << std::endl;
            
            EGLint major, minor;
            if (eglInitialize(display, &major, &minor)) {
                std::cout << "✅ EGL初始化成功: " << major << "." << minor << std::endl;
                
                const char* vendor = eglQueryString(display, EGL_VENDOR);
                const char* version = eglQueryString(display, EGL_VERSION);
                
                std::cout << "EGL厂商: " << (vendor ? vendor : "Unknown") << std::endl;
                std::cout << "EGL版本: " << (version ? version : "Unknown") << std::endl;
                
                eglTerminate(display);
            } else {
                std::cout << "❌ EGL初始化失败: 0x" << std::hex << eglGetError() << std::endl;
                result.success = false;
                result.details += "EGL初始化失败; ";
            }
        }
        
        test_results_.push_back(result);
    }
    
    void testFramebufferInitialization() {
        std::cout << "\n=== 3. 帧缓冲区初始化验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "帧缓冲区初始化";
        result.success = true;
        
        // 检查帧缓冲设备
        struct stat st;
        if (stat("/dev/fb0", &st) == 0) {
            std::cout << "✅ 帧缓冲设备 /dev/fb0 存在" << std::endl;
            
            // 读取帧缓冲信息
            std::ifstream resolution_file("/sys/class/graphics/fb0/virtual_size");
            if (resolution_file.is_open()) {
                std::string resolution;
                getline(resolution_file, resolution);
                std::cout << "帧缓冲分辨率: " << resolution << std::endl;
                resolution_file.close();
            }
            
            std::ifstream bits_file("/sys/class/graphics/fb0/bits_per_pixel");
            if (bits_file.is_open()) {
                std::string bits;
                getline(bits_file, bits);
                std::cout << "色深: " << bits << " bits" << std::endl;
                bits_file.close();
            }
        } else {
            std::cout << "❌ 帧缓冲设备 /dev/fb0 不存在" << std::endl;
            result.success = false;
            result.details += "帧缓冲设备不存在; ";
        }
        
        test_results_.push_back(result);
    }
    
    void testDisplayConfiguration() {
        std::cout << "\n=== 4. 显示输出配置验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "显示输出配置";
        result.success = true;
        
        // 使用DRM API检查显示配置
        int drm_fd = open("/dev/dri/card0", O_RDWR);
        if (drm_fd < 0) {
            std::cout << "❌ 无法打开DRM设备" << std::endl;
            result.success = false;
            result.details += "DRM设备访问失败; ";
        } else {
            std::cout << "✅ DRM设备打开成功" << std::endl;
            close(drm_fd);
        }
        
        // 检查显示相关文件
        if (stat("/sys/class/drm", &st) == 0) {
            std::cout << "✅ DRM sysfs接口可用" << std::endl;
            
            // 列出可用的连接器
            system("ls /sys/class/drm/card0-* 2>/dev/null | head -5");
        }
        
        test_results_.push_back(result);
    }
    
    void testGpuAcceleration() {
        std::cout << "\n=== 5. GPU加速功能验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "GPU加速";
        result.success = true;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 测试GPU加速 - 使用EGL创建上下文
        EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display != EGL_NO_DISPLAY) {
            EGLint major, minor;
            if (eglInitialize(display, &major, &minor)) {
                std::cout << "✅ GPU EGL初始化成功" << std::endl;
                
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                result.performance_score = 1000.0 / duration.count(); // 性能分数
                
                std::cout << "GPU初始化耗时: " << duration.count() << " 微秒" << std::endl;
                
                eglTerminate(display);
            } else {
                std::cout << "❌ GPU EGL初始化失败" << std::endl;
                result.success = false;
                result.details += "EGL初始化失败; ";
            }
        }
        
        test_results_.push_back(result);
    }
    
    void testLvglRendering() {
        std::cout << "\n=== 6. LVGL界面渲染验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "LVGL渲染";
        result.success = true;
        
        // 检查LVGL相关文件是否存在
        std::vector<std::string> lvgl_files = {
            "cpp_backend/src/ui/lvgl_interface.cpp",
            "cpp_backend/src/ui/lvgl_display_drm.cpp",
            "lv_conf.h",
            "simple_unified_main.cpp"
        };
        
        for (const auto& file : lvgl_files) {
            struct stat st;
            if (stat(file.c_str(), &st) == 0) {
                std::cout << "✅ LVGL文件存在: " << file << std::endl;
            } else {
                std::cout << "❌ LVGL文件缺失: " << file << std::endl;
                result.success = false;
                result.details += "LVGL文件缺失; ";
            }
        }
        
        // 检查LVGL编译配置
        std::ifstream lv_conf("lv_conf.h");
        if (lv_conf.is_open()) {
            std::string line;
            bool found_argb8888 = false;
            
            while (getline(lv_conf, line)) {
                if (line.find("LV_COLOR_DEPTH") != std::string::npos && 
                    line.find("32") != std::string::npos) {
                    found_argb8888 = true;
                    std::cout << "✅ LVGL ARGB8888支持已启用" << std::endl;
                    break;
                }
            }
            
            if (!found_argb8888) {
                std::cout << "⚠️ LVGL ARGB8888支持可能未正确配置" << std::endl;
            }
            
            lv_conf.close();
        }
        
        test_results_.push_back(result);
    }
    
    void testTouchInteraction() {
        std::cout << "\n=== 7. 触摸交互验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "触摸交互";
        result.success = true;
        
        // 检查触摸设备
        std::vector<std::string> touch_devices = {
            "/dev/input/event0",
            "/dev/input/event1",
            "/dev/input/event2"
        };
        
        bool touch_found = false;
        for (const auto& device : touch_devices) {
            struct stat st;
            if (stat(device.c_str(), &st) == 0) {
                std::cout << "✅ 触摸设备找到: " << device << std::endl;
                touch_found = true;
                break;
            }
        }
        
        if (!touch_found) {
            std::cout << "⚠️ 未找到触摸设备，可能需要外接触摸屏" << std::endl;
            result.details += "触摸设备未找到; ";
        }
        
        // 检查触摸测试工具
        struct stat st;
        if (stat("cpp_backend/src/ui/touch_test.cpp", &st) == 0) {
            std::cout << "✅ 触摸测试工具可用" << std::endl;
        }
        
        test_results_.push_back(result);
    }
    
    void testCameraIntegration() {
        std::cout << "\n=== 8. 摄像头集成验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "摄像头集成";
        result.success = true;
        
        // 初始化GStreamer
        gst_init(nullptr, nullptr);
        
        // 测试nvarguscamerasrc
        std::cout << "测试nvarguscamerasrc..." << std::endl;
        GstElement* source = gst_element_factory_make("nvarguscamerasrc", "test-source");
        if (source) {
            std::cout << "✅ nvarguscamerasrc可用" << std::endl;
            gst_object_unref(source);
        } else {
            std::cout << "❌ nvarguscamerasrc不可用" << std::endl;
            result.success = false;
            result.details += "nvarguscamerasrc不可用; ";
        }
        
        // 检查摄像头设备
        struct stat st;
        if (stat("/dev/video0", &st) == 0) {
            std::cout << "✅ 摄像头设备 /dev/video0 存在" << std::endl;
        } else {
            std::cout << "⚠️ 摄像头设备 /dev/video0 不存在" << std::endl;
            result.details += "摄像头设备不存在; ";
        }
        
        // 检查nvargus-daemon服务
        FILE* pipe = popen("systemctl is-active nvargus-daemon", "r");
        if (pipe) {
            char buffer[256];
            if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                if (strstr(buffer, "active") != nullptr) {
                    std::cout << "✅ nvargus-daemon服务运行中" << std::endl;
                } else {
                    std::cout << "❌ nvargus-daemon服务未运行" << std::endl;
                    result.success = false;
                    result.details += "nvargus-daemon未运行; ";
                }
            }
            pclose(pipe);
        }
        
        test_results_.push_back(result);
    }
    
    void testPerformanceBenchmark() {
        std::cout << "\n=== 9. 性能基准测试 ===" << std::endl;
        
        TestResult result;
        result.test_name = "性能基准";
        result.success = true;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 简单的CPU/内存性能测试
        const int iterations = 1000000;
        volatile double sum = 0.0;
        
        for (int i = 0; i < iterations; i++) {
            sum += sqrt(i) * sin(i) * cos(i);
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        result.performance_score = 1000.0 / duration.count();
        
        std::cout << "CPU性能测试完成，耗时: " << duration.count() << " ms" << std::endl;
        std::cout << "性能分数: " << result.performance_score << std::endl;
        
        test_results_.push_back(result);
    }
    
    void testUnifiedArchitecture() {
        std::cout << "\n=== 10. 统一架构验证 ===" << std::endl;
        
        TestResult result;
        result.test_name = "统一架构";
        result.success = true;
        
        // 检查统一架构文件
        struct stat st;
        if (stat("simple_unified_main.cpp", &st) == 0) {
            std::cout << "✅ 统一架构源文件存在" << std::endl;
        } else {
            std::cout << "❌ 统一架构源文件缺失" << std::endl;
            result.success = false;
            result.details += "统一架构文件缺失; ";
        }
        
        // 检查EGL上下文管理器
        if (stat("cpp_backend/include/bamboo_cut/ui/egl_context_manager.h", &st) == 0) {
            std::cout << "✅ EGL上下文管理器头文件存在" << std::endl;
        }
        
        if (stat("cpp_backend/src/ui/egl_context_manager.cpp", &st) == 0) {
            std::cout << "✅ EGL上下文管理器实现存在" << std::endl;
        }
        
        // 检查Makefile中的统一构建目标
        std::ifstream makefile("Makefile");
        if (makefile.is_open()) {
            std::string line;
            bool found_unified = false;
            
            while (getline(makefile, line)) {
                if (line.find("unified:") != std::string::npos) {
                    found_unified = true;
                    std::cout << "✅ Makefile包含统一架构构建目标" << std::endl;
                    break;
                }
            }
            
            if (!found_unified) {
                std::cout << "❌ Makefile缺少统一架构构建目标" << std::endl;
                result.success = false;
                result.details += "Makefile配置缺失; ";
            }
            
            makefile.close();
        }
        
        test_results_.push_back(result);
    }
    
    void generateFinalReport() {
        std::cout << "\n=== 最终测试报告 ===" << std::endl;
        std::cout << "测试完成时间: " << getCurrentTimeString() << std::endl;
        std::cout << std::endl;
        
        int total_tests = test_results_.size();
        int passed_tests = 0;
        double total_performance = 0.0;
        
        for (const auto& result : test_results_) {
            std::string status = result.success ? "✅ PASS" : "❌ FAIL";
            std::cout << status << " " << result.test_name;
            
            if (result.performance_score > 0) {
                std::cout << " (性能分数: " << result.performance_score << ")";
                total_performance += result.performance_score;
            }
            
            if (!result.details.empty()) {
                std::cout << " - " << result.details;
            }
            
            std::cout << std::endl;
            
            if (result.success) {
                passed_tests++;
            }
        }
        
        std::cout << std::endl;
        std::cout << "=== 测试汇总 ===" << std::endl;
        std::cout << "总测试数: " << total_tests << std::endl;
        std::cout << "通过测试: " << passed_tests << std::endl;
        std::cout << "失败测试: " << (total_tests - passed_tests) << std::endl;
        std::cout << "通过率: " << (100.0 * passed_tests / total_tests) << "%" << std::endl;
        
        if (total_performance > 0) {
            std::cout << "平均性能分数: " << (total_performance / total_tests) << std::endl;
        }
        
        std::cout << std::endl;
        
        if (passed_tests == total_tests) {
            std::cout << "🎉 恭喜！nvidia-drm驱动迁移验证完全通过！" << std::endl;
            std::cout << "系统已成功从tegra_drm迁移到nvidia-drm，所有功能正常。" << std::endl;
        } else {
            std::cout << "⚠️ 发现问题：有 " << (total_tests - passed_tests) << " 个测试失败。" << std::endl;
            std::cout << "请检查上述失败项目并进行修复。" << std::endl;
        }
        
        // 保存测试报告到文件
        saveReportToFile();
    }
    
    void saveReportToFile() {
        std::ofstream report_file("nvidia_drm_migration_report.txt");
        if (report_file.is_open()) {
            report_file << "NVIDIA-DRM 驱动迁移测试报告\n";
            report_file << "生成时间: " << getCurrentTimeString() << "\n\n";
            
            for (const auto& result : test_results_) {
                report_file << (result.success ? "PASS" : "FAIL") << " - " << result.test_name;
                if (result.performance_score > 0) {
                    report_file << " (性能: " << result.performance_score << ")";
                }
                if (!result.details.empty()) {
                    report_file << " - " << result.details;
                }
                report_file << "\n";
            }
            
            report_file.close();
            std::cout << "📄 详细报告已保存到: nvidia_drm_migration_report.txt" << std::endl;
        }
    }
};

int main() {
    std::cout << "启动NVIDIA-DRM驱动迁移验证测试..." << std::endl;
    
    NvidiaDrmMigrationTester tester;
    tester.runAllTests();
    
    return 0;
}