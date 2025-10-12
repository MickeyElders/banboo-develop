#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <sys/stat.h>
#include <gst/gst.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

class CameraDiagnostics {
private:
    static void checkCameraDevices() {
        std::cout << "\n=== 摄像头设备检查 ===" << std::endl;
        
        // 检查 /dev/video* 设备
        for (int i = 0; i < 10; i++) {
            std::string device = "/dev/video" + std::to_string(i);
            struct stat buffer;
            if (stat(device.c_str(), &buffer) == 0) {
                std::cout << "发现设备: " << device << std::endl;
            }
        }
        
        // 检查 Argus 设备
        std::vector<std::string> argus_devices = {
            "/dev/argus-daemon",
            "/dev/nvhost-vi",
            "/dev/nvhost-isp",
            "/dev/nvhost-nvcsi"
        };
        
        for (const auto& device : argus_devices) {
            struct stat buffer;
            if (stat(device.c_str(), &buffer) == 0) {
                std::cout << "Argus设备存在: " << device << std::endl;
            } else {
                std::cout << "Argus设备缺失: " << device << std::endl;
            }
        }
    }
    
    static void checkProcesses() {
        std::cout << "\n=== 进程占用检查 ===" << std::endl;
        
        // 检查可能占用摄像头的进程
        std::vector<std::string> camera_processes = {
            "nvargus-daemon",
            "argus_daemon", 
            "nvarguscamerasrc",
            "gst-launch-1.0"
        };
        
        for (const auto& process : camera_processes) {
            std::string cmd = "pgrep -l " + process;
            FILE* pipe = popen(cmd.c_str(), "r");
            if (pipe) {
                char buffer[256];
                while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
                    std::cout << "发现进程: " << buffer;
                }
                pclose(pipe);
            }
        }
    }
    
    static void checkEGLEnvironment() {
        std::cout << "\n=== EGL环境检查 ===" << std::endl;
        
        // 检查EGL相关环境变量
        std::vector<std::string> egl_vars = {
            "EGL_PLATFORM",
            "EGL_DISPLAY", 
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "__EGL_VENDOR_LIBRARY_DIRS"
        };
        
        for (const auto& var : egl_vars) {
            const char* value = getenv(var.c_str());
            if (value) {
                std::cout << var << "=" << value << std::endl;
            } else {
                std::cout << var << " 未设置" << std::endl;
            }
        }
        
        // 测试EGL初始化
        EGLDisplay display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        if (display != EGL_NO_DISPLAY) {
            EGLint major, minor;
            if (eglInitialize(display, &major, &minor)) {
                std::cout << "EGL初始化成功: " << major << "." << minor << std::endl;
                eglTerminate(display);
            } else {
                std::cout << "EGL初始化失败: 0x" << std::hex << eglGetError() << std::endl;
            }
        } else {
            std::cout << "无法获取EGL Display" << std::endl;
        }
    }
    
    static void checkPermissions() {
        std::cout << "\n=== 权限检查 ===" << std::endl;
        
        // 检查用户组
        std::cout << "当前用户组:" << std::endl;
        system("groups");
        
        // 检查关键设备文件权限
        std::vector<std::string> devices = {
            "/dev/video0",
            "/dev/nvhost-vi", 
            "/dev/nvhost-isp",
            "/dev/nvhost-nvcsi"
        };
        
        for (const auto& device : devices) {
            struct stat buffer;
            if (stat(device.c_str(), &buffer) == 0) {
                std::cout << device << " 权限: " << std::oct << (buffer.st_mode & 0777) << std::endl;
            }
        }
    }
    
    static void testBasicGStreamer() {
        std::cout << "\n=== GStreamer基础测试 ===" << std::endl;
        
        gst_init(nullptr, nullptr);
        
        // 测试 nvarguscamerasrc 元素创建
        GstElement* source = gst_element_factory_make("nvarguscamerasrc", "test-source");
        if (source) {
            std::cout << "nvarguscamerasrc元素创建成功" << std::endl;
            
            // 设置基本属性
            g_object_set(source, "sensor-id", 0, nullptr);
            g_object_set(source, "bufapi-version", TRUE, nullptr);
            
            // 测试状态变更
            GstStateChangeReturn ret = gst_element_set_state(source, GST_STATE_READY);
            switch (ret) {
                case GST_STATE_CHANGE_SUCCESS:
                    std::cout << "状态变更至READY成功" << std::endl;
                    break;
                case GST_STATE_CHANGE_ASYNC:
                    std::cout << "状态变更至READY异步进行中" << std::endl;
                    break;
                case GST_STATE_CHANGE_FAILURE:
                    std::cout << "状态变更至READY失败" << std::endl;
                    break;
                default:
                    std::cout << "状态变更未知结果" << std::endl;
            }
            
            gst_element_set_state(source, GST_STATE_NULL);
            gst_object_unref(source);
        } else {
            std::cout << "nvarguscamerasrc元素创建失败" << std::endl;
        }
    }
    
    static void suggestFixes() {
        std::cout << "\n=== 修复建议 ===" << std::endl;
        
        std::cout << "1. 停止可能冲突的进程:" << std::endl;
        std::cout << "   sudo pkill nvargus-daemon" << std::endl;
        std::cout << "   sudo pkill gst-launch-1.0" << std::endl;
        
        std::cout << "\n2. 重启Argus服务:" << std::endl;
        std::cout << "   sudo systemctl restart nvargus-daemon" << std::endl;
        
        std::cout << "\n3. 检查用户权限:" << std::endl;
        std::cout << "   sudo usermod -a -G video $USER" << std::endl;
        std::cout << "   sudo usermod -a -G camera $USER" << std::endl;
        
        std::cout << "\n4. 设置EGL环境变量:" << std::endl;
        std::cout << "   export EGL_PLATFORM=drm" << std::endl;
        std::cout << "   export __EGL_VENDOR_LIBRARY_DIRS=/usr/lib/aarch64-linux-gnu/tegra-egl" << std::endl;
        
        std::cout << "\n5. 尝试不同的sensor-id:" << std::endl;
        std::cout << "   sensor-id=0 (默认)" << std::endl;
        std::cout << "   sensor-id=1 (备选)" << std::endl;
    }

public:
    static void runFullDiagnostics() {
        std::cout << "=== NVARGUSCAMERASRC 诊断工具 ===" << std::endl;
        
        checkCameraDevices();
        checkProcesses();
        checkEGLEnvironment();
        checkPermissions();
        testBasicGStreamer();
        suggestFixes();
        
        std::cout << "\n=== 诊断完成 ===" << std::endl;
    }
    
    static void testCameraAccess(int sensor_id = 0) {
        std::cout << "\n=== 摄像头访问测试 (sensor-id=" << sensor_id << ") ===" << std::endl;
        
        gst_init(nullptr, nullptr);
        
        // 创建测试管道
        std::string pipeline_str = 
            "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) + 
            " bufapi-version=1 ! "
            "video/x-raw(memory:NVMM), width=640, height=480, framerate=30/1 ! "
            "fakesink";
            
        std::cout << "测试管道: " << pipeline_str << std::endl;
        
        GError* error = nullptr;
        GstElement* pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
        
        if (error) {
            std::cout << "管道创建失败: " << error->message << std::endl;
            g_error_free(error);
            return;
        }
        
        if (!pipeline) {
            std::cout << "管道创建失败" << std::endl;
            return;
        }
        
        std::cout << "管道创建成功，测试状态变更..." << std::endl;
        
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        
        if (ret == GST_STATE_CHANGE_FAILURE) {
            std::cout << "启动失败" << std::endl;
        } else {
            std::cout << "启动成功，运行5秒..." << std::endl;
            
            GstBus* bus = gst_element_get_bus(pipeline);
            GstMessage* msg = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND, 
                static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            
            if (msg) {
                if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                    GError* error;
                    gchar* debug_info;
                    gst_message_parse_error(msg, &error, &debug_info);
                    std::cout << "错误: " << error->message << std::endl;
                    if (debug_info) {
                        std::cout << "调试信息: " << debug_info << std::endl;
                        g_free(debug_info);
                    }
                    g_error_free(error);
                }
                gst_message_unref(msg);
            } else {
                std::cout << "测试成功完成" << std::endl;
            }
            
            gst_object_unref(bus);
        }
        
        gst_element_set_state(pipeline, GST_STATE_NULL);
        gst_object_unref(pipeline);
    }
};

int main(int argc, char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "test") {
        int sensor_id = 0;
        if (argc > 2) {
            sensor_id = std::stoi(argv[2]);
        }
        CameraDiagnostics::testCameraAccess(sensor_id);
    } else {
        CameraDiagnostics::runFullDiagnostics();
    }
    
    return 0;
}