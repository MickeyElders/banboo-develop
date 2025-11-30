/**
 * @file lvgl_drm_interface.cpp
 * @brief LVGL DRM/GBM 接口实现（无合成器），共享 EGL/GBM/EGLStream
 */

#include "bamboo_cut/ui/lvgl_wayland_interface.h" // 复用相同头定义
#include "bamboo_cut/ui/egl_context_manager.h"
#include "bamboo_cut/ui/lvgl_ui_utils.h"
#include "bamboo_cut/ui/ui_context.h"
#include "bamboo_cut/ui/ui_components.h"
#include "bamboo_cut/ui/ui_updaters.h"
#include "bamboo_cut/utils/logger.h"
#include "bamboo_cut/core/data_bridge.h"
#include "bamboo_cut/utils/jetson_monitor.h"

#include <lvgl.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

namespace bamboo_cut {
namespace ui {

class LVGLWaylandInterface::Impl {
public:
    LVGLWaylandConfig config_;
    UIContext ui_ctx_;

    lv_display_t* display_ = nullptr;
    lv_obj_t* main_screen_ = nullptr;

    lv_color_t* front_buffer_ = nullptr;
    lv_color_t* back_buffer_ = nullptr;
    uint32_t* full_frame_buffer_ = nullptr;

    EGLDisplay egl_display_ = EGL_NO_DISPLAY;
    EGLContext egl_context_ = EGL_NO_CONTEXT;
    EGLSurface egl_surface_ = EGL_NO_SURFACE;
    EGLConfig egl_config_{};
    EGLStreamKHR egl_stream_ = EGL_NO_STREAM_KHR;

    GLuint shader_program_ = 0;
    GLuint texture_id_ = 0;
    GLuint vbo_ = 0;
    bool gl_resources_initialized_ = false;

    std::mutex ui_mutex_;
    std::atomic<bool> should_stop_{false};
    std::atomic<bool> ui_created_{false};

    std::thread ui_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> fully_initialized_{false};

    std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge_;
    std::shared_ptr<bamboo_cut::utils::JetsonMonitor> jetson_monitor_;
    int fps_frame_count_ = 0;
    std::chrono::steady_clock::time_point fps_last_update_{std::chrono::steady_clock::now()};

    Impl() = default;
    ~Impl() { cleanup(); }

    bool initializeDrmDisplay();
    void uiThreadLoop();
    void flushDisplay(const lv_area_t* area, lv_color_t* color_p);
    bool initializeGLResources();
    void cleanupGLResources();
    bool createShaderProgram();
    void cleanup();
};

LVGLWaylandInterface::LVGLWaylandInterface(std::shared_ptr<bamboo_cut::core::DataBridge> data_bridge)
    : pImpl_(std::make_unique<Impl>()) {
    pImpl_->data_bridge_ = data_bridge;
    pImpl_->jetson_monitor_ = std::make_shared<bamboo_cut::utils::JetsonMonitor>();
}

LVGLWaylandInterface::~LVGLWaylandInterface() {
    stop();
}

bool LVGLWaylandInterface::initialize(const LVGLWaylandConfig& config) {
    pImpl_->config_ = config;

    if (!lv_is_initialized()) {
        lv_init();
    }

    if (!pImpl_->initializeDrmDisplay()) {
        std::cerr << "[LVGL DRM] 显示初始化失败" << std::endl;
        return false;
    }

    pImpl_->initializeGLResources();

    // 构建 UI（复刻原 Wayland UI）
    if (!build_full_ui(pImpl_->ui_ctx_, config, false)) {
        std::cerr << "[LVGL DRM] UI 构建失败" << std::endl;
        return false;
    }

    if (pImpl_->jetson_monitor_) {
        pImpl_->jetson_monitor_->start();
    }

    pImpl_->ui_created_.store(true);
    pImpl_->fully_initialized_.store(true);
    return true;
}

bool LVGLWaylandInterface::start() {
    if (pImpl_->running_.load()) {
        return true;
    }
    pImpl_->should_stop_.store(false);
    pImpl_->ui_thread_ = std::thread(&LVGLWaylandInterface::Impl::uiThreadLoop, pImpl_.get());
    pImpl_->running_.store(true);
    return true;
}

void LVGLWaylandInterface::stop() {
    if (!pImpl_ || !pImpl_->running_.load()) return;
    pImpl_->should_stop_.store(true);
    if (pImpl_->ui_thread_.joinable()) {
        pImpl_->ui_thread_.join();
    }
    pImpl_->running_.store(false);
    pImpl_->fully_initialized_.store(false);
}

bool LVGLWaylandInterface::isFullyInitialized() const {
    return pImpl_ && pImpl_->fully_initialized_.load();
}

bool LVGLWaylandInterface::isRunning() const {
    return pImpl_ && pImpl_->running_.load();
}

lv_obj_t* LVGLWaylandInterface::getCameraCanvas() {
    return nullptr; // DRM 版本不使用 canvas 合成
}

void LVGLWaylandInterface::updateCameraCanvas(const cv::Mat&) {
    // no-op
}

bool LVGLWaylandInterface::isWaylandEnvironmentAvailable() {
    return false;
}

void* LVGLWaylandInterface::getImpl() { return pImpl_.get(); }
void* LVGLWaylandInterface::getWaylandDisplay() { return nullptr; }
void* LVGLWaylandInterface::getWaylandCompositor() { return nullptr; }
void* LVGLWaylandInterface::getWaylandSubcompositor() { return nullptr; }
void* LVGLWaylandInterface::getWaylandSurface() { return nullptr; }

LVGLWaylandInterface::SubsurfaceHandle LVGLWaylandInterface::createSubsurface(int, int, int, int) {
    return {};
}

void LVGLWaylandInterface::destroySubsurface(SubsurfaceHandle) {}

bool LVGLWaylandInterface::getCameraPanelCoords(int& x, int& y, int& width, int& height) {
    if (!pImpl_ || !pImpl_->ui_ctx_.camera_panel) {
        return false;
    }
    lv_obj_t* panel = pImpl_->ui_ctx_.camera_panel;
    x = lv_obj_get_x(panel);
    y = lv_obj_get_y(panel);
    width = lv_obj_get_width(panel);
    height = lv_obj_get_height(panel);
    return true;
}

// Impl methods
bool LVGLWaylandInterface::Impl::initializeDrmDisplay() {
    std::cout << "[LVGL DRM] Creating LVGL display: " << config_.screen_width << "x" << config_.screen_height << std::endl;
    display_ = lv_display_create(config_.screen_width, config_.screen_height);
    if (!display_) return false;

    size_t full_frame_size = config_.screen_width * config_.screen_height * sizeof(uint32_t);
    full_frame_buffer_ = static_cast<uint32_t*>(malloc(full_frame_size));
    if (!full_frame_buffer_) return false;
    memset(full_frame_buffer_, 0x1A, full_frame_size);
    std::cout << "[LVGL DRM] Allocated full frame buffer size=" << full_frame_size << " bytes" << std::endl;

    size_t partial_buffer_size = (config_.screen_width * config_.screen_height / 10) * sizeof(lv_color_t);
    front_buffer_ = static_cast<lv_color_t*>(malloc(partial_buffer_size));
    back_buffer_ = static_cast<lv_color_t*>(malloc(partial_buffer_size));
    if (!front_buffer_ || !back_buffer_) return false;
    std::cout << "[LVGL DRM] Allocated partial buffers size=" << partial_buffer_size << " bytes each" << std::endl;

    lv_display_set_buffers(display_, front_buffer_, back_buffer_,
                           partial_buffer_size, LV_DISPLAY_RENDER_MODE_PARTIAL);

    lv_display_set_user_data(display_, this);
    lv_display_set_flush_cb(display_, [](lv_display_t* disp, const lv_area_t* area, uint8_t* color_p) {
        auto* impl = static_cast<LVGLWaylandInterface::Impl*>(lv_display_get_user_data(disp));
        if (impl && impl->ui_created_) {
            impl->flushDisplay(area, reinterpret_cast<lv_color_t*>(color_p));
        }
        lv_display_flush_ready(disp);
    });

    auto& egl_manager = bamboo_cut::ui::EGLContextManager::getInstance();
    if (!egl_manager.ensureInitialized(nullptr, nullptr, config_.screen_width, config_.screen_height)) {
        std::cerr << "[LVGL DRM] 共享 EGL 初始化失败" << std::endl;
        return false;
    }
    egl_display_ = egl_manager.getDisplay();
    egl_context_ = egl_manager.getContext();
    egl_surface_ = egl_manager.getSurface();
    egl_config_ = egl_manager.getConfig();
    egl_stream_ = egl_manager.getStream();

    if (eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_) != EGL_TRUE) {
        std::cerr << "[LVGL DRM] eglMakeCurrent 失败: 0x" << std::hex << eglGetError() << std::dec << std::endl;
        return false;
    }
    std::cout << "[LVGL DRM] EGL current: display=" << egl_display_ << " surface=" << egl_surface_ << " context=" << egl_context_ << std::endl;

    glViewport(0, 0, config_.screen_width, config_.screen_height);
    glClearColor(0.11f, 0.12f, 0.14f, 1.0f);
    return true;
}

void LVGLWaylandInterface::Impl::uiThreadLoop() {
    const auto frame_time = std::chrono::milliseconds(1000 / config_.refresh_rate);
    const auto data_interval = std::chrono::milliseconds(500);
    auto last_data_update = std::chrono::steady_clock::now();
    while (!should_stop_.load()) {
        {
            std::lock_guard<std::mutex> lock(ui_mutex_);
            lv_timer_handler();
        }
        auto now = std::chrono::steady_clock::now();
        if (now - last_data_update >= data_interval) {
            update_ui_data(ui_ctx_, data_bridge_, jetson_monitor_, fps_frame_count_, fps_last_update_);
            last_data_update = now;
        }
        std::this_thread::sleep_for(frame_time);
    }
}

void LVGLWaylandInterface::Impl::flushDisplay(const lv_area_t* area, lv_color_t* color_p) {
    if (!egl_display_ || !egl_surface_ || !egl_context_ || !full_frame_buffer_) return;

    const int width = config_.screen_width;
    const int area_width = area->x2 - area->x1 + 1;

#if LV_COLOR_DEPTH == 32
    const uint32_t* src = reinterpret_cast<const uint32_t*>(color_p);
    for (int y = area->y1; y <= area->y2; ++y) {
        uint32_t* dst = full_frame_buffer_ + y * width + area->x1;
        memcpy(dst, src + (y - area->y1) * area_width, area_width * sizeof(uint32_t));
    }
#else
#error "Only LV_COLOR_DEPTH=32 supported in DRM path"
#endif

    if (eglMakeCurrent(egl_display_, egl_surface_, egl_surface_, egl_context_) != EGL_TRUE) {
        return;
    }

    if (!gl_resources_initialized_) {
        if (!initializeGLResources()) return;
    }

    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, config_.screen_height, GL_BGRA_EXT, GL_UNSIGNED_BYTE, full_frame_buffer_);

    glViewport(0, 0, width, config_.screen_height);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shader_program_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    GLint pos = glGetAttribLocation(shader_program_, "aPos");
    GLint tex = glGetAttribLocation(shader_program_, "aTex");
    if (pos >= 0) {
        glEnableVertexAttribArray(pos);
        glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)0);
    }
    if (tex >= 0) {
        glEnableVertexAttribArray(tex);
        glVertexAttribPointer(tex, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (void*)(2 * sizeof(GLfloat)));
    }

    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    if (pos >= 0) glDisableVertexAttribArray(pos);
    if (tex >= 0) glDisableVertexAttribArray(tex);

    eglSwapBuffers(egl_display_, egl_surface_);
}

bool LVGLWaylandInterface::Impl::initializeGLResources() {
    if (gl_resources_initialized_) return true;
    if (!createShaderProgram()) return false;

    static const GLfloat vertices[] = {
        -1.0f, -1.0f, 0.0f, 1.0f,
         1.0f, -1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 0.0f,
    };

    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glGenTextures(1, &texture_id_);
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, config_.screen_width, config_.screen_height, 0,
                 GL_BGRA_EXT, GL_UNSIGNED_BYTE, full_frame_buffer_);

    GLint sampler = glGetUniformLocation(shader_program_, "uTex");
    if (sampler >= 0) glUniform1i(sampler, 0);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    gl_resources_initialized_ = true;
    return true;
}

void LVGLWaylandInterface::Impl::cleanupGLResources() {
    if (texture_id_) glDeleteTextures(1, &texture_id_);
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (shader_program_) glDeleteProgram(shader_program_);
    texture_id_ = 0;
    vbo_ = 0;
    shader_program_ = 0;
    gl_resources_initialized_ = false;
}

bool LVGLWaylandInterface::Impl::createShaderProgram() {
    const char* vs_src =
        "attribute vec2 aPos;\n"
        "attribute vec2 aTex;\n"
        "varying vec2 vTex;\n"
        "void main(){ gl_Position=vec4(aPos,0.0,1.0); vTex=vec2(aTex.x,1.0-aTex.y); }\n";
    const char* fs_src =
        "precision mediump float;\n"
        "varying vec2 vTex;\n"
        "uniform sampler2D uTex;\n"
        "void main(){ gl_FragColor = texture2D(uTex, vTex); }\n";

    auto compile = [](GLenum type, const char* src) -> GLuint {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            glDeleteShader(s);
            return 0;
        }
        return s;
    };

    GLuint vs = compile(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return false;

    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vs);
    glAttachShader(shader_program_, fs);
    glBindAttribLocation(shader_program_, 0, "aPos");
    glBindAttribLocation(shader_program_, 1, "aTex");
    glLinkProgram(shader_program_);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint linked = 0;
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &linked);
    if (!linked) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
        return false;
    }
    glUseProgram(shader_program_);
    return true;
}

void LVGLWaylandInterface::Impl::cleanup() {
    cleanupGLResources();
    if (display_) {
        lv_display_delete(display_);
        display_ = nullptr;
    }
    if (front_buffer_) free(front_buffer_);
    if (back_buffer_) free(back_buffer_);
    if (full_frame_buffer_) free(full_frame_buffer_);
    front_buffer_ = back_buffer_ = nullptr;
    full_frame_buffer_ = nullptr;
}

}  // namespace ui
}  // namespace bamboo_cut
