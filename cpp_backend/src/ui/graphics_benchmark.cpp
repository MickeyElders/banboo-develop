/**
 * @file graphics_benchmark.cpp
 * @brief LVGL图形性能基准测试
 * 用于验证nvidia-drm驱动下的图形性能
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <numeric>

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#endif

namespace bamboo_cut {
namespace ui {

class GraphicsBenchmark {
public:
    GraphicsBenchmark() : test_running_(false), frame_count_(0) {}
    
    bool initialize() {
        #ifdef ENABLE_LVGL
        std::cout << "🔧 初始化图形性能基准测试..." << std::endl;
        
        // 创建测试界面
        createBenchmarkUI();
        
        std::cout << "✅ 性能测试界面创建完成" << std::endl;
        return true;
        #else
        std::cout << "❌ LVGL未启用，跳过性能测试" << std::endl;
        return false;
        #endif
    }
    
    void runBenchmark() {
        #ifdef ENABLE_LVGL
        std::cout << "🚀 开始图形性能基准测试..." << std::endl;
        
        // 1. 基础渲染性能测试
        runBasicRenderTest();
        
        // 2. 复杂界面渲染测试
        runComplexUITest();
        
        // 3. 动画性能测试
        runAnimationTest();
        
        // 4. 内存使用测试
        runMemoryTest();
        
        // 5. 帧率稳定性测试
        runFrameRateTest();
        
        showBenchmarkResults();
        #endif
    }

private:
    #ifdef ENABLE_LVGL
    void createBenchmarkUI() {
        // 创建测试容器
        benchmark_container_ = lv_obj_create(lv_screen_active());
        lv_obj_set_size(benchmark_container_, LV_HOR_RES, LV_VER_RES);
        lv_obj_set_style_bg_color(benchmark_container_, lv_color_hex(0x2E3440), 0);
        lv_obj_set_style_border_width(benchmark_container_, 0, 0);
        
        // 创建标题
        lv_obj_t* title = lv_label_create(benchmark_container_);
        lv_label_set_text(title, "⚡ LVGL图形性能基准测试");
        lv_obj_set_style_text_font(title, &lv_font_montserrat_24, 0);
        lv_obj_set_style_text_color(title, lv_color_hex(0xECEFF4), 0);
        lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 20);
        
        // 创建测试状态显示
        status_label_ = lv_label_create(benchmark_container_);
        lv_label_set_text(status_label_, "状态: 准备开始");
        lv_obj_set_style_text_color(status_label_, lv_color_hex(0xD08770), 0);
        lv_obj_align(status_label_, LV_ALIGN_TOP_MID, 0, 60);
        
        // 创建性能指标显示区域
        createMetricsDisplay();
    }
    
    void createMetricsDisplay() {
        // 性能指标背景
        metrics_area_ = lv_obj_create(benchmark_container_);
        lv_obj_set_size(metrics_area_, LV_HOR_RES - 100, 200);
        lv_obj_align(metrics_area_, LV_ALIGN_CENTER, 0, 0);
        lv_obj_set_style_bg_color(metrics_area_, lv_color_hex(0x3B4252), 0);
        lv_obj_set_style_border_color(metrics_area_, lv_color_hex(0x434C5E), 0);
        lv_obj_set_style_border_width(metrics_area_, 1, 0);
        
        // FPS显示
        fps_label_ = lv_label_create(metrics_area_);
        lv_label_set_text(fps_label_, "FPS: --");
        lv_obj_set_style_text_color(fps_label_, lv_color_hex(0xA3BE8C), 0);
        lv_obj_align(fps_label_, LV_ALIGN_TOP_LEFT, 20, 20);
        
        // 渲染时间显示
        render_time_label_ = lv_label_create(metrics_area_);
        lv_label_set_text(render_time_label_, "渲染时间: -- ms");
        lv_obj_set_style_text_color(render_time_label_, lv_color_hex(0x88C0D0), 0);
        lv_obj_align(render_time_label_, LV_ALIGN_TOP_RIGHT, -20, 20);
        
        // 内存使用显示
        memory_label_ = lv_label_create(metrics_area_);
        lv_label_set_text(memory_label_, "内存使用: -- MB");
        lv_obj_set_style_text_color(memory_label_, lv_color_hex(0xEBCB8B), 0);
        lv_obj_align(memory_label_, LV_ALIGN_TOP_LEFT, 20, 60);
        
        // GPU使用率显示
        gpu_label_ = lv_label_create(metrics_area_);
        lv_label_set_text(gpu_label_, "GPU使用率: --%");
        lv_obj_set_style_text_color(gpu_label_, lv_color_hex(0xB48EAD), 0);
        lv_obj_align(gpu_label_, LV_ALIGN_TOP_RIGHT, -20, 60);
        
        // 测试进度显示
        progress_bar_ = lv_bar_create(metrics_area_);
        lv_obj_set_size(progress_bar_, LV_HOR_RES - 180, 20);
        lv_obj_align(progress_bar_, LV_ALIGN_BOTTOM_MID, 0, -20);
        lv_bar_set_range(progress_bar_, 0, 100);
        lv_bar_set_value(progress_bar_, 0, LV_ANIM_OFF);
        
        // 进度标签
        progress_label_ = lv_label_create(metrics_area_);
        lv_label_set_text(progress_label_, "测试进度: 0%");
        lv_obj_set_style_text_color(progress_label_, lv_color_hex(0xECEFF4), 0);
        lv_obj_align(progress_label_, LV_ALIGN_BOTTOM_MID, 0, -50);
    }
    
    void runBasicRenderTest() {
        std::cout << "📊 执行基础渲染性能测试..." << std::endl;
        updateStatus("基础渲染测试中...");
        
        auto start_time = std::chrono::high_resolution_clock::now();
        const int test_frames = 60; // 测试60帧
        
        for (int i = 0; i < test_frames; i++) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 执行基础渲染操作
            lv_task_handler();
            lv_refr_now(nullptr);
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(
                frame_end - frame_start).count();
            
            basic_render_times_.push_back(frame_time);
            
            // 更新进度
            updateProgress(i * 20 / test_frames);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // 60fps
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        basic_fps_ = (test_frames * 1000.0) / total_time;
        
        std::cout << "✅ 基础渲染测试完成 - FPS: " << basic_fps_ << std::endl;
    }
    
    void runComplexUITest() {
        std::cout << "🎨 执行复杂界面渲染测试..." << std::endl;
        updateStatus("复杂界面测试中...");
        
        // 创建复杂界面元素
        std::vector<lv_obj_t*> test_objects;
        
        // 创建多个按钮
        for (int i = 0; i < 20; i++) {
            lv_obj_t* btn = lv_btn_create(benchmark_container_);
            lv_obj_set_size(btn, 80, 40);
            lv_obj_set_pos(btn, (i % 10) * 90 + 50, (i / 10) * 50 + 400);
            lv_obj_set_style_bg_color(btn, lv_color_hex(0x5E81AC), 0);
            
            lv_obj_t* label = lv_label_create(btn);
            lv_label_set_text_fmt(label, "B%d", i);
            lv_obj_center(label);
            
            test_objects.push_back(btn);
        }
        
        // 测试复杂界面渲染性能
        auto start_time = std::chrono::high_resolution_clock::now();
        const int test_frames = 30;
        
        for (int i = 0; i < test_frames; i++) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 更新界面状态
            for (auto obj : test_objects) {
                lv_obj_set_style_bg_color(obj, 
                    lv_color_hex(0x5E81AC + (i * 0x1000)), 0);
            }
            
            lv_task_handler();
            lv_refr_now(nullptr);
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(
                frame_end - frame_start).count();
            
            complex_render_times_.push_back(frame_time);
            
            // 更新进度
            updateProgress(20 + i * 20 / test_frames);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(33)); // 30fps
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        complex_fps_ = (test_frames * 1000.0) / total_time;
        
        // 清理测试对象
        for (auto obj : test_objects) {
            lv_obj_del(obj);
        }
        
        std::cout << "✅ 复杂界面测试完成 - FPS: " << complex_fps_ << std::endl;
    }
    
    void runAnimationTest() {
        std::cout << "🎬 执行动画性能测试..." << std::endl;
        updateStatus("动画性能测试中...");
        
        // 创建动画测试对象
        lv_obj_t* anim_obj = lv_obj_create(benchmark_container_);
        lv_obj_set_size(anim_obj, 100, 100);
        lv_obj_set_style_bg_color(anim_obj, lv_color_hex(0xA3BE8C), 0);
        lv_obj_set_style_radius(anim_obj, 50, 0);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        const int animation_frames = 60;
        
        for (int i = 0; i < animation_frames; i++) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 执行动画更新
            int x = 100 + (int)(200 * sin(i * 0.1));
            int y = 300 + (int)(100 * cos(i * 0.1));
            lv_obj_set_pos(anim_obj, x, y);
            
            // 旋转效果（通过颜色变化模拟）
            lv_color_t color = lv_color_hsv_to_rgb(i * 6, 100, 100);
            lv_obj_set_style_bg_color(anim_obj, color, 0);
            
            lv_task_handler();
            lv_refr_now(nullptr);
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(
                frame_end - frame_start).count();
            
            animation_times_.push_back(frame_time);
            
            // 更新进度
            updateProgress(40 + i * 20 / animation_frames);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        animation_fps_ = (animation_frames * 1000.0) / total_time;
        
        lv_obj_del(anim_obj);
        
        std::cout << "✅ 动画测试完成 - FPS: " << animation_fps_ << std::endl;
    }
    
    void runMemoryTest() {
        std::cout << "💾 执行内存使用测试..." << std::endl;
        updateStatus("内存使用测试中...");
        
        // 获取初始内存使用
        size_t initial_memory = getCurrentMemoryUsage();
        
        // 创建大量对象进行内存压力测试
        std::vector<lv_obj_t*> memory_objects;
        
        for (int i = 0; i < 100; i++) {
            lv_obj_t* obj = lv_obj_create(benchmark_container_);
            lv_obj_set_size(obj, 50, 50);
            lv_obj_set_pos(obj, rand() % 1000, rand() % 600);
            memory_objects.push_back(obj);
            
            // 更新进度
            updateProgress(60 + i * 20 / 100);
            
            if (i % 10 == 0) {
                lv_task_handler();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        
        size_t peak_memory = getCurrentMemoryUsage();
        memory_usage_ = peak_memory - initial_memory;
        
        // 清理对象
        for (auto obj : memory_objects) {
            lv_obj_del(obj);
        }
        
        lv_task_handler();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        size_t final_memory = getCurrentMemoryUsage();
        memory_leak_ = final_memory - initial_memory;
        
        std::cout << "✅ 内存测试完成 - 峰值使用: " << memory_usage_ << " KB" << std::endl;
    }
    
    void runFrameRateTest() {
        std::cout << "📊 执行帧率稳定性测试..." << std::endl;
        updateStatus("帧率稳定性测试中...");
        
        const int test_duration = 5; // 5秒测试
        auto start_time = std::chrono::high_resolution_clock::now();
        frame_times_.clear();
        
        while (true) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            lv_task_handler();
            lv_refr_now(nullptr);
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                frame_end - start_time).count();
            
            if (elapsed >= test_duration) break;
            
            auto frame_time = std::chrono::duration_cast<std::chrono::microseconds>(
                frame_end - frame_start).count();
            frame_times_.push_back(frame_time);
            
            // 更新进度
            updateProgress(80 + (elapsed * 20) / test_duration);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        // 计算帧率统计
        calculateFrameRateStats();
        
        std::cout << "✅ 帧率测试完成 - 平均FPS: " << average_fps_ << std::endl;
    }
    
    void calculateFrameRateStats() {
        if (frame_times_.empty()) return;
        
        // 计算平均帧时间
        double total_time = std::accumulate(frame_times_.begin(), frame_times_.end(), 0.0);
        double avg_frame_time = total_time / frame_times_.size();
        average_fps_ = 1000000.0 / avg_frame_time; // 微秒转FPS
        
        // 计算帧时间标准差
        double variance = 0;
        for (auto time : frame_times_) {
            variance += (time - avg_frame_time) * (time - avg_frame_time);
        }
        frame_time_std_ = sqrt(variance / frame_times_.size());
        
        // 计算最小/最大帧时间
        auto minmax = std::minmax_element(frame_times_.begin(), frame_times_.end());
        min_frame_time_ = *minmax.first;
        max_frame_time_ = *minmax.second;
    }
    
    size_t getCurrentMemoryUsage() {
        // 简化的内存使用估算（在实际项目中应该使用系统API）
        FILE* file = fopen("/proc/self/status", "r");
        if (!file) return 0;
        
        char line[128];
        size_t memory_kb = 0;
        
        while (fgets(line, 128, file) != nullptr) {
            if (strncmp(line, "VmRSS:", 6) == 0) {
                sscanf(line, "VmRSS: %zu kB", &memory_kb);
                break;
            }
        }
        
        fclose(file);
        return memory_kb;
    }
    
    void updateStatus(const char* status) {
        #ifdef ENABLE_LVGL
        lv_label_set_text(status_label_, status);
        lv_task_handler();
        #endif
    }
    
    void updateProgress(int progress) {
        #ifdef ENABLE_LVGL
        lv_bar_set_value(progress_bar_, progress, LV_ANIM_ON);
        
        char progress_text[50];
        snprintf(progress_text, sizeof(progress_text), "测试进度: %d%%", progress);
        lv_label_set_text(progress_label_, progress_text);
        
        lv_task_handler();
        #endif
    }
    
    void showBenchmarkResults() {
        updateStatus("测试完成 - 生成报告中...");
        updateProgress(100);
        
        std::cout << "\n🎯 图形性能基准测试结果:" << std::endl;
        std::cout << "================================" << std::endl;
        
        // 基础渲染性能
        std::cout << "📊 基础渲染性能:" << std::endl;
        std::cout << "   FPS: " << basic_fps_ << std::endl;
        if (!basic_render_times_.empty()) {
            double avg_render_time = std::accumulate(basic_render_times_.begin(), 
                basic_render_times_.end(), 0.0) / basic_render_times_.size() / 1000.0;
            std::cout << "   平均渲染时间: " << avg_render_time << " ms" << std::endl;
        }
        
        // 复杂界面性能
        std::cout << "\n🎨 复杂界面性能:" << std::endl;
        std::cout << "   FPS: " << complex_fps_ << std::endl;
        
        // 动画性能
        std::cout << "\n🎬 动画性能:" << std::endl;
        std::cout << "   FPS: " << animation_fps_ << std::endl;
        
        // 内存使用
        std::cout << "\n💾 内存使用:" << std::endl;
        std::cout << "   峰值使用: " << memory_usage_ << " KB" << std::endl;
        std::cout << "   内存泄漏: " << memory_leak_ << " KB" << std::endl;
        
        // 帧率稳定性
        std::cout << "\n📊 帧率稳定性:" << std::endl;
        std::cout << "   平均FPS: " << average_fps_ << std::endl;
        std::cout << "   帧时间标准差: " << frame_time_std_ / 1000.0 << " ms" << std::endl;
        std::cout << "   最小帧时间: " << min_frame_time_ / 1000.0 << " ms" << std::endl;
        std::cout << "   最大帧时间: " << max_frame_time_ / 1000.0 << " ms" << std::endl;
        
        // 性能评估
        std::cout << "\n🏆 性能评估:" << std::endl;
        evaluatePerformance();
        
        // 更新界面显示最终结果
        char fps_text[50];
        snprintf(fps_text, sizeof(fps_text), "平均FPS: %.1f", average_fps_);
        lv_label_set_text(fps_label_, fps_text);
        
        char memory_text[50];
        snprintf(memory_text, sizeof(memory_text), "内存使用: %zu KB", memory_usage_);
        lv_label_set_text(memory_label_, memory_text);
        
        updateStatus("✅ 所有测试完成");
    }
    
    void evaluatePerformance() {
        int score = 0;
        
        // FPS评分
        if (average_fps_ >= 55) score += 25;
        else if (average_fps_ >= 45) score += 20;
        else if (average_fps_ >= 30) score += 15;
        else if (average_fps_ >= 20) score += 10;
        else score += 5;
        
        // 帧时间稳定性评分
        if (frame_time_std_ < 5000) score += 25; // < 5ms
        else if (frame_time_std_ < 10000) score += 20; // < 10ms
        else if (frame_time_std_ < 15000) score += 15; // < 15ms
        else score += 10;
        
        // 内存使用评分
        if (memory_usage_ < 1024) score += 25; // < 1MB
        else if (memory_usage_ < 2048) score += 20; // < 2MB
        else if (memory_usage_ < 4096) score += 15; // < 4MB
        else score += 10;
        
        // 内存泄漏评分
        if (memory_leak_ < 100) score += 25; // < 100KB
        else if (memory_leak_ < 500) score += 20; // < 500KB
        else if (memory_leak_ < 1024) score += 15; // < 1MB
        else score += 10;
        
        std::cout << "   综合评分: " << score << "/100" << std::endl;
        
        if (score >= 90) {
            std::cout << "   等级: 优秀 🌟🌟🌟" << std::endl;
            std::cout << "   nvidia-drm驱动性能表现优异！" << std::endl;
        } else if (score >= 75) {
            std::cout << "   等级: 良好 🌟🌟" << std::endl;
            std::cout << "   nvidia-drm驱动性能良好" << std::endl;
        } else if (score >= 60) {
            std::cout << "   等级: 及格 🌟" << std::endl;
            std::cout << "   nvidia-drm驱动性能基本满足要求" << std::endl;
        } else {
            std::cout << "   等级: 待优化" << std::endl;
            std::cout << "   建议优化图形性能配置" << std::endl;
        }
    }
    #endif

private:
    bool test_running_;
    int frame_count_;
    
    // 性能指标
    double basic_fps_;
    double complex_fps_;
    double animation_fps_;
    double average_fps_;
    double frame_time_std_;
    long min_frame_time_;
    long max_frame_time_;
    size_t memory_usage_;
    size_t memory_leak_;
    
    // 测试数据
    std::vector<long> basic_render_times_;
    std::vector<long> complex_render_times_;
    std::vector<long> animation_times_;
    std::vector<long> frame_times_;
    
    #ifdef ENABLE_LVGL
    lv_obj_t* benchmark_container_;
    lv_obj_t* status_label_;
    lv_obj_t* metrics_area_;
    lv_obj_t* fps_label_;
    lv_obj_t* render_time_label_;
    lv_obj_t* memory_label_;
    lv_obj_t* gpu_label_;
    lv_obj_t* progress_bar_;
    lv_obj_t* progress_label_;
    #endif
};

} // namespace ui
} // namespace bamboo_cut

// 独立测试函数
int main() {
    std::cout << "🚀 启动LVGL图形性能基准测试" << std::endl;
    
    bamboo_cut::ui::GraphicsBenchmark benchmark;
    
    if (!benchmark.initialize()) {
        std::cerr << "❌ 性能测试初始化失败" << std::endl;
        return -1;
    }
    
    benchmark.runBenchmark();
    
    std::cout << "✅ 图形性能基准测试完成" << std::endl;
    return 0;
}