/**
 * @file touch_test.cpp
 * @brief LVGL触摸交互响应测试
 * 用于验证nvidia-drm驱动下的触摸功能
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <chrono>
#include <thread>

#ifdef ENABLE_LVGL
#include <lvgl/lvgl.h>
#endif

namespace bamboo_cut {
namespace ui {

class TouchTest {
public:
    TouchTest() : test_running_(false), touch_count_(0) {}
    
    bool initialize() {
        #ifdef ENABLE_LVGL
        std::cout << "🔧 初始化触摸交互测试..." << std::endl;
        
        // 创建测试界面
        createTestUI();
        
        std::cout << "✅ 触摸测试界面创建完成" << std::endl;
        return true;
        #else
        std::cout << "❌ LVGL未启用，跳过触摸测试" << std::endl;
        return false;
        #endif
    }
    
    void runTest() {
        #ifdef ENABLE_LVGL
        test_running_ = true;
        std::cout << "🚀 开始触摸交互响应测试..." << std::endl;
        std::cout << "📱 请在屏幕上点击不同区域进行测试" << std::endl;
        
        auto start_time = std::chrono::steady_clock::now();
        const auto test_duration = std::chrono::seconds(30); // 30秒测试
        
        while (test_running_) {
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed = current_time - start_time;
            
            if (elapsed >= test_duration) {
                std::cout << "⏰ 测试时间到，自动结束测试" << std::endl;
                break;
            }
            
            // 更新LVGL事件处理
            lv_task_handler();
            
            // 更新测试状态显示
            updateTestStatus();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60fps
        }
        
        showTestResults();
        #endif
    }
    
    void stopTest() {
        test_running_ = false;
        std::cout << "🛑 触摸测试已停止" << std::endl;
    }

private:
    #ifdef ENABLE_LVGL
    void createTestUI() {
        // 创建测试容器
        test_container_ = lv_obj_create(lv_screen_active());
        lv_obj_set_size(test_container_, LV_HOR_RES, LV_VER_RES);
        lv_obj_set_style_bg_color(test_container_, lv_color_hex(0x2E3440), 0);
        lv_obj_set_style_border_width(test_container_, 0, 0);
        
        // 创建标题
        lv_obj_t* title = lv_label_create(test_container_);
        lv_label_set_text(title, "🔧 LVGL触摸交互测试");
        lv_obj_set_style_text_font(title, &lv_font_montserrat_24, 0);
        lv_obj_set_style_text_color(title, lv_color_hex(0xECEFF4), 0);
        lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 20);
        
        // 创建测试按钮区域
        createTestButtons();
        
        // 创建状态显示区域
        createStatusArea();
        
        // 创建触摸坐标显示
        createTouchDisplay();
    }
    
    void createTestButtons() {
        // 按钮1: 启动/停止按钮
        start_stop_btn_ = lv_btn_create(test_container_);
        lv_obj_set_size(start_stop_btn_, 150, 60);
        lv_obj_align(start_stop_btn_, LV_ALIGN_TOP_LEFT, 50, 80);
        lv_obj_set_style_bg_color(start_stop_btn_, lv_color_hex(0x5E81AC), 0);
        lv_obj_add_event_cb(start_stop_btn_, onStartStopClicked, LV_EVENT_CLICKED, this);
        
        lv_obj_t* btn1_label = lv_label_create(start_stop_btn_);
        lv_label_set_text(btn1_label, "开始测试");
        lv_obj_center(btn1_label);
        
        // 按钮2: 重置计数器
        reset_btn_ = lv_btn_create(test_container_);
        lv_obj_set_size(reset_btn_, 150, 60);
        lv_obj_align(reset_btn_, LV_ALIGN_TOP_MID, 0, 80);
        lv_obj_set_style_bg_color(reset_btn_, lv_color_hex(0xD08770), 0);
        lv_obj_add_event_cb(reset_btn_, onResetClicked, LV_EVENT_CLICKED, this);
        
        lv_obj_t* btn2_label = lv_label_create(reset_btn_);
        lv_label_set_text(btn2_label, "重置");
        lv_obj_center(btn2_label);
        
        // 按钮3: 退出测试
        exit_btn_ = lv_btn_create(test_container_);
        lv_obj_set_size(exit_btn_, 150, 60);
        lv_obj_align(exit_btn_, LV_ALIGN_TOP_RIGHT, -50, 80);
        lv_obj_set_style_bg_color(exit_btn_, lv_color_hex(0xBF616A), 0);
        lv_obj_add_event_cb(exit_btn_, onExitClicked, LV_EVENT_CLICKED, this);
        
        lv_obj_t* btn3_label = lv_label_create(exit_btn_);
        lv_label_set_text(btn3_label, "退出");
        lv_obj_center(btn3_label);
        
        // 创建触摸区域测试
        createTouchAreas();
    }
    
    void createTouchAreas() {
        // 左侧触摸区域
        left_touch_area_ = lv_obj_create(test_container_);
        lv_obj_set_size(left_touch_area_, 200, 150);
        lv_obj_align(left_touch_area_, LV_ALIGN_LEFT_MID, 50, 0);
        lv_obj_set_style_bg_color(left_touch_area_, lv_color_hex(0x8FBCBB), 0);
        lv_obj_set_style_border_color(left_touch_area_, lv_color_hex(0x88C0D0), 0);
        lv_obj_set_style_border_width(left_touch_area_, 2, 0);
        lv_obj_add_event_cb(left_touch_area_, onLeftAreaClicked, LV_EVENT_CLICKED, this);
        
        lv_obj_t* left_label = lv_label_create(left_touch_area_);
        lv_label_set_text(left_label, "左侧区域\n点击测试");
        lv_obj_center(left_label);
        
        // 右侧触摸区域
        right_touch_area_ = lv_obj_create(test_container_);
        lv_obj_set_size(right_touch_area_, 200, 150);
        lv_obj_align(right_touch_area_, LV_ALIGN_RIGHT_MID, -50, 0);
        lv_obj_set_style_bg_color(right_touch_area_, lv_color_hex(0xA3BE8C), 0);
        lv_obj_set_style_border_color(right_touch_area_, lv_color_hex(0x8FBCBB), 0);
        lv_obj_set_style_border_width(right_touch_area_, 2, 0);
        lv_obj_add_event_cb(right_touch_area_, onRightAreaClicked, LV_EVENT_CLICKED, this);
        
        lv_obj_t* right_label = lv_label_create(right_touch_area_);
        lv_label_set_text(right_label, "右侧区域\n点击测试");
        lv_obj_center(right_label);
    }
    
    void createStatusArea() {
        // 状态显示背景
        status_area_ = lv_obj_create(test_container_);
        lv_obj_set_size(status_area_, LV_HOR_RES - 100, 100);
        lv_obj_align(status_area_, LV_ALIGN_BOTTOM_MID, 0, -120);
        lv_obj_set_style_bg_color(status_area_, lv_color_hex(0x3B4252), 0);
        lv_obj_set_style_border_color(status_area_, lv_color_hex(0x434C5E), 0);
        lv_obj_set_style_border_width(status_area_, 1, 0);
        
        // 触摸计数器
        touch_counter_label_ = lv_label_create(status_area_);
        lv_label_set_text(touch_counter_label_, "触摸次数: 0");
        lv_obj_set_style_text_color(touch_counter_label_, lv_color_hex(0xECEFF4), 0);
        lv_obj_align(touch_counter_label_, LV_ALIGN_TOP_LEFT, 20, 10);
        
        // 响应时间显示
        response_time_label_ = lv_label_create(status_area_);
        lv_label_set_text(response_time_label_, "响应时间: -- ms");
        lv_obj_set_style_text_color(response_time_label_, lv_color_hex(0xECEFF4), 0);
        lv_obj_align(response_time_label_, LV_ALIGN_TOP_RIGHT, -20, 10);
        
        // 测试状态
        test_status_label_ = lv_label_create(status_area_);
        lv_label_set_text(test_status_label_, "状态: 等待开始");
        lv_obj_set_style_text_color(test_status_label_, lv_color_hex(0xD08770), 0);
        lv_obj_align(test_status_label_, LV_ALIGN_BOTTOM_LEFT, 20, -10);
    }
    
    void createTouchDisplay() {
        // 触摸坐标显示
        touch_coord_label_ = lv_label_create(test_container_);
        lv_label_set_text(touch_coord_label_, "坐标: (---, ---)");
        lv_obj_set_style_text_color(touch_coord_label_, lv_color_hex(0x88C0D0), 0);
        lv_obj_align(touch_coord_label_, LV_ALIGN_BOTTOM_MID, 0, -20);
        
        // 添加全屏触摸事件监听
        lv_obj_add_event_cb(lv_screen_active(), onScreenTouched, LV_EVENT_PRESSED, this);
        lv_obj_add_event_cb(lv_screen_active(), onScreenTouched, LV_EVENT_RELEASED, this);
    }
    
    void updateTestStatus() {
        if (!test_running_) return;
        
        // 更新触摸计数
        char count_text[50];
        snprintf(count_text, sizeof(count_text), "触摸次数: %d", touch_count_);
        lv_label_set_text(touch_counter_label_, count_text);
        
        // 更新测试状态
        lv_label_set_text(test_status_label_, "状态: 测试中...");
        lv_obj_set_style_text_color(test_status_label_, lv_color_hex(0xA3BE8C), 0);
    }
    
    void showTestResults() {
        std::cout << "\n🎯 触摸交互测试结果:" << std::endl;
        std::cout << "   📊 总触摸次数: " << touch_count_ << std::endl;
        std::cout << "   ⚡ 平均响应时间: " << (total_response_time_ / std::max(1, touch_count_)) << " ms" << std::endl;
        
        if (touch_count_ >= 5) {
            std::cout << "   ✅ 触摸响应测试通过 - 界面响应正常" << std::endl;
        } else {
            std::cout << "   ⚠️  触摸次数较少，建议进行更多测试" << std::endl;
        }
        
        // 更新界面状态
        lv_label_set_text(test_status_label_, "状态: 测试完成");
        lv_obj_set_style_text_color(test_status_label_, lv_color_hex(0x5E81AC), 0);
        
        char result_text[100];
        snprintf(result_text, sizeof(result_text), "测试完成 - 总计 %d 次触摸", touch_count_);
        lv_label_set_text(touch_counter_label_, result_text);
    }
    
    // 事件回调函数
    static void onStartStopClicked(lv_event_t* e) {
        TouchTest* test = static_cast<TouchTest*>(lv_event_get_user_data(e));
        if (test->test_running_) {
            test->stopTest();
        } else {
            test->test_running_ = true;
            std::cout << "▶️  触摸测试已开始" << std::endl;
        }
    }
    
    static void onResetClicked(lv_event_t* e) {
        TouchTest* test = static_cast<TouchTest*>(lv_event_get_user_data(e));
        test->touch_count_ = 0;
        test->total_response_time_ = 0;
        std::cout << "🔄 触摸计数器已重置" << std::endl;
    }
    
    static void onExitClicked(lv_event_t* e) {
        TouchTest* test = static_cast<TouchTest*>(lv_event_get_user_data(e));
        test->stopTest();
        std::cout << "🚪 退出触摸测试" << std::endl;
    }
    
    static void onLeftAreaClicked(lv_event_t* e) {
        TouchTest* test = static_cast<TouchTest*>(lv_event_get_user_data(e));
        test->recordTouch("左侧区域");
    }
    
    static void onRightAreaClicked(lv_event_t* e) {
        TouchTest* test = static_cast<TouchTest*>(lv_event_get_user_data(e));
        test->recordTouch("右侧区域");
    }
    
    static void onScreenTouched(lv_event_t* e) {
        TouchTest* test = static_cast<TouchTest*>(lv_event_get_user_data(e));
        lv_indev_t* indev = lv_indev_get_act();
        if (indev) {
            lv_point_t point;
            lv_indev_get_point(indev, &point);
            
            char coord_text[50];
            snprintf(coord_text, sizeof(coord_text), "坐标: (%d, %d)", point.x, point.y);
            lv_label_set_text(test->touch_coord_label_, coord_text);
        }
    }
    
    void recordTouch(const char* area_name) {
        if (!test_running_) return;
        
        auto current_time = std::chrono::steady_clock::now();
        auto response_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - last_touch_time_).count();
        
        touch_count_++;
        total_response_time_ += response_time;
        last_touch_time_ = current_time;
        
        std::cout << "👆 触摸事件: " << area_name << " (响应时间: " << response_time << "ms)" << std::endl;
        
        // 更新响应时间显示
        char time_text[50];
        snprintf(time_text, sizeof(time_text), "响应时间: %ld ms", response_time);
        lv_label_set_text(response_time_label_, time_text);
    }
    #endif

private:
    bool test_running_;
    int touch_count_;
    long total_response_time_;
    std::chrono::steady_clock::time_point last_touch_time_;
    
    #ifdef ENABLE_LVGL
    lv_obj_t* test_container_;
    lv_obj_t* start_stop_btn_;
    lv_obj_t* reset_btn_;
    lv_obj_t* exit_btn_;
    lv_obj_t* left_touch_area_;
    lv_obj_t* right_touch_area_;
    lv_obj_t* status_area_;
    lv_obj_t* touch_counter_label_;
    lv_obj_t* response_time_label_;
    lv_obj_t* test_status_label_;
    lv_obj_t* touch_coord_label_;
    #endif
};

} // namespace ui
} // namespace bamboo_cut

// 独立测试函数
int main() {
    std::cout << "🚀 启动LVGL触摸交互响应测试" << std::endl;
    
    bamboo_cut::ui::TouchTest test;
    
    if (!test.initialize()) {
        std::cerr << "❌ 触摸测试初始化失败" << std::endl;
        return -1;
    }
    
    test.runTest();
    
    std::cout << "✅ 触摸交互测试完成" << std::endl;
    return 0;
}