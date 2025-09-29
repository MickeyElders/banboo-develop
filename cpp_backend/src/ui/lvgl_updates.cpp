/**
 * @file lvgl_updates.cpp
 * @brief LVGL界面数据更新逻辑实现
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <sstream>

namespace bamboo_cut {
namespace ui {

void LVGLInterface::updateInterface() {
#ifdef ENABLE_LVGL
    if (!data_bridge_) {
        std::cout << "[LVGLInterface] DataBridge未初始化，跳过数据更新" << std::endl;
        return;
    }
    
    // 添加静态计数器，确保界面完全初始化后才开始更新
    static bool first_update_done = false;
    static int safe_update_counter = 0;
    
    if (!first_update_done) {
        safe_update_counter++;
        if (safe_update_counter < 10) {  // 前10次调用跳过，确保界面稳定
            std::cout << "[LVGLInterface] 界面初始化期间，跳过数据更新 (" << safe_update_counter << "/10)" << std::endl;
            return;
        }
        first_update_done = true;
        std::cout << "[LVGLInterface] 界面初始化完成，开始正常数据更新" << std::endl;
    }
    
    try {
        // 更新系统状态 - 添加异常保护
        updateSystemStats();
        
        // 更新Modbus显示 - 添加异常保护
        updateModbusDisplay();
        
        // 更新工作流程 - 添加异常保护
        updateWorkflowStatus();
        
        // 更新摄像头 - 添加异常保护
        updateCameraView();
        
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] 数据更新异常: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[LVGLInterface] 数据更新未知异常" << std::endl;
    }
#endif
}

void LVGLInterface::updateSystemStats() {
#ifdef ENABLE_LVGL
    // 添加空指针保护 - 检查关键组件是否已初始化
    if (!header_panel_ || !control_panel_) {
        std::cout << "[LVGLInterface] 面板未初始化，跳过系统状态更新" << std::endl;
        return;
    }
    
    // 更新头部响应时间标签 - 添加空指针检查
    if (header_widgets_.response_label) {
        static int counter = 0;
        int response_ms = 12 + (counter++ % 10);
        lv_label_set_text_fmt(header_widgets_.response_label,
            LV_SYMBOL_LOOP " %dms", response_ms);
    }
    
    // 获取真实的Jetson系统状态
    if (jetson_monitor_ && jetson_monitor_->isRunning()) {
        utils::SystemStats stats = jetson_monitor_->getLatestStats();
        
        // 更新CPU信息
        if (control_widgets_.cpu_bar && !stats.cpu_cores.empty()) {
            // 计算平均CPU使用率
            int total_usage = 0;
            int total_freq = 0;
            for (const auto& core : stats.cpu_cores) {
                total_usage += core.usage_percent;
                total_freq += core.frequency_mhz;
            }
            int avg_usage = total_usage / stats.cpu_cores.size();
            int avg_freq = total_freq / stats.cpu_cores.size();
            
            lv_bar_set_value(control_widgets_.cpu_bar, avg_usage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.cpu_label, "CPU: %d%% @%dMHz", avg_usage, avg_freq);
            
            // 使用黄色条块代表已使用的量
            lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_warning_, LV_PART_INDICATOR);
        }
        
        // 更新GPU信息
        if (control_widgets_.gpu_bar) {
            int gpu_usage = stats.gpu.usage_percent;
            lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.gpu_label, "GPU: %d%% @%dMHz",
                                 gpu_usage, stats.gpu.frequency_mhz);
            
            // 使用黄色条块代表已使用的量
            lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
        }
        
        // 更新内存信息
        if (control_widgets_.mem_bar && stats.memory.ram_total_mb > 0) {
            int mem_percentage = (stats.memory.ram_used_mb * 100) / stats.memory.ram_total_mb;
            lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
            lv_label_set_text_fmt(control_widgets_.mem_label, "RAM: %d%% %d/%dMB (LFB:%dx%dMB)",
                                 mem_percentage, stats.memory.ram_used_mb, stats.memory.ram_total_mb,
                                 stats.memory.lfb_blocks, stats.memory.lfb_size_mb);
            
            // 使用黄色条块代表已使用的量
            lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_warning_, LV_PART_INDICATOR);
        }
        
        updateAIModelStats();
        updateCameraStats();
        updateTemperatureStats(stats);
        updatePowerStats(stats);
        updateSystemExtendedStats(stats);
        
    } else {
        // 如果Jetson监控不可用，回退到模拟数据
        updateSimulatedStats();
    }
    
    // 更新系统指标（12项指标的动态数据）
    updateMetricLabels();
#endif
}

void LVGLInterface::updateAIModelStats() {
#ifdef ENABLE_LVGL
    // 从DataBridge获取真实的AI模型统计数据
    core::SystemStats databridge_stats = data_bridge_->getStats();
    
    // 更新AI模型版本
    if (control_widgets_.ai_model_version_label) {
        lv_label_set_text_fmt(control_widgets_.ai_model_version_label, "模型版本: %s",
                              databridge_stats.ai_model.model_version.c_str());
    }
    
    // 更新推理时间
    if (control_widgets_.ai_inference_time_label) {
        lv_label_set_text_fmt(control_widgets_.ai_inference_time_label, "推理时间: %.1fms",
                              databridge_stats.ai_model.inference_time_ms);
        
        // 根据推理时间设置颜色
        if (databridge_stats.ai_model.inference_time_ms > 30.0f) {
            lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_error_, 0);
        } else if (databridge_stats.ai_model.inference_time_ms > 20.0f) {
            lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_warning_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_success_, 0);
        }
    }
    
    // 更新置信阈值
    if (control_widgets_.ai_confidence_threshold_label) {
        lv_label_set_text_fmt(control_widgets_.ai_confidence_threshold_label, "置信阈值: %.2f",
                              databridge_stats.ai_model.confidence_threshold);
    }
    
    // 更新检测精度
    if (control_widgets_.ai_detection_accuracy_label) {
        lv_label_set_text_fmt(control_widgets_.ai_detection_accuracy_label, "检测精度: %.1f%%",
                              databridge_stats.ai_model.detection_accuracy);
        
        // 根据检测精度设置颜色
        if (databridge_stats.ai_model.detection_accuracy > 90.0f) {
            lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_success_, 0);
        } else if (databridge_stats.ai_model.detection_accuracy > 80.0f) {
            lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_warning_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_error_, 0);
        }
    }
    
    // 更新总检测数
    if (control_widgets_.ai_total_detections_label) {
        lv_label_set_text_fmt(control_widgets_.ai_total_detections_label, "总检测数: %d",
                              databridge_stats.ai_model.total_detections);
    }
    
    // 更新今日检测数
    if (control_widgets_.ai_daily_detections_label) {
        lv_label_set_text_fmt(control_widgets_.ai_daily_detections_label, "今日检测: %d",
                              databridge_stats.ai_model.daily_detections);
    }
    
    // 更新当前竹子检测状态
    updateBambooDetectionStats(databridge_stats.ai_model.current_bamboo);
#endif
}

void LVGLInterface::updateBambooDetectionStats(const core::BambooDetection& bamboo_detection) {
#ifdef ENABLE_LVGL
    // 更新竹子直径
    if (control_widgets_.bamboo_diameter_label) {
        if (bamboo_detection.has_bamboo) {
            lv_label_set_text_fmt(control_widgets_.bamboo_diameter_label, "- 直径：%.1fmm",
                                 bamboo_detection.diameter_mm);
            
            // 根据直径设置颜色 (合理范围20-80mm)
            if (bamboo_detection.diameter_mm >= 20.0f && bamboo_detection.diameter_mm <= 80.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_primary_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_warning_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_diameter_label, "- 直径：无检测");
            lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新竹子长度
    if (control_widgets_.bamboo_length_label) {
        if (bamboo_detection.has_bamboo) {
            lv_label_set_text_fmt(control_widgets_.bamboo_length_label, "- 长度：%.0fmm",
                                 bamboo_detection.length_mm);
            
            // 根据长度设置颜色 (合理范围1000-5000mm)
            if (bamboo_detection.length_mm >= 1000.0f && bamboo_detection.length_mm <= 5000.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_primary_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_warning_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_length_label, "- 长度：无检测");
            lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新预切位置
    if (control_widgets_.bamboo_cut_positions_label) {
        if (bamboo_detection.has_bamboo && !bamboo_detection.cut_positions.empty()) {
            // 构建预切位置字符串
            std::string positions_str = "- 预切位置：[";
            for (size_t i = 0; i < bamboo_detection.cut_positions.size(); i++) {
                if (i > 0) positions_str += ", ";
                positions_str += std::to_string(static_cast<int>(bamboo_detection.cut_positions[i])) + "mm";
            }
            positions_str += "]";
            
            lv_label_set_text(control_widgets_.bamboo_cut_positions_label, positions_str.c_str());
            lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, color_success_, 0);
        } else {
            lv_label_set_text(control_widgets_.bamboo_cut_positions_label, "- 预切位置：无数据");
            lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新检测置信度
    if (control_widgets_.bamboo_confidence_label) {
        if (bamboo_detection.has_bamboo) {
            lv_label_set_text_fmt(control_widgets_.bamboo_confidence_label, "- 置信度：%.2f",
                                 bamboo_detection.confidence);
            
            // 根据置信度设置颜色
            if (bamboo_detection.confidence >= 0.9f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_success_, 0);
            } else if (bamboo_detection.confidence >= 0.7f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_error_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_confidence_label, "- 置信度：N/A");
            lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新检测耗时
    if (control_widgets_.bamboo_detection_time_label) {
        if (bamboo_detection.has_bamboo) {
            lv_label_set_text_fmt(control_widgets_.bamboo_detection_time_label, "- 检测耗时：%.1fms",
                                 bamboo_detection.detection_time_ms);
            
            // 根据检测时间设置颜色
            if (bamboo_detection.detection_time_ms <= 20.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_success_, 0);
            } else if (bamboo_detection.detection_time_ms <= 35.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_error_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_detection_time_label, "- 检测耗时：N/A");
            lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, lv_color_hex(0x8A92A1), 0);
        }
    }
#endif
}

void LVGLInterface::updateCameraStats() {
#ifdef ENABLE_LVGL
    // 从DataBridge获取摄像头系统状态
    core::SystemStats databridge_stats = data_bridge_->getStats();
    const auto& camera_system = databridge_stats.ai_model.camera_system;
    
    // 更新摄像头-1状态
    updateSingleCameraStats(1, camera_system.camera1,
                           control_widgets_.camera1_status_label,
                           control_widgets_.camera1_fps_label,
                           control_widgets_.camera1_resolution_label,
                           control_widgets_.camera1_exposure_label,
                           control_widgets_.camera1_lighting_label);
    
    // 更新摄像头-2状态
    updateSingleCameraStats(2, camera_system.camera2,
                           control_widgets_.camera2_status_label,
                           control_widgets_.camera2_fps_label,
                           control_widgets_.camera2_resolution_label,
                           control_widgets_.camera2_exposure_label,
                           control_widgets_.camera2_lighting_label);
#endif
}

void LVGLInterface::updateSingleCameraStats(int camera_id, const core::CameraStatus& camera_info,
                                           lv_obj_t* status_label, lv_obj_t* fps_label,
                                           lv_obj_t* resolution_label, lv_obj_t* exposure_label,
                                           lv_obj_t* lighting_label) {
#ifdef ENABLE_LVGL
    if (status_label) {
        if (camera_info.is_online) {
            lv_label_set_text_fmt(status_label, "摄像头-%d：在线 ✓", camera_id);
            lv_obj_set_style_text_color(status_label, color_success_, 0);
        } else {
            lv_label_set_text_fmt(status_label, "摄像头-%d：未安装 ✗", camera_id);
            lv_obj_set_style_text_color(status_label, color_error_, 0);
        }
    }
    
    if (fps_label) {
        if (camera_info.is_online) {
            lv_label_set_text_fmt(fps_label, "  帧率：%.0f FPS", camera_info.fps);
            lv_obj_set_style_text_color(fps_label, lv_color_hex(0xB0B8C1), 0);
        } else {
            lv_label_set_text(fps_label, "  帧率：N/A");
            lv_obj_set_style_text_color(fps_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    if (resolution_label) {
        if (camera_info.is_online) {
            lv_label_set_text_fmt(resolution_label, "  分辨率：%dx%d",
                                 camera_info.width, camera_info.height);
            lv_obj_set_style_text_color(resolution_label, lv_color_hex(0xB0B8C1), 0);
        } else {
            lv_label_set_text(resolution_label, "  分辨率：N/A");
            lv_obj_set_style_text_color(resolution_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    if (exposure_label) {
        if (camera_info.is_online) {
            lv_label_set_text_fmt(exposure_label, "  曝光：%s",
                                 camera_info.exposure_mode.c_str());
            lv_obj_set_style_text_color(exposure_label, color_primary_, 0);
        } else {
            lv_label_set_text(exposure_label, "  曝光：N/A");
            lv_obj_set_style_text_color(exposure_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    if (lighting_label) {
        if (camera_info.is_online) {
            lv_label_set_text_fmt(lighting_label, "  光照评分：%s",
                                 camera_info.lighting_quality.c_str());
            
            // 根据光照质量设置颜色
            if (camera_info.lighting_quality == "良好") {
                lv_obj_set_style_text_color(lighting_label, color_success_, 0);
            } else if (camera_info.lighting_quality == "一般") {
                lv_obj_set_style_text_color(lighting_label, color_warning_, 0);
            } else if (camera_info.lighting_quality == "差") {
                lv_obj_set_style_text_color(lighting_label, color_error_, 0);
            } else {
                lv_obj_set_style_text_color(lighting_label, color_primary_, 0);
            }
        } else {
            lv_label_set_text(lighting_label, "  光照评分：N/A");
            lv_obj_set_style_text_color(lighting_label, lv_color_hex(0x8A92A1), 0);
        }
    }
#endif
}

void LVGLInterface::updateTemperatureStats(const utils::SystemStats& stats) {
#ifdef ENABLE_LVGL
    // 更新温度信息
    if (control_widgets_.cpu_temp_label) {
        lv_label_set_text_fmt(control_widgets_.cpu_temp_label, "CPU: %.1f°C", stats.temperature.cpu_temp);
        
        // 根据温度设置颜色
        if (stats.temperature.cpu_temp > 80.0f) {
            lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_error_, 0);
        } else if (stats.temperature.cpu_temp > 70.0f) {
            lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_warning_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_success_, 0);
        }
    }
    
    if (control_widgets_.gpu_temp_label) {
        lv_label_set_text_fmt(control_widgets_.gpu_temp_label, "GPU: %.1f°C", stats.temperature.gpu_temp);
        
        // 根据温度设置颜色
        if (stats.temperature.gpu_temp > 75.0f) {
            lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_error_, 0);
        } else if (stats.temperature.gpu_temp > 65.0f) {
            lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_warning_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_success_, 0);
        }
    }
    
    if (control_widgets_.soc_temp_label) {
        lv_label_set_text_fmt(control_widgets_.soc_temp_label, "SOC: %.1f°C", stats.temperature.soc_temp);
    }
    
    if (control_widgets_.thermal_temp_label) {
        lv_label_set_text_fmt(control_widgets_.thermal_temp_label, "Thermal: %.1f°C", stats.temperature.thermal_temp);
    }
#endif
}

void LVGLInterface::updatePowerStats(const utils::SystemStats& stats) {
#ifdef ENABLE_LVGL
    // 更新电源信息
    if (control_widgets_.power_in_label) {
        lv_label_set_text_fmt(control_widgets_.power_in_label, "VDD_IN: %dmA/%dmW",
                             stats.power.vdd_in_current_ma, stats.power.vdd_in_power_mw);
    }
    
    if (control_widgets_.power_cpu_gpu_label) {
        lv_label_set_text_fmt(control_widgets_.power_cpu_gpu_label, "CPU_GPU: %dmA/%dmW",
                             stats.power.vdd_cpu_gpu_cv_current_ma, stats.power.vdd_cpu_gpu_cv_power_mw);
    }
    
    if (control_widgets_.power_soc_label) {
        lv_label_set_text_fmt(control_widgets_.power_soc_label, "SOC: %dmA/%dmW",
                             stats.power.vdd_soc_current_ma, stats.power.vdd_soc_power_mw);
    }
#endif
}

void LVGLInterface::updateSystemExtendedStats(const utils::SystemStats& stats) {
#ifdef ENABLE_LVGL
    // 更新SWAP使用情况
    if (control_widgets_.swap_usage_label && stats.memory.swap_total_mb > 0) {
        lv_label_set_text_fmt(control_widgets_.swap_usage_label, "SWAP: %d/%dMB",
                             stats.memory.swap_used_mb, stats.memory.swap_total_mb);
        
        // 根据SWAP使用率设置颜色
        int swap_percentage = (stats.memory.swap_used_mb * 100) / stats.memory.swap_total_mb;
        if (swap_percentage > 80) {
            lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_error_, 0);
        } else if (swap_percentage > 50) {
            lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_warning_, 0);
        } else {
            lv_obj_set_style_text_color(control_widgets_.swap_usage_label, color_success_, 0);
        }
    }
    
    // 更新EMC频率信息
    if (control_widgets_.emc_freq_label) {
        lv_label_set_text_fmt(control_widgets_.emc_freq_label, "EMC: %d%%@%dMHz",
                             stats.other.emc_freq_percent, stats.other.emc_freq_mhz);
    }
    
    // 更新VIC使用率
    if (control_widgets_.vic_usage_label) {
        lv_label_set_text_fmt(control_widgets_.vic_usage_label, "VIC: %d%%@%dMHz",
                             stats.other.vic_usage_percent, stats.other.vic_freq_mhz);
    }
    
    // 更新风扇转速
    if (control_widgets_.fan_speed_label) {
        if (stats.other.fan_rpm > 0) {
            lv_label_set_text_fmt(control_widgets_.fan_speed_label, "FAN: %dRPM", stats.other.fan_rpm);
            lv_obj_set_style_text_color(control_widgets_.fan_speed_label, color_primary_, 0);
        } else {
            lv_label_set_text(control_widgets_.fan_speed_label, "FAN: N/A");
            lv_obj_set_style_text_color(control_widgets_.fan_speed_label, lv_color_hex(0x8A92A1), 0);
        }
    }
#endif
}

void LVGLInterface::updateSimulatedStats() {
#ifdef ENABLE_LVGL
    // 如果Jetson监控不可用，回退到模拟数据
    if (control_widgets_.cpu_bar) {
        static int cpu_usage = 45;
        cpu_usage = 40 + (rand() % 30);  // 模拟40-70%的CPU使用率
        lv_bar_set_value(control_widgets_.cpu_bar, cpu_usage, LV_ANIM_ON);
        lv_label_set_text_fmt(control_widgets_.cpu_label, "CPU: %d%% @1.9GHz (模拟)", cpu_usage);
    }
    
    if (control_widgets_.gpu_bar) {
        static int gpu_usage = 72;
        gpu_usage = 60 + (rand() % 25);  // 模拟60-85%的GPU使用率
        lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
        lv_label_set_text_fmt(control_widgets_.gpu_label, "GPU: %d%% @624MHz (模拟)", gpu_usage);
    }
    
    if (control_widgets_.mem_bar) {
        static float mem_used = 4.6f;
        static float mem_total = 8.0f;
        mem_used = 3.8f + ((rand() % 200) / 100.0f);  // 模拟3.8-5.8GB内存使用
        int mem_percentage = (int)((mem_used / mem_total) * 100);
        lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
        lv_label_set_text_fmt(control_widgets_.mem_label, "RAM: %d%% %.1f/%.0fGB (模拟)", mem_percentage, mem_used, mem_total);
    }
#endif
}

void LVGLInterface::updateMetricLabels() {
#ifdef ENABLE_LVGL
    // 更新系统指标（12项指标的动态数据）
    if (!status_widgets_.metric_labels.empty()) {
        static const char* dynamic_values[][12] = {
            {"1.9GHz", "1.3GHz", "1600MHz", "62°C", "58°C", "45°C", "3200RPM", "19.2V", "2.1A", "12ms", "450MB/s", "380MB/s"},
            {"2.0GHz", "1.4GHz", "1600MHz", "64°C", "60°C", "47°C", "3400RPM", "19.1V", "2.2A", "11ms", "460MB/s", "390MB/s"},
            {"1.8GHz", "1.2GHz", "1533MHz", "59°C", "55°C", "43°C", "3000RPM", "19.3V", "2.0A", "13ms", "440MB/s", "370MB/s"}
        };
        
        static int update_cycle = 0;
        static int cycle_counter = 0;
        
        if (++cycle_counter >= 30) {  // 每30次更新切换一次数据组
            cycle_counter = 0;
            update_cycle = (update_cycle + 1) % 3;
        }
        
        for (size_t i = 0; i < status_widgets_.metric_labels.size() && i < 12; i++) {
            lv_label_set_text(status_widgets_.metric_labels[i], dynamic_values[update_cycle][i]);
        }
    }
#endif
}

void LVGLInterface::updateWorkflowStatus() {
#ifdef ENABLE_LVGL
    // 添加空指针保护 - 检查头部面板是否已初始化
    if (!header_panel_ || header_widgets_.workflow_buttons.empty()) {
        return;  // 静默跳过，避免日志过多
    }
    
    // 更新工作流程步骤指示器 - 添加空指针检查
    for(size_t i = 0; i < header_widgets_.workflow_buttons.size(); i++) {
        lv_obj_t* step = header_widgets_.workflow_buttons[i];
        if (!step) continue;  // 跳过空指针
        
        bool is_active = (i == (size_t)current_step_ - 1);
        bool is_completed = (i < (size_t)current_step_ - 1);
        
        if(is_active) {
            lv_obj_set_style_bg_color(step, color_primary_, 0);
            lv_obj_set_style_border_color(step, color_primary_, 0);
        } else if(is_completed) {
            lv_obj_set_style_bg_color(step, color_success_, 0);
            lv_obj_set_style_border_color(step, color_success_, 0);
        } else {
            lv_obj_set_style_bg_color(step, lv_color_hex(0x2A3441), 0);
            lv_obj_set_style_border_color(step, lv_color_hex(0x4A5461), 0);
        }
    }
#endif
}

void LVGLInterface::updateCameraView() {
#ifdef ENABLE_LVGL
    // 添加空指针保护 - 检查摄像头面板是否已初始化
    if (!camera_panel_) {
        return;  // 静默跳过
    }
    
    // 这里应该更新canvas内容
    // 从DataBridge获取最新图像并绘制到canvas上
    
    // 示例：更新信息标签 - 添加空指针检查
    if (camera_widgets_.coord_value) {
        static float x = 0.0f;
        x += 0.1f;
        lv_label_set_text_fmt(camera_widgets_.coord_value,
            LV_SYMBOL_GPS " X: %.2f Y: %.2f Z: %.2f", x, x*0.8f, x*0.5f);
    }
#endif
}

} // namespace ui
} // namespace bamboo_cut