/**
 * @file lvgl_updates.cpp
 * @brief LVGL interface data update logic implementation
 */

#include "bamboo_cut/ui/lvgl_interface.h"
#include <iostream>
#include <sstream>

namespace bamboo_cut {
namespace ui {

void LVGLInterface::updateInterface() {
#ifdef ENABLE_LVGL
    if (!data_bridge_) {
        std::cout << "[LVGLInterface] DataBridge not initialized, skipping data update" << std::endl;
        return;
    }
    
    // Add static counter to ensure UI is fully initialized before starting updates
    static bool first_update_done = false;
    static int safe_update_counter = 0;
    
    if (!first_update_done) {
        safe_update_counter++;
        if (safe_update_counter < 10) {  // Skip first 10 calls to ensure UI stability
            std::cout << "[LVGLInterface] UI initialization in progress, skipping data update (" << safe_update_counter << "/10)" << std::endl;
            return;
        }
        first_update_done = true;
        std::cout << "[LVGLInterface] UI initialization completed, starting normal data updates" << std::endl;
    }
    
    try {
        // Update system stats - add exception protection
        updateSystemStats();
        
        // Update Modbus display - add exception protection
        updateModbusDisplay();
        
        // Update workflow - add exception protection
        updateWorkflowStatus();
        
        // Update camera view - add exception protection
        updateCameraView();
        
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] Data update exception: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "[LVGLInterface] Data update unknown exception" << std::endl;
    }
#endif
}

void LVGLInterface::updateSystemStats() {
#ifdef ENABLE_LVGL
    // Add comprehensive null pointer protection
    if (!header_panel_ || !control_panel_) {
        std::cout << "[LVGLInterface] Key panels not initialized, skipping system stats update" << std::endl;
        return;
    }
    
    // Add additional safety check for widget structures
    if (!header_widgets_.response_label) {
        std::cout << "[LVGLInterface] Header widgets not initialized, skipping system stats update" << std::endl;
        return;
    }
    
    // Update header response time label with safe counter initialization
    try {
        static int response_counter = 0;
        response_counter++;
        int response_ms = 12 + (response_counter % 10);
        // Ensure response time is within reasonable range
        response_ms = std::max(1, std::min(999, response_ms));
        std::string response_text = LV_SYMBOL_LOOP " " + std::to_string(response_ms) + "ms";
        lv_label_set_text(header_widgets_.response_label, response_text.c_str());
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] Exception in response time update: " << e.what() << std::endl;
        return;
    }
    
    // 获取真实的Jetson系统状态 - 添加安全检查
    if (jetson_monitor_ && jetson_monitor_->isRunning()) {
        utils::SystemStats stats;
        try {
            stats = jetson_monitor_->getLatestStats();
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception getting Jetson stats: " << e.what() << std::endl;
            updateSimulatedStats();
            return;
        }
        
        // 更新CPU信息 - 添加完善的空指针检查和数据验证
        if (control_widgets_.cpu_bar && control_widgets_.cpu_label && !stats.cpu_cores.empty()) {
            // 计算平均CPU使用率，添加数据验证
            int total_usage = 0;
            int total_freq = 0;
            int valid_cores = 0;
            
            for (const auto& core : stats.cpu_cores) {
                // 验证CPU核心数据的有效性
                if (core.usage_percent >= 0 && core.usage_percent <= 100 &&
                    core.frequency_mhz >= 0 && core.frequency_mhz <= 10000) {
                    total_usage += core.usage_percent;
                    total_freq += core.frequency_mhz;
                    valid_cores++;
                }
            }
            
            if (valid_cores > 0) {
                int avg_usage = total_usage / valid_cores;
                int avg_freq = total_freq / valid_cores;
                
                // 确保数值在合理范围内
                avg_usage = std::max(0, std::min(100, avg_usage));
                avg_freq = std::max(0, std::min(9999, avg_freq));
                
                lv_bar_set_value(control_widgets_.cpu_bar, avg_usage, LV_ANIM_ON);
                std::string cpu_text = "CPU: " + std::to_string(avg_usage) + "% @" + std::to_string(avg_freq) + "MHz";
                lv_label_set_text(control_widgets_.cpu_label, cpu_text.c_str());
                
                // 使用黄色条块代表已使用的量
                lv_obj_set_style_bg_color(control_widgets_.cpu_bar, color_warning_, LV_PART_INDICATOR);
            } else {
                // 没有有效数据时显示默认值
                lv_bar_set_value(control_widgets_.cpu_bar, 0, LV_ANIM_ON);
                lv_label_set_text(control_widgets_.cpu_label, "CPU: N/A");
            }
        } else if (control_widgets_.cpu_label) {
            // CPU栏不可用时的回退显示
            lv_label_set_text(control_widgets_.cpu_label, "CPU: Not Detected");
        }
        
        // 更新GPU信息 - 添加完善的空指针检查和数据验证
        if (control_widgets_.gpu_bar && control_widgets_.gpu_label) {
            int gpu_usage = stats.gpu.usage_percent;
            int gpu_freq = stats.gpu.frequency_mhz;
            
            // 验证GPU数据的有效性
            if (gpu_usage >= 0 && gpu_usage <= 100 && gpu_freq >= 0 && gpu_freq <= 10000) {
                lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
                std::string gpu_text = "GPU: " + std::to_string(gpu_usage) + "% @" + std::to_string(gpu_freq) + "MHz";
                lv_label_set_text(control_widgets_.gpu_label, gpu_text.c_str());
                
                // 使用黄色条块代表已使用的量
                lv_obj_set_style_bg_color(control_widgets_.gpu_bar, color_warning_, LV_PART_INDICATOR);
            } else {
                // 数据无效时显示默认值
                lv_bar_set_value(control_widgets_.gpu_bar, 0, LV_ANIM_ON);
                lv_label_set_text(control_widgets_.gpu_label, "GPU: Invalid Data");
            }
        } else if (control_widgets_.gpu_label) {
            // GPU栏不可用时的回退显示
            lv_label_set_text(control_widgets_.gpu_label, "GPU: Not Detected");
        }
        
        // 更新内存信息 - 添加完善的空指针检查和数据验证
        if (control_widgets_.mem_bar && control_widgets_.mem_label) {
            if (stats.memory.ram_total_mb > 0 && stats.memory.ram_used_mb >= 0 &&
                stats.memory.ram_used_mb <= stats.memory.ram_total_mb &&
                stats.memory.ram_total_mb <= 100000) {  // 合理的内存上限100GB
                
                int mem_percentage = (stats.memory.ram_used_mb * 100) / stats.memory.ram_total_mb;
                mem_percentage = std::max(0, std::min(100, mem_percentage));  // 确保在0-100范围内
                
                lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
                
                // 验证LFB数据的有效性
                int lfb_blocks = std::max(0, stats.memory.lfb_blocks);
                int lfb_size = std::max(0, stats.memory.lfb_size_mb);
                
                std::string mem_text = "RAM: " + std::to_string(mem_percentage) + "% " +
                                     std::to_string(stats.memory.ram_used_mb) + "/" +
                                     std::to_string(stats.memory.ram_total_mb) + "MB (LFB:" +
                                     std::to_string(lfb_blocks) + "x" + std::to_string(lfb_size) + "MB)";
                lv_label_set_text(control_widgets_.mem_label, mem_text.c_str());
                
                // 使用黄色条块代表已使用的量
                lv_obj_set_style_bg_color(control_widgets_.mem_bar, color_warning_, LV_PART_INDICATOR);
            } else {
                // 内存数据无效时显示默认值
                lv_bar_set_value(control_widgets_.mem_bar, 0, LV_ANIM_ON);
                lv_label_set_text(control_widgets_.mem_label, "RAM: Invalid Data");
            }
        } else if (control_widgets_.mem_label) {
            // 内存栏不可用时的回退显示
            lv_label_set_text(control_widgets_.mem_label, "RAM: Not Detected");
        }
        
        // Call update functions with exception protection
        try {
            updateAIModelStats();
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception in updateAIModelStats: " << e.what() << std::endl;
        }
        
        try {
            updateCameraStats();
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception in updateCameraStats: " << e.what() << std::endl;
        }
        
        try {
            updateTemperatureStats(stats);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception in updateTemperatureStats: " << e.what() << std::endl;
        }
        
        try {
            updatePowerStats(stats);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception in updatePowerStats: " << e.what() << std::endl;
        }
        
        try {
            updateSystemExtendedStats(stats);
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception in updateSystemExtendedStats: " << e.what() << std::endl;
        }
        
    } else {
        // 如果Jetson监控不可用，回退到模拟数据
        try {
            updateSimulatedStats();
        } catch (const std::exception& e) {
            std::cerr << "[LVGLInterface] Exception in updateSimulatedStats: " << e.what() << std::endl;
        }
    }
    
    // 更新系统指标（12项指标的动态数据） - 添加异常保护
    try {
        updateMetricLabels();
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] Exception in updateMetricLabels: " << e.what() << std::endl;
    }
#endif
}

void LVGLInterface::updateAIModelStats() {
#ifdef ENABLE_LVGL
    // 添加DataBridge空指针保护
    if (!data_bridge_) {
        std::cout << "[LVGLInterface] DataBridge未初始化，跳过AI模型统计更新" << std::endl;
        return;
    }
    
    // 获取AI模型统计数据，添加异常保护
    core::SystemStats databridge_stats;
    try {
        databridge_stats = data_bridge_->getStats();
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] 获取AI模型统计数据异常: " << e.what() << std::endl;
        return;
    }
    
    // 更新AI模型版本 - 添加默认值处理
    if (control_widgets_.ai_model_version_label) {
        const std::string& model_version = databridge_stats.ai_model.model_version;
        const char* version_text = model_version.empty() ? "Unknown Version" : model_version.c_str();
        std::string version_text_str = "Model Version: " + std::string(version_text);
        lv_label_set_text(control_widgets_.ai_model_version_label, version_text_str.c_str());
    }
    
    // 更新推理时间 - 添加有效性检查和默认值
    if (control_widgets_.ai_inference_time_label) {
        float inference_time = databridge_stats.ai_model.inference_time_ms;
        if (inference_time < 0.0f || inference_time > 1000.0f) {
            inference_time = 0.0f;  // 无效值时使用默认值
            lv_label_set_text(control_widgets_.ai_inference_time_label, "Inference Time: N/A");
            lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, lv_color_hex(0x8A92A1), 0);
        } else {
            std::string inference_text = "Inference Time: " + std::to_string(static_cast<int>(inference_time * 10) / 10.0) + "ms";
            lv_label_set_text(control_widgets_.ai_inference_time_label, inference_text.c_str());
        
            // 根据推理时间设置颜色
            if (inference_time > 30.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_error_, 0);
            } else if (inference_time > 20.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.ai_inference_time_label, color_success_, 0);
            }
        }
    }
    
    // 更新置信阈值 - 添加范围检查和默认值
    if (control_widgets_.ai_confidence_threshold_label) {
        float confidence_threshold = databridge_stats.ai_model.confidence_threshold;
        if (confidence_threshold < 0.0f || confidence_threshold > 1.0f) {
            confidence_threshold = 0.5f;  // 默认值
            lv_label_set_text(control_widgets_.ai_confidence_threshold_label, "Confidence Threshold: Default(0.50)");
        } else {
            std::string threshold_text = "Confidence Threshold: " + std::to_string(static_cast<int>(confidence_threshold * 100) / 100.0);
            lv_label_set_text(control_widgets_.ai_confidence_threshold_label, threshold_text.c_str());
        }
    }
    
    // 更新检测精度 - 添加范围检查和默认值
    if (control_widgets_.ai_detection_accuracy_label) {
        float detection_accuracy = databridge_stats.ai_model.detection_accuracy;
        if (detection_accuracy < 0.0f || detection_accuracy > 100.0f) {
            detection_accuracy = 0.0f;  // 无效值时使用0
            lv_label_set_text(control_widgets_.ai_detection_accuracy_label, "Detection Accuracy: N/A");
            lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, lv_color_hex(0x8A92A1), 0);
        } else {
            std::string accuracy_text = "Detection Accuracy: " + std::to_string(static_cast<int>(detection_accuracy * 10) / 10.0) + "%";
            lv_label_set_text(control_widgets_.ai_detection_accuracy_label, accuracy_text.c_str());
        
            // 根据检测精度设置颜色
            if (detection_accuracy > 90.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_success_, 0);
            } else if (detection_accuracy > 80.0f) {
                lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.ai_detection_accuracy_label, color_error_, 0);
            }
        }
    }
    
    // 更新总检测数 - 添加有效性检查
    if (control_widgets_.ai_total_detections_label) {
        int total_detections = databridge_stats.ai_model.total_detections;
        if (total_detections < 0) {
            total_detections = 0;  // 负值时使用0
        }
        std::string total_text = "Total Detections: " + std::to_string(total_detections);
        lv_label_set_text(control_widgets_.ai_total_detections_label, total_text.c_str());
    }
    
    // 更新今日检测数 - 添加有效性检查
    if (control_widgets_.ai_daily_detections_label) {
        int daily_detections = databridge_stats.ai_model.daily_detections;
        if (daily_detections < 0) {
            daily_detections = 0;  // 负值时使用0
        }
        std::string daily_text = "Daily Detections: " + std::to_string(daily_detections);
        lv_label_set_text(control_widgets_.ai_daily_detections_label, daily_text.c_str());
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
            std::string diameter_text = "- Diameter: " +
                                      std::to_string(static_cast<int>(bamboo_detection.diameter_mm * 10) / 10.0) + "mm";
            lv_label_set_text(control_widgets_.bamboo_diameter_label, diameter_text.c_str());
            
            // 根据直径设置颜色 (合理范围20-80mm)
            if (bamboo_detection.diameter_mm >= 20.0f && bamboo_detection.diameter_mm <= 80.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_primary_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, color_warning_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_diameter_label, "- Diameter: No Detection");
            lv_obj_set_style_text_color(control_widgets_.bamboo_diameter_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新竹子长度
    if (control_widgets_.bamboo_length_label) {
        if (bamboo_detection.has_bamboo) {
            std::string length_text = "- Length: " + std::to_string(static_cast<int>(bamboo_detection.length_mm)) + "mm";
            lv_label_set_text(control_widgets_.bamboo_length_label, length_text.c_str());
            
            // 根据长度设置颜色 (合理范围1000-5000mm)
            if (bamboo_detection.length_mm >= 1000.0f && bamboo_detection.length_mm <= 5000.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_primary_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, color_warning_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_length_label, "- Length: No Detection");
            lv_obj_set_style_text_color(control_widgets_.bamboo_length_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新预切位置
    if (control_widgets_.bamboo_cut_positions_label) {
        if (bamboo_detection.has_bamboo && !bamboo_detection.cut_positions.empty()) {
            // 构建预切位置字符串
            std::string positions_str = "- Cut Positions: [";
            for (size_t i = 0; i < bamboo_detection.cut_positions.size(); i++) {
                if (i > 0) positions_str += ", ";
                positions_str += std::to_string(static_cast<int>(bamboo_detection.cut_positions[i])) + "mm";
            }
            positions_str += "]";
            
            lv_label_set_text(control_widgets_.bamboo_cut_positions_label, positions_str.c_str());
            lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, color_success_, 0);
        } else {
            lv_label_set_text(control_widgets_.bamboo_cut_positions_label, "- Cut Positions: No Data");
            lv_obj_set_style_text_color(control_widgets_.bamboo_cut_positions_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新检测置信度
    if (control_widgets_.bamboo_confidence_label) {
        if (bamboo_detection.has_bamboo) {
            std::string confidence_text = "- Confidence: " +
                                        std::to_string(static_cast<int>(bamboo_detection.confidence * 100) / 100.0);
            lv_label_set_text(control_widgets_.bamboo_confidence_label, confidence_text.c_str());
            
            // 根据置信度设置颜色
            if (bamboo_detection.confidence >= 0.9f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_success_, 0);
            } else if (bamboo_detection.confidence >= 0.7f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, color_error_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_confidence_label, "- Confidence: N/A");
            lv_obj_set_style_text_color(control_widgets_.bamboo_confidence_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新检测耗时
    if (control_widgets_.bamboo_detection_time_label) {
        if (bamboo_detection.has_bamboo) {
            std::string detection_time_text = "- Detection Time: " +
                                             std::to_string(static_cast<int>(bamboo_detection.detection_time_ms * 10) / 10.0) + "ms";
            lv_label_set_text(control_widgets_.bamboo_detection_time_label, detection_time_text.c_str());
            
            // 根据检测时间设置颜色
            if (bamboo_detection.detection_time_ms <= 20.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_success_, 0);
            } else if (bamboo_detection.detection_time_ms <= 35.0f) {
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, color_error_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.bamboo_detection_time_label, "- Detection Time: N/A");
            lv_obj_set_style_text_color(control_widgets_.bamboo_detection_time_label, lv_color_hex(0x8A92A1), 0);
        }
    }
#endif
}

void LVGLInterface::updateCameraStats() {
#ifdef ENABLE_LVGL
    // 添加DataBridge空指针保护
    if (!data_bridge_) {
        std::cout << "[LVGLInterface] DataBridge未初始化，跳过摄像头统计更新" << std::endl;
        return;
    }
    
    // 获取摄像头系统状态，添加异常保护
    core::SystemStats databridge_stats;
    try {
        databridge_stats = data_bridge_->getStats();
    } catch (const std::exception& e) {
        std::cerr << "[LVGLInterface] 获取摄像头统计数据异常: " << e.what() << std::endl;
        return;
    }
    
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
    // 验证摄像头ID的有效性
    if (camera_id < 1 || camera_id > 99) {
        camera_id = 1;  // 默认值
    }
    
    if (status_label) {
        if (camera_info.is_online) {
            std::string status_text = "Camera-" + std::to_string(camera_id) + ": Online ✓";
            lv_label_set_text(status_label, status_text.c_str());
            lv_obj_set_style_text_color(status_label, color_success_, 0);
        } else {
            std::string status_text = "Camera-" + std::to_string(camera_id) + ": Not Installed ✗";
            lv_label_set_text(status_label, status_text.c_str());
            lv_obj_set_style_text_color(status_label, color_error_, 0);
        }
    }
    
    if (fps_label) {
        if (camera_info.is_online) {
            float fps = camera_info.fps;
            // 验证帧率数据的有效性
            if (fps >= 0.0f && fps <= 240.0f) {  // 合理的帧率范围
                std::string fps_text = "  FPS: " + std::to_string(static_cast<int>(fps));
                lv_label_set_text(fps_label, fps_text.c_str());
                lv_obj_set_style_text_color(fps_label, lv_color_hex(0xB0B8C1), 0);
            } else {
                lv_label_set_text(fps_label, "  FPS: Invalid Data");
                lv_obj_set_style_text_color(fps_label, lv_color_hex(0x8A92A1), 0);
            }
        } else {
            lv_label_set_text(fps_label, "  FPS: N/A");
            lv_obj_set_style_text_color(fps_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    if (resolution_label) {
        if (camera_info.is_online) {
            int width = camera_info.width;
            int height = camera_info.height;
            // 验证分辨率数据的有效性
            if (width > 0 && width <= 8192 && height > 0 && height <= 8192) {
                std::string resolution_text = "  Resolution: " + std::to_string(width) + "x" + std::to_string(height);
                lv_label_set_text(resolution_label, resolution_text.c_str());
                lv_obj_set_style_text_color(resolution_label, lv_color_hex(0xB0B8C1), 0);
            } else {
                lv_label_set_text(resolution_label, "  Resolution: Invalid Data");
                lv_obj_set_style_text_color(resolution_label, lv_color_hex(0x8A92A1), 0);
            }
        } else {
            lv_label_set_text(resolution_label, "  Resolution: N/A");
            lv_obj_set_style_text_color(resolution_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    if (exposure_label) {
        if (camera_info.is_online) {
            const std::string& exposure_mode = camera_info.exposure_mode;
            if (!exposure_mode.empty()) {
                std::string exposure_text = "  Exposure: " + exposure_mode;
                lv_label_set_text(exposure_label, exposure_text.c_str());
                lv_obj_set_style_text_color(exposure_label, color_primary_, 0);
            } else {
                lv_label_set_text(exposure_label, "  Exposure: Unknown Mode");
                lv_obj_set_style_text_color(exposure_label, lv_color_hex(0x8A92A1), 0);
            }
        } else {
            lv_label_set_text(exposure_label, "  Exposure: N/A");
            lv_obj_set_style_text_color(exposure_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    if (lighting_label) {
        if (camera_info.is_online) {
            const std::string& lighting_quality = camera_info.lighting_quality;
            if (!lighting_quality.empty()) {
                std::string lighting_text = "  Lighting Score: " + lighting_quality;
                lv_label_set_text(lighting_label, lighting_text.c_str());
                
                // 根据光照质量设置颜色
                if (lighting_quality == "Good" || lighting_quality == "良好") {
                    lv_obj_set_style_text_color(lighting_label, color_success_, 0);
                } else if (lighting_quality == "Fair" || lighting_quality == "一般") {
                    lv_obj_set_style_text_color(lighting_label, color_warning_, 0);
                } else if (lighting_quality == "Poor" || lighting_quality == "差") {
                    lv_obj_set_style_text_color(lighting_label, color_error_, 0);
                } else {
                    lv_obj_set_style_text_color(lighting_label, color_primary_, 0);
                }
            } else {
                lv_label_set_text(lighting_label, "  Lighting Score: Unknown");
                lv_obj_set_style_text_color(lighting_label, lv_color_hex(0x8A92A1), 0);
            }
        } else {
            lv_label_set_text(lighting_label, "  Lighting Score: N/A");
            lv_obj_set_style_text_color(lighting_label, lv_color_hex(0x8A92A1), 0);
        }
    }
#endif
}

void LVGLInterface::updateTemperatureStats(const utils::SystemStats& stats) {
#ifdef ENABLE_LVGL
    // 更新CPU温度信息 - 添加数据验证和默认值处理
    if (control_widgets_.cpu_temp_label) {
        float cpu_temp = stats.temperature.cpu_temp;
        if (cpu_temp >= -50.0f && cpu_temp <= 150.0f) {  // 合理的温度范围
            std::string cpu_temp_text = "CPU: " + std::to_string(static_cast<int>(cpu_temp * 10) / 10.0) + "°C";
            lv_label_set_text(control_widgets_.cpu_temp_label, cpu_temp_text.c_str());
            
            // 根据温度设置颜色
            if (cpu_temp > 80.0f) {
                lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_error_, 0);
            } else if (cpu_temp > 70.0f) {
                lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, color_success_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.cpu_temp_label, "CPU: N/A");
            lv_obj_set_style_text_color(control_widgets_.cpu_temp_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新GPU温度信息 - 添加数据验证和默认值处理
    if (control_widgets_.gpu_temp_label) {
        float gpu_temp = stats.temperature.gpu_temp;
        if (gpu_temp >= -50.0f && gpu_temp <= 150.0f) {  // 合理的温度范围
            std::string gpu_temp_text = "GPU: " + std::to_string(static_cast<int>(gpu_temp * 10) / 10.0) + "°C";
            lv_label_set_text(control_widgets_.gpu_temp_label, gpu_temp_text.c_str());
            
            // 根据温度设置颜色
            if (gpu_temp > 75.0f) {
                lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_error_, 0);
            } else if (gpu_temp > 65.0f) {
                lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_warning_, 0);
            } else {
                lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, color_success_, 0);
            }
        } else {
            lv_label_set_text(control_widgets_.gpu_temp_label, "GPU: N/A");
            lv_obj_set_style_text_color(control_widgets_.gpu_temp_label, lv_color_hex(0x8A92A1), 0);
        }
    }
    
    // 更新SOC温度信息 - 添加数据验证
    if (control_widgets_.soc_temp_label) {
        float soc_temp = stats.temperature.soc_temp;
        if (soc_temp >= -50.0f && soc_temp <= 150.0f) {
            std::string soc_temp_text = "SOC: " + std::to_string(static_cast<int>(soc_temp * 10) / 10.0) + "°C";
            lv_label_set_text(control_widgets_.soc_temp_label, soc_temp_text.c_str());
        } else {
            lv_label_set_text(control_widgets_.soc_temp_label, "SOC: N/A");
        }
    }
    
    // 更新Thermal温度信息 - 添加数据验证
    if (control_widgets_.thermal_temp_label) {
        float thermal_temp = stats.temperature.thermal_temp;
        if (thermal_temp >= -50.0f && thermal_temp <= 150.0f) {
            std::string thermal_temp_text = "Thermal: " + std::to_string(static_cast<int>(thermal_temp * 10) / 10.0) + "°C";
            lv_label_set_text(control_widgets_.thermal_temp_label, thermal_temp_text.c_str());
        } else {
            lv_label_set_text(control_widgets_.thermal_temp_label, "Thermal: N/A");
        }
    }
#endif
}

void LVGLInterface::updatePowerStats(const utils::SystemStats& stats) {
#ifdef ENABLE_LVGL
    // 更新VDD_IN电源信息 - 添加数据验证和默认值处理
    if (control_widgets_.power_in_label) {
        int current_ma = stats.power.vdd_in_current_ma;
        int power_mw = stats.power.vdd_in_power_mw;
        
        // 验证电流和功率数据的有效性
        if (current_ma >= 0 && current_ma <= 50000 && power_mw >= 0 && power_mw <= 500000) {
            std::string power_in_text = "VDD_IN: " + std::to_string(current_ma) + "mA/" + std::to_string(power_mw) + "mW";
            lv_label_set_text(control_widgets_.power_in_label, power_in_text.c_str());
        } else {
            lv_label_set_text(control_widgets_.power_in_label, "VDD_IN: N/A");
        }
    }
    
    // 更新CPU_GPU电源信息 - 添加数据验证和默认值处理
    if (control_widgets_.power_cpu_gpu_label) {
        int current_ma = stats.power.vdd_cpu_gpu_cv_current_ma;
        int power_mw = stats.power.vdd_cpu_gpu_cv_power_mw;
        
        // 验证电流和功率数据的有效性
        if (current_ma >= 0 && current_ma <= 50000 && power_mw >= 0 && power_mw <= 500000) {
            std::string power_cpu_gpu_text = "CPU_GPU: " + std::to_string(current_ma) + "mA/" + std::to_string(power_mw) + "mW";
            lv_label_set_text(control_widgets_.power_cpu_gpu_label, power_cpu_gpu_text.c_str());
        } else {
            lv_label_set_text(control_widgets_.power_cpu_gpu_label, "CPU_GPU: N/A");
        }
    }
    
    // 更新SOC电源信息 - 添加数据验证和默认值处理
    if (control_widgets_.power_soc_label) {
        int current_ma = stats.power.vdd_soc_current_ma;
        int power_mw = stats.power.vdd_soc_power_mw;
        
        // 验证电流和功率数据的有效性
        if (current_ma >= 0 && current_ma <= 50000 && power_mw >= 0 && power_mw <= 500000) {
            std::string power_soc_text = "SOC: " + std::to_string(current_ma) + "mA/" + std::to_string(power_mw) + "mW";
            lv_label_set_text(control_widgets_.power_soc_label, power_soc_text.c_str());
        } else {
            lv_label_set_text(control_widgets_.power_soc_label, "SOC: N/A");
        }
    }
#endif
}

void LVGLInterface::updateSystemExtendedStats(const utils::SystemStats& stats) {
#ifdef ENABLE_LVGL
    // 更新SWAP使用情况
    if (control_widgets_.swap_usage_label && stats.memory.swap_total_mb > 0) {
        std::string swap_text = "SWAP: " + std::to_string(stats.memory.swap_used_mb) + "/" + std::to_string(stats.memory.swap_total_mb) + "MB";
        lv_label_set_text(control_widgets_.swap_usage_label, swap_text.c_str());
        
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
        std::string emc_text = "EMC: " + std::to_string(stats.other.emc_freq_percent) + "%@" + std::to_string(stats.other.emc_freq_mhz) + "MHz";
        lv_label_set_text(control_widgets_.emc_freq_label, emc_text.c_str());
    }
    
    // 更新VIC使用率
    if (control_widgets_.vic_usage_label) {
        std::string vic_text = "VIC: " + std::to_string(stats.other.vic_usage_percent) + "%@" + std::to_string(stats.other.vic_freq_mhz) + "MHz";
        lv_label_set_text(control_widgets_.vic_usage_label, vic_text.c_str());
    }
    
    // 更新风扇转速
    if (control_widgets_.fan_speed_label) {
        if (stats.other.fan_rpm > 0) {
            std::string fan_text = "FAN: " + std::to_string(stats.other.fan_rpm) + "RPM";
            lv_label_set_text(control_widgets_.fan_speed_label, fan_text.c_str());
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
        std::string cpu_sim_text = "CPU: " + std::to_string(cpu_usage) + "% @1.9GHz (Simulated)";
        lv_label_set_text(control_widgets_.cpu_label, cpu_sim_text.c_str());
    }
    
    if (control_widgets_.gpu_bar) {
        static int gpu_usage = 72;
        gpu_usage = 60 + (rand() % 25);  // 模拟60-85%的GPU使用率
        lv_bar_set_value(control_widgets_.gpu_bar, gpu_usage, LV_ANIM_ON);
        std::string gpu_sim_text = "GPU: " + std::to_string(gpu_usage) + "% @624MHz (Simulated)";
        lv_label_set_text(control_widgets_.gpu_label, gpu_sim_text.c_str());
    }
    
    if (control_widgets_.mem_bar) {
        static float mem_used = 4.6f;
        static float mem_total = 8.0f;
        mem_used = 3.8f + ((rand() % 200) / 100.0f);  // 模拟3.8-5.8GB内存使用
        int mem_percentage = (int)((mem_used / mem_total) * 100);
        lv_bar_set_value(control_widgets_.mem_bar, mem_percentage, LV_ANIM_ON);
        std::string mem_sim_text = "RAM: " + std::to_string(mem_percentage) + "% " +
                                 std::to_string(static_cast<int>(mem_used * 10) / 10.0) + "/" +
                                 std::to_string(static_cast<int>(mem_total)) + "GB (Simulated)";
        lv_label_set_text(control_widgets_.mem_label, mem_sim_text.c_str());
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
        std::string coord_text = LV_SYMBOL_GPS " X: " +
                               std::to_string(static_cast<int>(x * 100) / 100.0) + " Y: " +
                               std::to_string(static_cast<int>((x*0.8f) * 100) / 100.0) + " Z: " +
                               std::to_string(static_cast<int>((x*0.5f) * 100) / 100.0);
        lv_label_set_text(camera_widgets_.coord_value, coord_text.c_str());
    }
#endif
}

} // namespace ui
} // namespace bamboo_cut