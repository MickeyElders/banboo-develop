#ifndef GUI_CONTROL_PANEL_H
#define GUI_CONTROL_PANEL_H

#include "lvgl.h"
#include "common/types.h"

class Control_panel {
public:
    Control_panel();
    ~Control_panel();
    bool initialize();
    
    // 获取控制面板容器
    lv_obj_t* get_container() const { return container_; }
    
    // 更新各个区块的数据
    void update_modbus_registers(const char* reg_values[]);
    void update_plc_status(const char* status, const char* address, uint32_t response_ms, uint32_t total_cuts);
    void update_jetson_info(const performance_stats_t& stats);
    void update_ai_model_status(float inference_time, float accuracy, uint32_t total_detections, uint32_t today_detections);
    void update_communication_stats(const char* connection_time, uint32_t packets, float error_rate, const char* throughput);
    
    // 刀片选择
    void set_blade_selection(int blade_id);

private:
    lv_obj_t* container_;           // 主容器
    
    // 各个功能区块
    lv_obj_t* modbus_section_;      // Modbus寄存器状态
    lv_obj_t* plc_section_;         // PLC通信状态
    lv_obj_t* jetson_section_;      // Jetson系统信息
    lv_obj_t* ai_section_;          // AI模型状态
    lv_obj_t* comm_section_;        // 通信统计
    
    // Modbus区块组件
    lv_obj_t* modbus_table_;        // Modbus寄存器表格
    lv_obj_t* register_labels_[8];  // 寄存器值标签
    
    // PLC区块组件
    lv_obj_t* plc_status_label_;    // PLC连接状态
    lv_obj_t* plc_address_label_;   // PLC地址
    lv_obj_t* plc_response_label_;  // 响应时间
    lv_obj_t* total_cuts_label_;    // 总切割数
    lv_obj_t* blade_buttons_[3];    // 刀片选择按钮
    
    // Jetson区块组件
    lv_obj_t* cpu_progress_;        // CPU进度条
    lv_obj_t* gpu_progress_;        // GPU进度条
    lv_obj_t* memory_progress_;     // 内存进度条
    lv_obj_t* cpu_usage_label_;     // CPU使用率标签
    lv_obj_t* gpu_usage_label_;     // GPU使用率标签
    lv_obj_t* memory_usage_label_;  // 内存使用率标签
    lv_obj_t* system_info_labels_[12]; // 系统详细信息标签
    
    // AI模型区块组件
    lv_obj_t* inference_time_label_; // 推理时间
    lv_obj_t* accuracy_label_;      // 检测精度
    lv_obj_t* total_detections_label_; // 总检测数
    lv_obj_t* today_detections_label_; // 今日检测数
    
    // 通信统计区块组件
    lv_obj_t* connection_time_label_; // 连接时长
    lv_obj_t* packets_label_;       // 数据包数量
    lv_obj_t* error_rate_label_;    // 错误率
    lv_obj_t* throughput_label_;    // 吞吐量
    
    // 样式
    lv_style_t style_container_;    // 容器样式
    lv_style_t style_section_;      // 区块样式
    lv_style_t style_modbus_;       // Modbus区块样式
    lv_style_t style_plc_;          // PLC区块样式
    lv_style_t style_jetson_;       // Jetson区块样式
    lv_style_t style_table_;        // 表格样式
    lv_style_t style_progress_;     // 进度条样式
    lv_style_t style_button_;       // 按钮样式
    lv_style_t style_button_active_; // 激活按钮样式
    
    void create_layout();
    void setup_styles();
    void create_modbus_section();
    void create_plc_section();
    void create_jetson_section();
    void create_ai_section();
    void create_comm_section();
    
    // 事件处理
    static void blade_button_event_cb(lv_event_t* e);
    static void plc_command_button_event_cb(lv_event_t* e);
};

#endif