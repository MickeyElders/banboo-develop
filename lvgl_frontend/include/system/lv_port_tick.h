/**
 * @file lv_port_tick.h
 * @brief LVGL时钟接口头文件 - Linux系统
 */

#ifndef LV_PORT_TICK_H
#define LV_PORT_TICK_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/**
 * @brief 获取系统时钟毫秒值
 * @return 系统启动以来的毫秒数
 */
uint32_t lv_tick_get_ms(void);

/**
 * @brief LVGL时钟初始化
 */
void lv_port_tick_init(void);

/**
 * @brief 精确延时函数 (microseconds)
 * @param us 延时微秒数
 */
void lv_delay_us(uint32_t us);

/**
 * @brief 精确延时函数 (milliseconds)
 * @param ms 延时毫秒数
 */
void lv_delay_ms(uint32_t ms);

#ifdef __cplusplus
}
#endif

#endif /* LV_PORT_TICK_H */