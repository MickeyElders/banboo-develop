/**
 * @file lv_port_tick.c
 * @brief LVGL时钟接口实现 - Linux系统
 */

#include "lvgl.h"
#include <sys/time.h>
#include <time.h>
#include <stdint.h>

/**
 * @brief 获取系统时钟毫秒值
 * @return 系统启动以来的毫秒数
 */
uint32_t lv_tick_get_ms(void)
{
    struct timespec ts;
    
    // 使用CLOCK_MONOTONIC获取单调时钟
    // 这个时钟不受系统时间调整影响，适合做相对时间计算
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
    }
    
    // 备用方案：使用gettimeofday
    struct timeval tv;
    if (gettimeofday(&tv, NULL) == 0) {
        return (uint32_t)(tv.tv_sec * 1000 + tv.tv_usec / 1000);
    }
    
    // 最后备用方案：使用time()
    return (uint32_t)(time(NULL) * 1000);
}

/**
 * @brief LVGL时钟初始化
 * 
 * 注意：LVGL需要定期调用lv_tick_inc()来更新内部时钟
 * 通常在主循环中每1ms调用一次，或使用定时器
 */
void lv_port_tick_init(void)
{
    // Linux下通常使用系统时钟，不需要特殊初始化
    // 如果需要高精度定时器，可以在这里设置
}

/**
 * @brief 精确延时函数 (microseconds)
 * @param us 延时微秒数
 */
void lv_delay_us(uint32_t us)
{
    struct timespec ts;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (us % 1000000) * 1000;
    nanosleep(&ts, NULL);
}

/**
 * @brief 精确延时函数 (milliseconds)
 * @param ms 延时毫秒数
 */
void lv_delay_ms(uint32_t ms)
{
    lv_delay_us(ms * 1000);
}