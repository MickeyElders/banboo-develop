#include "display/framebuffer_driver.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/fb.h>
#include <string.h>
#include <errno.h>

// Framebuffer相关变量
static int fb_fd = -1;
static struct fb_var_screeninfo vinfo;
static struct fb_fix_screeninfo finfo;
static char *fb_buffer = nullptr;
static long screensize = 0;

// LVGL缓冲区
static uint32_t *lvgl_buffer = nullptr;
static uint32_t *lvgl_buffer2 = nullptr;

bool framebuffer_driver_init() {
    printf("初始化 framebuffer_driver\n");
    
    // 打开framebuffer设备
    fb_fd = open("/dev/fb0", O_RDWR);
    if (fb_fd == -1) {
        printf("错误: 无法打开 /dev/fb0: %s\n", strerror(errno));
        return false;
    }
    
    // 获取固定屏幕信息
    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &finfo) == -1) {
        printf("错误: 无法获取固定屏幕信息: %s\n", strerror(errno));
        close(fb_fd);
        return false;
    }
    
    // 获取可变屏幕信息
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &vinfo) == -1) {
        printf("错误: 无法获取可变屏幕信息: %s\n", strerror(errno));
        close(fb_fd);
        return false;
    }
    
    // 打印framebuffer信息
    printf("Framebuffer信息:\n");
    printf("  分辨率: %dx%d\n", vinfo.xres, vinfo.yres);
    printf("  虚拟分辨率: %dx%d\n", vinfo.xres_virtual, vinfo.yres_virtual);
    printf("  色深: %d位\n", vinfo.bits_per_pixel);
    printf("  红色: 偏移%d, 长度%d\n", vinfo.red.offset, vinfo.red.length);
    printf("  绿色: 偏移%d, 长度%d\n", vinfo.green.offset, vinfo.green.length);
    printf("  蓝色: 偏移%d, 长度%d\n", vinfo.blue.offset, vinfo.blue.length);
    printf("  行长度: %d字节\n", finfo.line_length);
    
    // 计算屏幕大小
    screensize = vinfo.yres_virtual * finfo.line_length;
    printf("  屏幕缓冲区大小: %ld字节\n", screensize);
    
    // 映射framebuffer到内存
    fb_buffer = (char*)mmap(0, screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fb_fd, 0);
    if (fb_buffer == MAP_FAILED) {
        printf("错误: 无法映射framebuffer到内存: %s\n", strerror(errno));
        close(fb_fd);
        return false;
    }
    
    // 清空framebuffer
    memset(fb_buffer, 0, screensize);
    
    // 分配LVGL缓冲区
    size_t buffer_size = vinfo.xres * vinfo.yres * sizeof(uint32_t);
    lvgl_buffer = (uint32_t*)malloc(buffer_size);
    lvgl_buffer2 = (uint32_t*)malloc(buffer_size);
    
    if (!lvgl_buffer || !lvgl_buffer2) {
        printf("错误: 无法分配LVGL缓冲区\n");
        framebuffer_driver_deinit();
        return false;
    }
    
    // 清空LVGL缓冲区
    memset(lvgl_buffer, 0, buffer_size);
    memset(lvgl_buffer2, 0, buffer_size);
    
    printf("Framebuffer驱动初始化成功\n");
    return true;
}

void framebuffer_driver_deinit() {
    printf("清理 framebuffer_driver\n");
    
    if (lvgl_buffer) {
        free(lvgl_buffer);
        lvgl_buffer = nullptr;
    }
    
    if (lvgl_buffer2) {
        free(lvgl_buffer2);
        lvgl_buffer2 = nullptr;
    }
    
    if (fb_buffer && fb_buffer != MAP_FAILED) {
        munmap(fb_buffer, screensize);
        fb_buffer = nullptr;
    }
    
    if (fb_fd >= 0) {
        close(fb_fd);
        fb_fd = -1;
    }
}

// 获取framebuffer信息
struct fb_var_screeninfo* get_fb_vinfo() {
    return &vinfo;
}

struct fb_fix_screeninfo* get_fb_finfo() {
    return &finfo;
}

char* get_fb_buffer() {
    return fb_buffer;
}

uint32_t* get_lvgl_buffer() {
    return lvgl_buffer;
}

uint32_t* get_lvgl_buffer2() {
    return lvgl_buffer2;
}

// 刷新显示 - 从LVGL缓冲区复制到framebuffer
void framebuffer_flush(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, const uint32_t* color_map) {
    if (!fb_buffer || !color_map) {
        return;
    }
    
    // 限制坐标范围
    if (x2 >= vinfo.xres) x2 = vinfo.xres - 1;
    if (y2 >= vinfo.yres) y2 = vinfo.yres - 1;
    
    // 根据色深进行像素格式转换
    for (uint32_t y = y1; y <= y2; y++) {
        for (uint32_t x = x1; x <= x2; x++) {
            uint32_t color = color_map[(y * vinfo.xres) + x];
            
            // 提取RGB分量
            uint8_t r = (color >> 16) & 0xFF;
            uint8_t g = (color >> 8) & 0xFF;
            uint8_t b = color & 0xFF;
            
            // 计算framebuffer位置
            long fb_offset = (y * finfo.line_length) + (x * (vinfo.bits_per_pixel / 8));
            
            if (vinfo.bits_per_pixel == 32) {
                // 32位ARGB
                uint32_t *fb_pixel = (uint32_t*)(fb_buffer + fb_offset);
                *fb_pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;
            } else if (vinfo.bits_per_pixel == 24) {
                // 24位RGB
                fb_buffer[fb_offset + 0] = b;
                fb_buffer[fb_offset + 1] = g;
                fb_buffer[fb_offset + 2] = r;
            } else if (vinfo.bits_per_pixel == 16) {
                // 16位RGB565
                uint16_t *fb_pixel = (uint16_t*)(fb_buffer + fb_offset);
                *fb_pixel = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
            }
        }
    }
}