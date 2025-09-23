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
    
    // 检查framebuffer设备是否存在（非阻塞）
    if (access("/dev/fb0", F_OK) != 0) {
        printf("警告: framebuffer设备 /dev/fb0 不存在，尝试替代方案\n");
        return init_virtual_framebuffer();
    }
    
    // 非阻塞方式打开framebuffer设备
    fb_fd = open("/dev/fb0", O_RDWR | O_NONBLOCK);
    if (fb_fd == -1) {
        printf("警告: 无法打开 /dev/fb0: %s，使用虚拟framebuffer\n", strerror(errno));
        return init_virtual_framebuffer();
    }
    
    // 切换回阻塞模式进行ioctl操作
    int flags = fcntl(fb_fd, F_GETFL, 0);
    fcntl(fb_fd, F_SETFL, flags & ~O_NONBLOCK);
    
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

// 虚拟framebuffer初始化（用于无显示设备的环境）
bool init_virtual_framebuffer() {
    printf("初始化虚拟framebuffer（无显示设备模式）\n");
    
    // 设置默认的虚拟屏幕参数
    memset(&vinfo, 0, sizeof(vinfo));
    memset(&finfo, 0, sizeof(finfo));
    
    vinfo.xres = 1024;          // 默认宽度
    vinfo.yres = 768;           // 默认高度
    vinfo.xres_virtual = 1024;
    vinfo.yres_virtual = 768;
    vinfo.bits_per_pixel = 32;
    vinfo.red.offset = 16;
    vinfo.red.length = 8;
    vinfo.green.offset = 8;
    vinfo.green.length = 8;
    vinfo.blue.offset = 0;
    vinfo.blue.length = 8;
    
    finfo.line_length = vinfo.xres * (vinfo.bits_per_pixel / 8);
    screensize = vinfo.yres * finfo.line_length;
    
    // 分配虚拟framebuffer内存
    fb_buffer = (char*)malloc(screensize);
    if (!fb_buffer) {
        printf("错误: 无法分配虚拟framebuffer内存\n");
        return false;
    }
    memset(fb_buffer, 0, screensize);
    
    // 分配LVGL缓冲区
    size_t buffer_size = vinfo.xres * vinfo.yres * sizeof(uint32_t);
    lvgl_buffer = (uint32_t*)malloc(buffer_size);
    lvgl_buffer2 = (uint32_t*)malloc(buffer_size);
    
    if (!lvgl_buffer || !lvgl_buffer2) {
        printf("错误: 无法分配LVGL缓冲区\n");
        if (fb_buffer) {
            free(fb_buffer);
            fb_buffer = nullptr;
        }
        return false;
    }
    
    memset(lvgl_buffer, 0, buffer_size);
    memset(lvgl_buffer2, 0, buffer_size);
    
    fb_fd = -1; // 标记为虚拟模式
    
    printf("虚拟framebuffer初始化成功 (%dx%d@%d位)\n",
           vinfo.xres, vinfo.yres, vinfo.bits_per_pixel);
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
    
    if (fb_buffer) {
        if (fb_fd >= 0 && fb_buffer != MAP_FAILED) {
            // 真实framebuffer，使用munmap
            munmap(fb_buffer, screensize);
        } else {
            // 虚拟framebuffer，使用free
            free(fb_buffer);
        }
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

// 刷新显示 - 从LVGL缓冲区复制到framebuffer（安全版本）
void framebuffer_flush(uint32_t x1, uint32_t y1, uint32_t x2, uint32_t y2, const uint32_t* color_map) {
    // 安全检查
    if (!fb_buffer || !color_map || fb_fd < 0) {
        return;
    }
    
    // 严格的边界检查
    if (x1 >= vinfo.xres || y1 >= vinfo.yres) {
        return;
    }
    
    if (x2 >= vinfo.xres) x2 = vinfo.xres - 1;
    if (y2 >= vinfo.yres) y2 = vinfo.yres - 1;
    if (x1 > x2 || y1 > y2) {
        return;
    }
    
    // 计算总像素数以防止越界
    uint32_t total_pixels = vinfo.xres * vinfo.yres;
    
    // 根据色深进行像素格式转换
    for (uint32_t y = y1; y <= y2; y++) {
        for (uint32_t x = x1; x <= x2; x++) {
            // 安全的索引计算
            uint32_t color_index = (y * vinfo.xres) + x;
            
            // 越界检查
            if (color_index >= total_pixels) {
                continue;
            }
            
            uint32_t color = color_map[color_index];
            
            // 提取RGB分量
            uint8_t r = (color >> 16) & 0xFF;
            uint8_t g = (color >> 8) & 0xFF;
            uint8_t b = color & 0xFF;
            
            // 计算framebuffer位置
            long fb_offset = (y * finfo.line_length) + (x * (vinfo.bits_per_pixel / 8));
            
            // 检查framebuffer边界
            if (fb_offset >= screensize - (vinfo.bits_per_pixel / 8)) {
                continue;
            }
            
            if (vinfo.bits_per_pixel == 32) {
                // 32位ARGB
                uint32_t *fb_pixel = (uint32_t*)(fb_buffer + fb_offset);
                *fb_pixel = (0xFF << 24) | (r << 16) | (g << 8) | b;
            } else if (vinfo.bits_per_pixel == 24) {
                // 24位RGB - 检查3字节边界
                if (fb_offset + 2 < screensize) {
                    fb_buffer[fb_offset + 0] = b;
                    fb_buffer[fb_offset + 1] = g;
                    fb_buffer[fb_offset + 2] = r;
                }
            } else if (vinfo.bits_per_pixel == 16) {
                // 16位RGB565
                uint16_t *fb_pixel = (uint16_t*)(fb_buffer + fb_offset);
                *fb_pixel = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
            }
        }
    }
}