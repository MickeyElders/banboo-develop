#include "display/lvgl_display.h"
#include <stdio.h>

bool lvgl_display_init() {
    printf("初始化 lvgl_display\n");
    return true;
}

void lvgl_display_deinit() {
    printf("清理 lvgl_display\n");
}