#include "display/framebuffer_driver.h"
#include <stdio.h>

bool framebuffer_driver_init() {
    printf("初始化 framebuffer_driver\n");
    return true;
}

void framebuffer_driver_deinit() {
    printf("清理 framebuffer_driver\n");
}