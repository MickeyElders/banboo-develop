#include "input/touch_driver.h"
#include <stdio.h>
bool touch_driver_init() {
    printf("初始化 touch_driver\n");
    return true;
}
void touch_driver_deinit() {
    printf("清理 touch_driver\n");
}