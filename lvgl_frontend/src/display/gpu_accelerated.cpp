#include "display/gpu_accelerated.h"
#include <stdio.h>
bool gpu_accelerated_init() {
    printf("初始化 gpu_accelerated\n");
    return true;
}
void gpu_accelerated_deinit() {
    printf("清理 gpu_accelerated\n");
}