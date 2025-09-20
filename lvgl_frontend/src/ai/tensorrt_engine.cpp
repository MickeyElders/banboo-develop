#include "ai/tensorrt_engine.h"
#include <stdio.h>
Tensorrt_engine::Tensorrt_engine() {}
Tensorrt_engine::~Tensorrt_engine() {}
bool Tensorrt_engine::initialize() {
    printf("初始化 tensorrt_engine\n");
    return true;
}