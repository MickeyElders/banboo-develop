#include "camera/cuda_processor.h"
#include <stdio.h>
Cuda_processor::Cuda_processor() {}
Cuda_processor::~Cuda_processor() {}
bool Cuda_processor::initialize() {
    printf("初始化 cuda_processor\n");
    return true;
}