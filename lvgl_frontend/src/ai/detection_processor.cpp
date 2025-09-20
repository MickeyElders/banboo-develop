#include "ai/detection_processor.h"
#include <stdio.h>
Detection_processor::Detection_processor() {}
Detection_processor::~Detection_processor() {}
bool Detection_processor::initialize() {
    printf("初始化 detection_processor\n");
    return true;
}