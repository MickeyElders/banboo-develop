#include "camera/camera_manager.h"
#include <stdio.h>
Camera_manager::Camera_manager() {}
Camera_manager::~Camera_manager() {}
bool Camera_manager::initialize() {
    printf("初始化 camera_manager\n");
    return true;
}