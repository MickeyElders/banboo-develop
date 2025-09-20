/**
 * 事件管理器实现
 */

#include "app/event_manager.h"
#include <stdio.h>

EventManager::EventManager() : initialized_(false) {
}

EventManager::~EventManager() {
}

bool EventManager::initialize() {
    printf("初始化事件管理器...\n");
    initialized_ = true;
    printf("事件管理器初始化成功\n");
    return true;
}

void EventManager::process_events() {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    while (!event_queue_.empty()) {
        event_data_t event = event_queue_.front();
        event_queue_.pop();
        
        // 处理事件
        auto it = handlers_.find(event.type);
        if (it != handlers_.end()) {
            for (auto& handler : it->second) {
                handler(event);
            }
        }
    }
}

void EventManager::post_event(const event_data_t& event) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(queue_mutex_);
    event_queue_.push(event);
}

void EventManager::register_handler(event_type_t type, std::function<void(const event_data_t&)> handler) {
    handlers_[type].push_back(handler);
}