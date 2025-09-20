/**
 * 事件管理器
 */

#ifndef APP_EVENT_MANAGER_H
#define APP_EVENT_MANAGER_H

#include <queue>
#include <mutex>
#include <functional>
#include <map>
#include "common/types.h"

class EventManager {
public:
    EventManager();
    ~EventManager();

    bool initialize();
    void process_events();
    void post_event(const event_data_t& event);
    void register_handler(event_type_t type, std::function<void(const event_data_t&)> handler);

private:
    std::queue<event_data_t> event_queue_;
    std::mutex queue_mutex_;
    std::map<event_type_t, std::vector<std::function<void(const event_data_t&)>>> handlers_;
    bool initialized_;
};

#endif // APP_EVENT_MANAGER_H