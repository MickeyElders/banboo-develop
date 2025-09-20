/**
 * 配置管理器
 */

#ifndef APP_CONFIG_MANAGER_H
#define APP_CONFIG_MANAGER_H

#include "common/types.h"

class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager();

    bool load(const char* config_file = nullptr);
    void set_debug_mode(bool enable);
    
    const system_config_t& get_config() const { return config_; }

private:
    system_config_t config_;
    bool load_from_file(const char* filename);
    void set_defaults();
};

#endif // APP_CONFIG_MANAGER_H