/**
 * @file config_loader.cpp
 * @brief C++ LVGL一体化系统配置加载器实现
 * @version 5.0.0
 * @date 2024
 * 
 * 支持YAML配置文件的加载和解析
 */

#include "bamboo_cut/utils/config_loader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace bamboo_cut {
namespace utils {

// ConfigValue实现
std::string ConfigValue::asString(const std::string& default_val) const {
    if (!str_value_.empty()) {
        return str_value_;
    } else if (has_int_) {
        return std::to_string(int_value_);
    } else if (has_float_) {
        return std::to_string(float_value_);
    } else if (has_bool_) {
        return bool_value_ ? "true" : "false";
    }
    return default_val;
}

int ConfigValue::asInt(int default_val) const {
    if (has_int_) {
        return int_value_;
    } else if (!str_value_.empty()) {
        try {
            return std::stoi(str_value_);
        } catch (const std::exception&) {
            return default_val;
        }
    } else if (has_float_) {
        return static_cast<int>(float_value_);
    } else if (has_bool_) {
        return bool_value_ ? 1 : 0;
    }
    return default_val;
}

float ConfigValue::asFloat(float default_val) const {
    if (has_float_) {
        return float_value_;
    } else if (!str_value_.empty()) {
        try {
            return std::stof(str_value_);
        } catch (const std::exception&) {
            return default_val;
        }
    } else if (has_int_) {
        return static_cast<float>(int_value_);
    } else if (has_bool_) {
        return bool_value_ ? 1.0f : 0.0f;
    }
    return default_val;
}

bool ConfigValue::asBool(bool default_val) const {
    if (has_bool_) {
        return bool_value_;
    } else if (!str_value_.empty()) {
        std::string lower_str = str_value_;
        std::transform(lower_str.begin(), lower_str.end(), lower_str.begin(), ::tolower);
        if (lower_str == "true" || lower_str == "1" || lower_str == "yes" || lower_str == "on") {
            return true;
        } else if (lower_str == "false" || lower_str == "0" || lower_str == "no" || lower_str == "off") {
            return false;
        }
    } else if (has_int_) {
        return int_value_ != 0;
    } else if (has_float_) {
        return float_value_ != 0.0f;
    }
    return default_val;
}

std::vector<std::string> ConfigValue::asStringList() const {
    std::vector<std::string> result;
    if (!str_value_.empty()) {
        // 简单的逗号分隔解析
        std::stringstream ss(str_value_);
        std::string item;
        while (std::getline(ss, item, ',')) {
            // 去除前后空白
            item.erase(0, item.find_first_not_of(" \t"));
            item.erase(item.find_last_not_of(" \t") + 1);
            if (!item.empty()) {
                result.push_back(item);
            }
        }
    }
    return result;
}

// ConfigNode实现
std::shared_ptr<ConfigNode> ConfigNode::getChild(const std::string& key) const {
    auto it = children_.find(key);
    if (it != children_.end()) {
        return it->second;
    }
    return nullptr;
}

ConfigValue ConfigNode::getValue(const std::string& key) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        return it->second;
    }
    return ConfigValue();  // 返回无效值
}

void ConfigNode::setValue(const std::string& key, const ConfigValue& value) {
    values_[key] = value;
}

void ConfigNode::setChild(const std::string& key, std::shared_ptr<ConfigNode> child) {
    children_[key] = child;
}

bool ConfigNode::hasKey(const std::string& key) const {
    return values_.find(key) != values_.end() || children_.find(key) != children_.end();
}

std::vector<std::string> ConfigNode::getKeys() const {
    std::vector<std::string> keys;
    for (const auto& pair : values_) {
        keys.push_back(pair.first);
    }
    for (const auto& pair : children_) {
        keys.push_back(pair.first);
    }
    return keys;
}

// ConfigLoader实现
bool ConfigLoader::loadFromFile(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        last_error_ = "无法打开配置文件: " + file_path;
        return false;
    }
    
    std::string content((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
    file.close();
    
    return loadFromString(content);
}

bool ConfigLoader::loadFromString(const std::string& config_data) {
    try {
        root_ = parseYamlContent(config_data);
        return root_ != nullptr;
    } catch (const std::exception& e) {
        last_error_ = "解析配置数据失败: " + std::string(e.what());
        return false;
    }
}

bool ConfigLoader::saveToFile(const std::string& file_path) const {
    if (!root_) {
        last_error_ = "没有配置数据可保存";
        return false;
    }
    
    std::ofstream file(file_path);
    if (!file.is_open()) {
        last_error_ = "无法创建配置文件: " + file_path;
        return false;
    }
    
    // 简化的YAML输出实现
    file << "# 竹子识别系统配置文件\n";
    file << "# 自动生成\n\n";
    
    // TODO: 实现完整的YAML序列化
    
    return true;
}

ConfigValue ConfigLoader::getValue(const std::string& path, const ConfigValue& default_value) const {
    if (!root_) {
        return default_value;
    }
    
    auto node = findNode(path);
    if (!node) {
        return default_value;
    }
    
    // 获取路径的最后一部分作为键
    auto path_parts = splitPath(path);
    if (path_parts.empty()) {
        return default_value;
    }
    
    std::string key = path_parts.back();
    return node->getValue(key);
}

void ConfigLoader::setValue(const std::string& path, const ConfigValue& value) {
    if (!root_) {
        root_ = std::make_shared<ConfigNode>();
    }
    
    auto path_parts = splitPath(path);
    if (path_parts.empty()) {
        return;
    }
    
    std::string key = path_parts.back();
    path_parts.pop_back();
    
    auto node = root_;
    for (const auto& part : path_parts) {
        auto child = node->getChild(part);
        if (!child) {
            child = std::make_shared<ConfigNode>();
            node->setChild(part, child);
        }
        node = child;
    }
    
    node->setValue(key, value);
}

std::shared_ptr<ConfigNode> ConfigLoader::parseYamlContent(const std::string& content) {
    auto root = std::make_shared<ConfigNode>();
    
    std::stringstream ss(content);
    std::string line;
    std::shared_ptr<ConfigNode> current_node = root;
    
    while (std::getline(ss, line)) {
        // 去除注释
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        // 去除前后空白
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) {
            continue;
        }
        
        // 简化的YAML解析（仅支持key: value格式）
        auto colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // 去除key和value的空白
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (!key.empty()) {
                // 尝试转换为不同类型
                ConfigValue config_value;
                
                if (value == "true") {
                    config_value = ConfigValue(true);
                } else if (value == "false") {
                    config_value = ConfigValue(false);
                } else if (value.find('.') != std::string::npos) {
                    // 可能是浮点数
                    try {
                        float f_val = std::stof(value);
                        config_value = ConfigValue(f_val);
                    } catch (const std::exception&) {
                        config_value = ConfigValue(value);
                    }
                } else {
                    // 可能是整数
                    try {
                        int i_val = std::stoi(value);
                        config_value = ConfigValue(i_val);
                    } catch (const std::exception&) {
                        config_value = ConfigValue(value);
                    }
                }
                
                current_node->setValue(key, config_value);
            }
        }
    }
    
    return root;
}

std::shared_ptr<ConfigNode> ConfigLoader::findNode(const std::string& path) const {
    if (!root_) {
        return nullptr;
    }
    
    auto path_parts = splitPath(path);
    if (path_parts.empty()) {
        return root_;
    }
    
    auto node = root_;
    for (size_t i = 0; i < path_parts.size() - 1; ++i) {
        node = node->getChild(path_parts[i]);
        if (!node) {
            return nullptr;
        }
    }
    
    return node;
}

std::vector<std::string> ConfigLoader::splitPath(const std::string& path) const {
    std::vector<std::string> parts;
    std::stringstream ss(path);
    std::string item;
    
    while (std::getline(ss, item, '.')) {
        if (!item.empty()) {
            parts.push_back(item);
        }
    }
    
    return parts;
}

} // namespace utils
} // namespace bamboo_cut