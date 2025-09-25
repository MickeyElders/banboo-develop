/**
 * @file config_loader.h  
 * @brief C++ LVGL一体化系统配置加载器
 * 支持YAML配置文件的加载和解析
 */

#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>

namespace bamboo_cut {
namespace utils {

/**
 * @brief 配置值类型
 */
class ConfigValue {
public:
    ConfigValue() = default;
    ConfigValue(const std::string& value) : str_value_(value) {}
    ConfigValue(int value) : int_value_(value), has_int_(true) {}
    ConfigValue(float value) : float_value_(value), has_float_(true) {}
    ConfigValue(bool value) : bool_value_(value), has_bool_(true) {}
    
    // 类型转换方法
    std::string asString(const std::string& default_val = "") const;
    int asInt(int default_val = 0) const;
    float asFloat(float default_val = 0.0f) const;
    bool asBool(bool default_val = false) const;
    std::vector<std::string> asStringList() const;
    
    bool isValid() const { return !str_value_.empty() || has_int_ || has_float_ || has_bool_; }

private:
    std::string str_value_;
    int int_value_ = 0;
    float float_value_ = 0.0f;
    bool bool_value_ = false;
    bool has_int_ = false;
    bool has_float_ = false;
    bool has_bool_ = false;
};

/**
 * @brief 配置节点类
 */
class ConfigNode {
public:
    ConfigNode() = default;
    
    // 获取子节点
    std::shared_ptr<ConfigNode> getChild(const std::string& key) const;
    
    // 获取配置值
    ConfigValue getValue(const std::string& key) const;
    
    // 设置配置值
    void setValue(const std::string& key, const ConfigValue& value);
    void setChild(const std::string& key, std::shared_ptr<ConfigNode> child);
    
    // 检查键是否存在
    bool hasKey(const std::string& key) const;
    
    // 获取所有键
    std::vector<std::string> getKeys() const;

private:
    std::map<std::string, ConfigValue> values_;
    std::map<std::string, std::shared_ptr<ConfigNode>> children_;
};

/**
 * @brief 配置加载器类
 */
class ConfigLoader {
public:
    ConfigLoader() = default;
    ~ConfigLoader() = default;
    
    /**
     * @brief 从文件加载配置
     */
    bool loadFromFile(const std::string& file_path);
    
    /**
     * @brief 从字符串加载配置
     */
    bool loadFromString(const std::string& config_data);
    
    /**
     * @brief 保存配置到文件
     */
    bool saveToFile(const std::string& file_path) const;
    
    /**
     * @brief 获取根节点
     */
    std::shared_ptr<ConfigNode> getRoot() const { return root_; }
    
    /**
     * @brief 获取配置值（支持路径格式 "section.key"）
     */
    ConfigValue getValue(const std::string& path, const ConfigValue& default_value = ConfigValue()) const;
    
    /**
     * @brief 设置配置值
     */
    void setValue(const std::string& path, const ConfigValue& value);
    
    /**
     * @brief 检查配置是否有效
     */
    bool isValid() const { return root_ != nullptr; }
    
    /**
     * @brief 获取错误信息
     */
    std::string getLastError() const { return last_error_; }

private:
    std::shared_ptr<ConfigNode> parseYamlContent(const std::string& content);
    std::shared_ptr<ConfigNode> findNode(const std::string& path) const;
    std::vector<std::string> splitPath(const std::string& path) const;
    
private:
    std::shared_ptr<ConfigNode> root_;
    std::string last_error_;
};

} // namespace utils
} // namespace bamboo_cut