#include "configmanager.h"
#include <QtCore/QSettings>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonDocument>
#include <QtCore/QFileSystemWatcher>
#include <QtCore/QTimer>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>

Q_LOGGING_CATEGORY(configManager, "app.configmanager")

const QString ConfigManager::DEFAULT_CONFIG_DIR = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
const QString ConfigManager::DEFAULT_CONFIG_FILE = "config.json";

ConfigManager::ConfigManager(QObject *parent)
    : QObject(parent)
    , m_fileWatcher(nullptr)
    , m_autoSaveTimer(nullptr)
    , m_isValid(false)
    , m_autoSave(true)
    , m_isDirty(false)
{
    qCInfo(configManager) << "Creating ConfigManager...";
    
    // 创建文件监视器
    m_fileWatcher = new QFileSystemWatcher(this);
    connect(m_fileWatcher, &QFileSystemWatcher::fileChanged, 
            this, &ConfigManager::onConfigFileChanged);
    
    // 创建自动保存定时器
    m_autoSaveTimer = new QTimer(this);
    m_autoSaveTimer->setSingleShot(true);
    m_autoSaveTimer->setInterval(5000); // 5秒后自动保存
    connect(m_autoSaveTimer, &QTimer::timeout, this, [this]() {
        if (m_isDirty) {
            save();
        }
    });
}

ConfigManager::~ConfigManager()
{
    qCInfo(configManager) << "Destroying ConfigManager...";
    
    if (m_isDirty && m_autoSave) {
        save();
    }
}

bool ConfigManager::initialize(const QString& configPath)
{
    qCInfo(configManager) << "Initializing ConfigManager with path:" << configPath;
    
    if (configPath.isEmpty()) {
        m_configFilePath = DEFAULT_CONFIG_DIR + "/" + DEFAULT_CONFIG_FILE;
    } else {
        m_configFilePath = configPath;
    }
    
    // 确保配置目录存在
    QFileInfo fileInfo(m_configFilePath);
    QDir configDir = fileInfo.dir();
    if (!configDir.exists()) {
        if (!configDir.mkpath(".")) {
            setError("Failed to create config directory");
            return false;
        }
    }
    
    // 加载默认配置
    loadDefaults();
    
    // 尝试加载配置文件
    if (QFileInfo::exists(m_configFilePath)) {
        if (!loadFromFile()) {
            qCWarning(configManager) << "Failed to load config file, using defaults";
        }
    } else {
        qCInfo(configManager) << "Config file not found, creating with defaults";
        if (!saveToFile()) {
            setError("Failed to create default config file");
            return false;
        }
    }
    
    // 添加文件监视
    if (!m_fileWatcher->addPath(m_configFilePath)) {
        qCWarning(configManager) << "Failed to add config file to watcher";
    }
    
    // 验证配置
    validateConfig();
    
    m_isValid = true;
    qCInfo(configManager) << "ConfigManager initialized successfully";
    return true;
}

bool ConfigManager::save()
{
    qCInfo(configManager) << "Saving configuration...";
    
    if (!m_isValid) {
        setError("ConfigManager not initialized");
        return false;
    }
    
    bool success = saveToFile();
    if (success) {
        m_isDirty = false;
        m_autoSaveTimer->stop();
        qCInfo(configManager) << "Configuration saved successfully";
    } else {
        setError("Failed to save configuration");
    }
    
    return success;
}

bool ConfigManager::load()
{
    qCInfo(configManager) << "Loading configuration...";
    
    if (!m_isValid) {
        setError("ConfigManager not initialized");
        return false;
    }
    
    bool success = loadFromFile();
    if (success) {
        validateConfig();
        qCInfo(configManager) << "Configuration loaded successfully";
    } else {
        setError("Failed to load configuration");
    }
    
    return success;
}

void ConfigManager::reset()
{
    qCInfo(configManager) << "Resetting configuration to defaults...";
    
    loadDefaults();
    m_isDirty = true;
    
    if (m_autoSave) {
        m_autoSaveTimer->start();
    }
    
    // 发射配置更改信号
    emit systemConfigChanged(m_systemConfig);
    emit cameraConfigChanged(m_cameraConfig);
    emit detectionConfigChanged(m_detectionConfig);
    emit uiConfigChanged(m_uiConfig);
    emit networkConfigChanged(m_networkConfig);
    emit hardwareConfigChanged(m_hardwareConfig);
    
    qCInfo(configManager) << "Configuration reset to defaults";
}

void ConfigManager::setSystemConfig(const SystemConfig& config)
{
    if (memcmp(&m_systemConfig, &config, sizeof(SystemConfig)) != 0) {
        m_systemConfig = config;
        m_isDirty = true;
        
        if (m_autoSave) {
            m_autoSaveTimer->start();
        }
        
        emit configChanged(System);
        emit systemConfigChanged(config);
    }
}

void ConfigManager::setCameraConfig(const CameraConfig& config)
{
    if (memcmp(&m_cameraConfig, &config, sizeof(CameraConfig)) != 0) {
        m_cameraConfig = config;
        m_isDirty = true;
        
        if (m_autoSave) {
            m_autoSaveTimer->start();
        }
        
        emit configChanged(Camera);
        emit cameraConfigChanged(config);
    }
}

void ConfigManager::setDetectionConfig(const DetectionConfig& config)
{
    if (memcmp(&m_detectionConfig, &config, sizeof(DetectionConfig)) != 0) {
        m_detectionConfig = config;
        m_isDirty = true;
        
        if (m_autoSave) {
            m_autoSaveTimer->start();
        }
        
        emit configChanged(Detection);
        emit detectionConfigChanged(config);
    }
}

void ConfigManager::setUIConfig(const UIConfig& config)
{
    if (memcmp(&m_uiConfig, &config, sizeof(UIConfig)) != 0) {
        m_uiConfig = config;
        m_isDirty = true;
        
        if (m_autoSave) {
            m_autoSaveTimer->start();
        }
        
        emit configChanged(UI);
        emit uiConfigChanged(config);
    }
}

void ConfigManager::setNetworkConfig(const NetworkConfig& config)
{
    if (memcmp(&m_networkConfig, &config, sizeof(NetworkConfig)) != 0) {
        m_networkConfig = config;
        m_isDirty = true;
        
        if (m_autoSave) {
            m_autoSaveTimer->start();
        }
        
        emit configChanged(Network);
        emit networkConfigChanged(config);
    }
}

void ConfigManager::setHardwareConfig(const HardwareConfig& config)
{
    if (memcmp(&m_hardwareConfig, &config, sizeof(HardwareConfig)) != 0) {
        m_hardwareConfig = config;
        m_isDirty = true;
        
        if (m_autoSave) {
            m_autoSaveTimer->start();
        }
        
        emit configChanged(Hardware);
        emit hardwareConfigChanged(config);
    }
}

QVariant ConfigManager::getValue(ConfigSection section, const QString& key, const QVariant& defaultValue) const
{
    QJsonObject json = configToJson();
    QString sectionName = sectionToString(section);
    
    if (!json.contains(sectionName)) {
        return defaultValue;
    }
    
    QJsonObject sectionObj = json[sectionName].toObject();
    if (!sectionObj.contains(key)) {
        return defaultValue;
    }
    
    return sectionObj[key].toVariant();
}

void ConfigManager::setValue(ConfigSection section, const QString& key, const QVariant& value)
{
    // 这里可以实现动态设置配置值的功能
    // 为了简化，目前使用结构化的配置设置方法
    m_isDirty = true;
    
    if (m_autoSave) {
        m_autoSaveTimer->start();
    }
    
    emit configChanged(section);
    
    qCDebug(configManager) << "Set" << sectionToString(section) << key << "to" << value;
}

void ConfigManager::onConfigFileChanged(const QString& path)
{
    Q_UNUSED(path)
    
    qCInfo(configManager) << "Config file changed externally, reloading...";
    
    // 短暂延迟，避免文件正在写入时读取
    QTimer::singleShot(100, this, [this]() {
        if (loadFromFile()) {
            validateConfig();
            emit configFileChanged();
            
            // 发射所有配置更改信号
            emit systemConfigChanged(m_systemConfig);
            emit cameraConfigChanged(m_cameraConfig);
            emit detectionConfigChanged(m_detectionConfig);
            emit uiConfigChanged(m_uiConfig);
            emit networkConfigChanged(m_networkConfig);
            emit hardwareConfigChanged(m_hardwareConfig);
        }
    });
}

void ConfigManager::onAutoSaveTimer()
{
    if (m_isDirty) {
        save();
    }
}

void ConfigManager::loadDefaults()
{
    qCInfo(configManager) << "Loading default configuration...";
    
    // 系统配置默认值
    m_systemConfig.language = "zh_CN";
    m_systemConfig.theme = "light";
    m_systemConfig.autoStart = false;
    m_systemConfig.enableLogging = true;
    m_systemConfig.logLevel = "info";
    m_systemConfig.maxLogFiles = 10;
    m_systemConfig.enableDebug = false;
    
    // 摄像头配置默认值
    m_cameraConfig.deviceId = 0;
    m_cameraConfig.width = 1920;
    m_cameraConfig.height = 1080;
    m_cameraConfig.fps = 30;
    m_cameraConfig.pipeline = "";
    m_cameraConfig.useHardwareAcceleration = true;
    m_cameraConfig.enableAutoExposure = true;
    m_cameraConfig.enableAutoWhiteBalance = true;
    m_cameraConfig.exposure = -1;
    m_cameraConfig.gain = -1;
    m_cameraConfig.brightness = 0;
    m_cameraConfig.contrast = 0;
    
    // 检测配置默认值
    m_detectionConfig.modelPath = "../models/best.pt";
    m_detectionConfig.configPath = "";
    m_detectionConfig.classNamesPath = "";
    m_detectionConfig.confidenceThreshold = 0.7f;
    m_detectionConfig.nmsThreshold = 0.4f;
    m_detectionConfig.useGPU = true;
    m_detectionConfig.useTensorRT = true;
    m_detectionConfig.useINT8 = true;
    m_detectionConfig.batchSize = 1;
    m_detectionConfig.maxDetections = 100;
    
    // UI配置默认值
    m_uiConfig.windowWidth = 1024;
    m_uiConfig.windowHeight = 768;
    m_uiConfig.fullscreen = false;
    m_uiConfig.showStatusBar = true;
    m_uiConfig.showToolbar = true;
    m_uiConfig.styleSheet = "";
    m_uiConfig.fontSize = 12;
    m_uiConfig.enableTouchMode = true;
    m_uiConfig.touchSensitivity = 10;
    
    // 网络配置默认值
    m_networkConfig.serverAddress = "127.0.0.1";
    m_networkConfig.serverPort = 8080;
    m_networkConfig.serialPort = "/dev/ttyUSB0";
    m_networkConfig.baudRate = 115200;
    m_networkConfig.connectionTimeout = 5000;
    m_networkConfig.enableSSL = false;
    m_networkConfig.certificatePath = "";
    
    // 硬件配置默认值
    m_hardwareConfig.cutSpeed = 50;
    m_hardwareConfig.cutDepth = 30;
    m_hardwareConfig.bladePosition = 0;
    m_hardwareConfig.autoRetract = true;
    m_hardwareConfig.retractDelay = 1000;
    m_hardwareConfig.forceThreshold = 80.0f;
    m_hardwareConfig.controllerType = "modbus_rtu";
    m_hardwareConfig.customSettings = QJsonObject();
}

void ConfigManager::validateConfig()
{
    // 验证并修正配置值
    
    // 系统配置验证
    if (m_systemConfig.maxLogFiles < 1) {
        m_systemConfig.maxLogFiles = 1;
    }
    if (m_systemConfig.maxLogFiles > 100) {
        m_systemConfig.maxLogFiles = 100;
    }
    
    // 摄像头配置验证
    m_cameraConfig.width = qBound(320, m_cameraConfig.width, 3840);
    m_cameraConfig.height = qBound(240, m_cameraConfig.height, 2160);
    m_cameraConfig.fps = qBound(1, m_cameraConfig.fps, 120);
    m_cameraConfig.brightness = qBound(-100, m_cameraConfig.brightness, 100);
    m_cameraConfig.contrast = qBound(-100, m_cameraConfig.contrast, 100);
    
    // 检测配置验证
    m_detectionConfig.confidenceThreshold = qBound(0.0f, m_detectionConfig.confidenceThreshold, 1.0f);
    m_detectionConfig.nmsThreshold = qBound(0.0f, m_detectionConfig.nmsThreshold, 1.0f);
    m_detectionConfig.batchSize = qBound(1, m_detectionConfig.batchSize, 16);
    m_detectionConfig.maxDetections = qBound(1, m_detectionConfig.maxDetections, 1000);
    
    // UI配置验证
    m_uiConfig.windowWidth = qBound(800, m_uiConfig.windowWidth, 3840);
    m_uiConfig.windowHeight = qBound(600, m_uiConfig.windowHeight, 2160);
    m_uiConfig.fontSize = qBound(8, m_uiConfig.fontSize, 72);
    m_uiConfig.touchSensitivity = qBound(1, m_uiConfig.touchSensitivity, 100);
    
    // 网络配置验证
    m_networkConfig.serverPort = qBound(1024, m_networkConfig.serverPort, 65535);
    m_networkConfig.baudRate = qBound(9600, m_networkConfig.baudRate, 921600);
    m_networkConfig.connectionTimeout = qBound(1000, m_networkConfig.connectionTimeout, 60000);
    
    // 硬件配置验证
    m_hardwareConfig.cutSpeed = qBound(1, m_hardwareConfig.cutSpeed, 100);
    m_hardwareConfig.cutDepth = qBound(1, m_hardwareConfig.cutDepth, 100);
    m_hardwareConfig.retractDelay = qBound(0, m_hardwareConfig.retractDelay, 10000);
    m_hardwareConfig.forceThreshold = qBound(1.0f, m_hardwareConfig.forceThreshold, 1000.0f);
}

bool ConfigManager::loadFromFile()
{
    QFileInfo fileInfo(m_configFilePath);
    if (!fileInfo.exists()) {
        setError("Config file does not exist");
        return false;
    }
    
    QFile file(m_configFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        setError(QString("Cannot open config file: %1").arg(file.errorString()));
        return false;
    }
    
    QByteArray data = file.readAll();
    file.close();
    
    QJsonParseError parseError;
    QJsonDocument doc = QJsonDocument::fromJson(data, &parseError);
    
    if (parseError.error != QJsonParseError::NoError) {
        setError(QString("JSON parse error: %1").arg(parseError.errorString()));
        return false;
    }
    
    if (!doc.isObject()) {
        setError("Config file is not a JSON object");
        return false;
    }
    
    return configFromJson(doc.object());
}

bool ConfigManager::saveToFile()
{
    QJsonObject json = configToJson();
    QJsonDocument doc(json);
    
    QFile file(m_configFilePath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        setError(QString("Cannot write to config file: %1").arg(file.errorString()));
        return false;
    }
    
    QByteArray data = doc.toJson(QJsonDocument::Indented);
    if (file.write(data) != data.size()) {
        setError("Failed to write complete config data");
        file.close();
        return false;
    }
    
    file.close();
    return true;
}

QJsonObject ConfigManager::configToJson() const
{
    QJsonObject json;
    json["version"] = "2.0.0";
    
    // 系统配置
    QJsonObject system;
    system["language"] = m_systemConfig.language;
    system["theme"] = m_systemConfig.theme;
    system["autoStart"] = m_systemConfig.autoStart;
    system["enableLogging"] = m_systemConfig.enableLogging;
    system["logLevel"] = m_systemConfig.logLevel;
    system["maxLogFiles"] = m_systemConfig.maxLogFiles;
    system["enableDebug"] = m_systemConfig.enableDebug;
    json["system"] = system;
    
    // 摄像头配置
    QJsonObject camera;
    camera["deviceId"] = m_cameraConfig.deviceId;
    camera["width"] = m_cameraConfig.width;
    camera["height"] = m_cameraConfig.height;
    camera["fps"] = m_cameraConfig.fps;
    camera["pipeline"] = m_cameraConfig.pipeline;
    camera["useHardwareAcceleration"] = m_cameraConfig.useHardwareAcceleration;
    camera["enableAutoExposure"] = m_cameraConfig.enableAutoExposure;
    camera["enableAutoWhiteBalance"] = m_cameraConfig.enableAutoWhiteBalance;
    camera["exposure"] = m_cameraConfig.exposure;
    camera["gain"] = m_cameraConfig.gain;
    camera["brightness"] = m_cameraConfig.brightness;
    camera["contrast"] = m_cameraConfig.contrast;
    json["camera"] = camera;
    
    // 检测配置
    QJsonObject detection;
    detection["modelPath"] = m_detectionConfig.modelPath;
    detection["configPath"] = m_detectionConfig.configPath;
    detection["classNamesPath"] = m_detectionConfig.classNamesPath;
    detection["confidenceThreshold"] = m_detectionConfig.confidenceThreshold;
    detection["nmsThreshold"] = m_detectionConfig.nmsThreshold;
    detection["useGPU"] = m_detectionConfig.useGPU;
    detection["useTensorRT"] = m_detectionConfig.useTensorRT;
    detection["useINT8"] = m_detectionConfig.useINT8;
    detection["batchSize"] = m_detectionConfig.batchSize;
    detection["maxDetections"] = m_detectionConfig.maxDetections;
    json["detection"] = detection;
    
    // UI配置
    QJsonObject ui;
    ui["windowWidth"] = m_uiConfig.windowWidth;
    ui["windowHeight"] = m_uiConfig.windowHeight;
    ui["fullscreen"] = m_uiConfig.fullscreen;
    ui["showStatusBar"] = m_uiConfig.showStatusBar;
    ui["showToolbar"] = m_uiConfig.showToolbar;
    ui["styleSheet"] = m_uiConfig.styleSheet;
    ui["fontSize"] = m_uiConfig.fontSize;
    ui["enableTouchMode"] = m_uiConfig.enableTouchMode;
    ui["touchSensitivity"] = m_uiConfig.touchSensitivity;
    json["ui"] = ui;
    
    // 网络配置
    QJsonObject network;
    network["serverAddress"] = m_networkConfig.serverAddress;
    network["serverPort"] = m_networkConfig.serverPort;
    network["serialPort"] = m_networkConfig.serialPort;
    network["baudRate"] = m_networkConfig.baudRate;
    network["connectionTimeout"] = m_networkConfig.connectionTimeout;
    network["enableSSL"] = m_networkConfig.enableSSL;
    network["certificatePath"] = m_networkConfig.certificatePath;
    json["network"] = network;
    
    // 硬件配置
    QJsonObject hardware;
    hardware["cutSpeed"] = m_hardwareConfig.cutSpeed;
    hardware["cutDepth"] = m_hardwareConfig.cutDepth;
    hardware["bladePosition"] = m_hardwareConfig.bladePosition;
    hardware["autoRetract"] = m_hardwareConfig.autoRetract;
    hardware["retractDelay"] = m_hardwareConfig.retractDelay;
    hardware["forceThreshold"] = m_hardwareConfig.forceThreshold;
    hardware["controllerType"] = m_hardwareConfig.controllerType;
    hardware["customSettings"] = m_hardwareConfig.customSettings;
    json["hardware"] = hardware;
    
    return json;
}

bool ConfigManager::configFromJson(const QJsonObject& json)
{
    try {
        // 系统配置
        if (json.contains("system")) {
            QJsonObject system = json["system"].toObject();
            m_systemConfig.language = system.value("language").toString("zh_CN");
            m_systemConfig.theme = system.value("theme").toString("light");
            m_systemConfig.autoStart = system.value("autoStart").toBool(false);
            m_systemConfig.enableLogging = system.value("enableLogging").toBool(true);
            m_systemConfig.logLevel = system.value("logLevel").toString("info");
            m_systemConfig.maxLogFiles = system.value("maxLogFiles").toInt(10);
            m_systemConfig.enableDebug = system.value("enableDebug").toBool(false);
        }
        
        // 摄像头配置
        if (json.contains("camera")) {
            QJsonObject camera = json["camera"].toObject();
            m_cameraConfig.deviceId = camera.value("deviceId").toInt(0);
            m_cameraConfig.width = camera.value("width").toInt(1920);
            m_cameraConfig.height = camera.value("height").toInt(1080);
            m_cameraConfig.fps = camera.value("fps").toInt(30);
            m_cameraConfig.pipeline = camera.value("pipeline").toString("");
            m_cameraConfig.useHardwareAcceleration = camera.value("useHardwareAcceleration").toBool(true);
            m_cameraConfig.enableAutoExposure = camera.value("enableAutoExposure").toBool(true);
            m_cameraConfig.enableAutoWhiteBalance = camera.value("enableAutoWhiteBalance").toBool(true);
            m_cameraConfig.exposure = camera.value("exposure").toInt(-1);
            m_cameraConfig.gain = camera.value("gain").toInt(-1);
            m_cameraConfig.brightness = camera.value("brightness").toInt(0);
            m_cameraConfig.contrast = camera.value("contrast").toInt(0);
        }
        
        // 检测配置
        if (json.contains("detection")) {
            QJsonObject detection = json["detection"].toObject();
            m_detectionConfig.modelPath = detection.value("modelPath").toString("../models/best.pt");
            m_detectionConfig.configPath = detection.value("configPath").toString("");
            m_detectionConfig.classNamesPath = detection.value("classNamesPath").toString("");
            m_detectionConfig.confidenceThreshold = static_cast<float>(detection.value("confidenceThreshold").toDouble(0.7));
            m_detectionConfig.nmsThreshold = static_cast<float>(detection.value("nmsThreshold").toDouble(0.4));
            m_detectionConfig.useGPU = detection.value("useGPU").toBool(true);
            m_detectionConfig.useTensorRT = detection.value("useTensorRT").toBool(true);
            m_detectionConfig.useINT8 = detection.value("useINT8").toBool(true);
            m_detectionConfig.batchSize = detection.value("batchSize").toInt(1);
            m_detectionConfig.maxDetections = detection.value("maxDetections").toInt(100);
        }
        
        // UI配置
        if (json.contains("ui")) {
            QJsonObject ui = json["ui"].toObject();
            m_uiConfig.windowWidth = ui.value("windowWidth").toInt(1024);
            m_uiConfig.windowHeight = ui.value("windowHeight").toInt(768);
            m_uiConfig.fullscreen = ui.value("fullscreen").toBool(false);
            m_uiConfig.showStatusBar = ui.value("showStatusBar").toBool(true);
            m_uiConfig.showToolbar = ui.value("showToolbar").toBool(true);
            m_uiConfig.styleSheet = ui.value("styleSheet").toString("");
            m_uiConfig.fontSize = ui.value("fontSize").toInt(12);
            m_uiConfig.enableTouchMode = ui.value("enableTouchMode").toBool(true);
            m_uiConfig.touchSensitivity = ui.value("touchSensitivity").toInt(10);
        }
        
        // 网络配置
        if (json.contains("network")) {
            QJsonObject network = json["network"].toObject();
            m_networkConfig.serverAddress = network.value("serverAddress").toString("127.0.0.1");
            m_networkConfig.serverPort = network.value("serverPort").toInt(8080);
            m_networkConfig.serialPort = network.value("serialPort").toString("/dev/ttyUSB0");
            m_networkConfig.baudRate = network.value("baudRate").toInt(115200);
            m_networkConfig.connectionTimeout = network.value("connectionTimeout").toInt(5000);
            m_networkConfig.enableSSL = network.value("enableSSL").toBool(false);
            m_networkConfig.certificatePath = network.value("certificatePath").toString("");
        }
        
        // 硬件配置
        if (json.contains("hardware")) {
            QJsonObject hardware = json["hardware"].toObject();
            m_hardwareConfig.cutSpeed = hardware.value("cutSpeed").toInt(50);
            m_hardwareConfig.cutDepth = hardware.value("cutDepth").toInt(30);
            m_hardwareConfig.bladePosition = hardware.value("bladePosition").toInt(0);
            m_hardwareConfig.autoRetract = hardware.value("autoRetract").toBool(true);
            m_hardwareConfig.retractDelay = hardware.value("retractDelay").toInt(1000);
            m_hardwareConfig.forceThreshold = static_cast<float>(hardware.value("forceThreshold").toDouble(80.0));
            m_hardwareConfig.controllerType = hardware.value("controllerType").toString("modbus_rtu");
            m_hardwareConfig.customSettings = hardware.value("customSettings").toObject();
        }
        
        return true;
        
    } catch (const std::exception& e) {
        setError(QString("Error parsing config JSON: %1").arg(e.what()));
        return false;
    }
}

QString ConfigManager::sectionToString(ConfigSection section) const
{
    switch (section) {
    case System: return "system";
    case Camera: return "camera";
    case Detection: return "detection";
    case UI: return "ui";
    case Network: return "network";
    case Hardware: return "hardware";
    default: return "unknown";
    }
}

void ConfigManager::setError(const QString& error)
{
    m_lastError = error;
    qCWarning(configManager) << "ConfigManager error:" << error;
    emit configError(error);
}