#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H

#include <QtCore/QObject>
#include <QtCore/QSettings>
#include <QtCore/QJsonObject>
#include <QtCore/QJsonDocument>
#include <QtCore/QFileSystemWatcher>
#include <QtCore/QTimer>

class ConfigManager : public QObject
{
    Q_OBJECT

public:
    enum ConfigSection {
        System,
        Camera,
        Detection,
        UI,
        Network,
        Hardware
    };

    struct SystemConfig {
        QString language;
        QString theme;
        bool autoStart;
        bool enableLogging;
        QString logLevel;
        int maxLogFiles;
        bool enableDebug;
    };

    struct CameraConfig {
        int deviceId;
        int width;
        int height;
        int fps;
        QString pipeline;
        bool useHardwareAcceleration;
        bool enableAutoExposure;
        bool enableAutoWhiteBalance;
        int exposure;
        int gain;
        int brightness;
        int contrast;
    };

    struct DetectionConfig {
        QString modelPath;
        QString configPath;
        QString classNamesPath;
        float confidenceThreshold;
        float nmsThreshold;
        bool useGPU;
        bool useTensorRT;
        bool useINT8;
        int batchSize;
        int maxDetections;
    };

    struct UIConfig {
        int windowWidth;
        int windowHeight;
        bool fullscreen;
        bool showStatusBar;
        bool showToolbar;
        QString styleSheet;
        int fontSize;
        bool enableTouchMode;
        int touchSensitivity;
    };

    struct NetworkConfig {
        QString serverAddress;
        int serverPort;
        QString serialPort;
        int baudRate;
        int connectionTimeout;
        bool enableSSL;
        QString certificatePath;
    };

    struct HardwareConfig {
        int cutSpeed;
        int cutDepth;
        int bladePosition;
        bool autoRetract;
        int retractDelay;
        float forceThreshold;
        QString controllerType;
        QJsonObject customSettings;
    };

    explicit ConfigManager(QObject *parent = nullptr);
    ~ConfigManager();

    bool initialize(const QString& configPath = QString());
    bool save();
    bool load();
    void reset();
    
    // 配置访问
    SystemConfig systemConfig() const { return m_systemConfig; }
    CameraConfig cameraConfig() const { return m_cameraConfig; }
    DetectionConfig detectionConfig() const { return m_detectionConfig; }
    UIConfig uiConfig() const { return m_uiConfig; }
    NetworkConfig networkConfig() const { return m_networkConfig; }
    HardwareConfig hardwareConfig() const { return m_hardwareConfig; }

    // 配置设置
    void setSystemConfig(const SystemConfig& config);
    void setCameraConfig(const CameraConfig& config);
    void setDetectionConfig(const DetectionConfig& config);
    void setUIConfig(const UIConfig& config);
    void setNetworkConfig(const NetworkConfig& config);
    void setHardwareConfig(const HardwareConfig& config);

    // 通用配置接口
    QVariant getValue(ConfigSection section, const QString& key, const QVariant& defaultValue = QVariant()) const;
    void setValue(ConfigSection section, const QString& key, const QVariant& value);
    
    // 配置文件管理
    QString configFilePath() const { return m_configFilePath; }
    bool isValid() const { return m_isValid; }
    QString lastError() const { return m_lastError; }

signals:
    void configChanged(ConfigSection section);
    void systemConfigChanged(const SystemConfig& config);
    void cameraConfigChanged(const CameraConfig& config);
    void detectionConfigChanged(const DetectionConfig& config);
    void uiConfigChanged(const UIConfig& config);
    void networkConfigChanged(const NetworkConfig& config);
    void hardwareConfigChanged(const HardwareConfig& config);
    void configFileChanged();
    void configError(const QString& error);

private slots:
    void onConfigFileChanged(const QString& path);
    void onAutoSaveTimer();

private:
    void loadDefaults();
    void validateConfig();
    bool loadFromFile();
    bool saveToFile();
    QJsonObject configToJson() const;
    bool configFromJson(const QJsonObject& json);
    QString sectionToString(ConfigSection section) const;
    void setError(const QString& error);

    // 配置数据
    SystemConfig m_systemConfig;
    CameraConfig m_cameraConfig;
    DetectionConfig m_detectionConfig;
    UIConfig m_uiConfig;
    NetworkConfig m_networkConfig;
    HardwareConfig m_hardwareConfig;

    // 文件管理
    QString m_configFilePath;
    QFileSystemWatcher *m_fileWatcher;
    QTimer *m_autoSaveTimer;
    
    // 状态
    bool m_isValid;
    QString m_lastError;
    bool m_autoSave;
    bool m_isDirty;
    
    // 默认配置路径
    static const QString DEFAULT_CONFIG_DIR;
    static const QString DEFAULT_CONFIG_FILE;
};

Q_DECLARE_METATYPE(ConfigManager::SystemConfig)
Q_DECLARE_METATYPE(ConfigManager::CameraConfig)
Q_DECLARE_METATYPE(ConfigManager::DetectionConfig)
Q_DECLARE_METATYPE(ConfigManager::UIConfig)
Q_DECLARE_METATYPE(ConfigManager::NetworkConfig)
Q_DECLARE_METATYPE(ConfigManager::HardwareConfig)

#endif // CONFIGMANAGER_H