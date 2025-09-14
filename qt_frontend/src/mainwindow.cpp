#include "mainwindow.h"
#include "videorenderer.h"
#include "touchcontroller.h"
#include "cameramanager.h"
#include "bamboodetector.h"
#include "systemcontroller.h"
#include "configmanager.h"

#include <QtWidgets/QApplication>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QLabel>
#include <QtWidgets/QProgressBar>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QMessageBox>
#include <QtCore/QTimer>
#include <QtCore/QStandardPaths>
#include <QtCore/QDir>
#include <QtCore/QLoggingCategory>
#include <QtCore/QDateTime>
#include <QtQml/QQmlContext>
#include <QtGui/QResizeEvent>
#include <QtGui/QCloseEvent>

Q_LOGGING_CATEGORY(mainWindow, "app.mainwindow")

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_centralWidget(nullptr)
    , m_mainLayout(nullptr)
    , m_topLayout(nullptr)
    , m_bottomLayout(nullptr)
    , m_videoWidget(nullptr)
    , m_startButton(nullptr)
    , m_stopButton(nullptr)
    , m_settingsButton(nullptr)
    , m_statusLabel(nullptr)
    , m_frameRateLabel(nullptr)
    , m_detectionLabel(nullptr)
    , m_progressBar(nullptr)
    , m_stackedWidget(nullptr)
    , m_statusTimer(nullptr)
    , m_isRunning(false)
    , m_frameCount(0)
    , m_lastFrameTime(0)
{
    qCInfo(mainWindow) << "Initializing MainWindow...";
    
    // 设置窗口属性
    setWindowTitle("智能切竹机控制系统 v2.0.0");
    setMinimumSize(800, 600);
    resize(1024, 768);
    
    // 启用触摸事件
    setAttribute(Qt::WA_AcceptTouchEvents, true);
    
    // 初始化核心组件
    initializeComponents();
    
    // 设置UI
    setupUI();
    
    // 建立连接
    setupConnections();
    
    // 启动状态定时器
    m_statusTimer = new QTimer(this);
    connect(m_statusTimer, &QTimer::timeout, this, &MainWindow::updateStatusBar);
    m_statusTimer->start(1000); // 每秒更新一次状态
    
    qCInfo(mainWindow) << "MainWindow initialized successfully";
}

MainWindow::~MainWindow()
{
    qCInfo(mainWindow) << "Destroying MainWindow...";
    
    if (m_isRunning) {
        stopOperation();
    }
    
    // 组件会通过智能指针自动释放
}

void MainWindow::initializeComponents()
{
    // 初始化配置管理器
    m_configManager = std::make_unique<ConfigManager>(this);
    QString configPath = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
    QDir().mkpath(configPath);
    m_configManager->initialize(configPath + "/config.json");
    
    // 初始化视频渲染器
    m_videoRenderer = std::make_unique<VideoRenderer>(this);
    
    // 初始化触摸控制器
    m_touchController = std::make_unique<TouchController>(this);
    
    // 初始化相机管理器
    m_cameraManager = std::make_unique<CameraManager>(this);
    
    // 初始化AI检测器
    m_bambooDetector = std::make_unique<BambooDetector>(this);
    
    // 初始化系统控制器
    m_systemController = std::make_unique<SystemController>(this);
    
    // 配置各组件
    configureComponents();
}

void MainWindow::configureComponents()
{
    // 配置相机
    auto cameraConfig = m_configManager->cameraConfig();
    CameraManager::CameraConfig camConfig;
    camConfig.deviceId = cameraConfig.deviceId;
    camConfig.width = cameraConfig.width;
    camConfig.height = cameraConfig.height;
    camConfig.fps = cameraConfig.fps;
    camConfig.type = CameraManager::CSI_Camera; // Jetson默认使用CSI
    camConfig.useHardwareAcceleration = cameraConfig.useHardwareAcceleration;
    camConfig.enableAutoExposure = cameraConfig.enableAutoExposure;
    camConfig.enableAutoWhiteBalance = cameraConfig.enableAutoWhiteBalance;
    camConfig.bufferSize = 3;
    
    // GStreamer管道配置（针对Jetson优化）
    camConfig.pipeline = QString(
        "nvarguscamerasrc sensor_id=%1 ! "
        "video/x-raw(memory:NVMM), width=%2, height=%3, framerate=%4/1, format=NV12 ! "
        "nvvidconv ! video/x-raw, format=BGRx ! "
        "videoconvert ! video/x-raw, format=BGR ! appsink"
    ).arg(camConfig.deviceId).arg(camConfig.width).arg(camConfig.height).arg(camConfig.fps);
    
    m_cameraManager->initialize(camConfig);
    
    // 配置AI检测器
    auto detectionConfig = m_configManager->detectionConfig();
    BambooDetector::ModelConfig modelConfig;
    modelConfig.modelPath = detectionConfig.modelPath;
    modelConfig.confidenceThreshold = detectionConfig.confidenceThreshold;
    modelConfig.nmsThreshold = detectionConfig.nmsThreshold;
    modelConfig.inputSize = cv::Size(640, 640); // YOLO标准输入尺寸
    modelConfig.useGPU = detectionConfig.useGPU;
    modelConfig.useTensorRT = detectionConfig.useTensorRT;
    modelConfig.useINT8 = detectionConfig.useINT8;
    modelConfig.batchSize = detectionConfig.batchSize;
    
    m_bambooDetector->initialize(modelConfig);
    
    // 初始化系统控制器
    m_systemController->initialize();
}

void MainWindow::setupUI()
{
    // 创建中央组件
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);
    
    // 主布局
    m_mainLayout = new QVBoxLayout(m_centralWidget);
    m_mainLayout->setContentsMargins(5, 5, 5, 5);
    m_mainLayout->setSpacing(5);
    
    // 顶部布局（视频显示区域）
    m_topLayout = new QHBoxLayout();
    m_topLayout->setSpacing(10);
    
    // 创建QML视频显示组件
    m_videoWidget = new QQuickWidget(this);
    m_videoWidget->setSource(QUrl("qrc:/qml/main.qml"));
    m_videoWidget->setResizeMode(QQuickWidget::SizeRootObjectToView);
    m_videoWidget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    
    // 设置视频渲染器到QML
    m_videoWidget->rootContext()->setContextProperty("videoRenderer", m_videoRenderer.get());
    m_videoWidget->rootContext()->setContextProperty("touchController", m_touchController.get());
    m_videoWidget->rootContext()->setContextProperty("cameraManager", m_cameraManager.get());
    m_videoWidget->rootContext()->setContextProperty("bambooDetector", m_bambooDetector.get());
    m_videoWidget->rootContext()->setContextProperty("systemController", m_systemController.get());
    m_videoWidget->rootContext()->setContextProperty("configManager", m_configManager.get());
    
    m_topLayout->addWidget(m_videoWidget, 3); // 占用3/4的空间
    
    // 创建控制面板（暂时用传统Widget，后续可迁移到QML）
    createControlPanel();
    
    m_mainLayout->addLayout(m_topLayout, 1);
    
    // 底部状态栏
    createStatusBar();
}

void MainWindow::createControlPanel()
{
    // 控制面板容器
    auto controlPanel = new QWidget(this);
    controlPanel->setMaximumWidth(300);
    controlPanel->setMinimumWidth(250);
    controlPanel->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);
    controlPanel->setStyleSheet("QWidget { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; }");
    
    auto controlLayout = new QVBoxLayout(controlPanel);
    controlLayout->setContentsMargins(15, 15, 15, 15);
    controlLayout->setSpacing(10);
    
    // 操作控制组
    auto operationGroup = new QWidget();
    auto operationLayout = new QVBoxLayout(operationGroup);
    
    // 主操作按钮
    auto buttonLayout = new QHBoxLayout();
    
    m_startButton = new QPushButton("开始作业", this);
    m_startButton->setStyleSheet("QPushButton { background-color: #28a745; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }");
    m_startButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    
    m_stopButton = new QPushButton("停止作业", this);
    m_stopButton->setStyleSheet("QPushButton { background-color: #fd7e14; color: white; font-weight: bold; padding: 10px; border-radius: 5px; }");
    m_stopButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_stopButton->setEnabled(false);
    
    buttonLayout->addWidget(m_startButton);
    buttonLayout->addWidget(m_stopButton);
    
    // 紧急停止按钮
    auto emergencyButton = new QPushButton("🚨 紧急停止 🚨", this);
    emergencyButton->setStyleSheet("QPushButton { background-color: #dc3545; color: white; font-weight: bold; font-size: 16px; padding: 15px; border-radius: 5px; }");
    
    // 设置按钮
    m_settingsButton = new QPushButton("系统设置", this);
    m_settingsButton->setStyleSheet("QPushButton { background-color: #6c757d; color: white; padding: 8px; border-radius: 5px; }");
    
    operationLayout->addLayout(buttonLayout);
    operationLayout->addWidget(emergencyButton);
    operationLayout->addWidget(m_settingsButton);
    
    controlLayout->addWidget(operationGroup);
    controlLayout->addStretch(); // 添加弹性空间
    
    m_topLayout->addWidget(controlPanel, 1); // 占用1/4的空间
}

void MainWindow::createStatusBar()
{
    // 底部状态栏
    m_bottomLayout = new QHBoxLayout();
    m_bottomLayout->setSpacing(15);
    
    // 系统状态
    m_statusLabel = new QLabel("系统状态: 待机", this);
    m_statusLabel->setStyleSheet("QLabel { font-weight: bold; color: #495057; }");
    
    // 帧率显示
    m_frameRateLabel = new QLabel("帧率: 0.0 FPS", this);
    m_frameRateLabel->setStyleSheet("QLabel { color: #6c757d; }");
    
    // 检测精度
    m_detectionLabel = new QLabel("检测精度: 0%", this);
    m_detectionLabel->setStyleSheet("QLabel { color: #6c757d; }");
    
    // 进度条
    m_progressBar = new QProgressBar(this);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_progressBar->setVisible(false);
    
    m_bottomLayout->addWidget(m_statusLabel);
    m_bottomLayout->addWidget(m_frameRateLabel);
    m_bottomLayout->addWidget(m_detectionLabel);
    m_bottomLayout->addStretch();
    m_bottomLayout->addWidget(m_progressBar);
    
    m_mainLayout->addLayout(m_bottomLayout);
}

void MainWindow::setupConnections()
{
    // 按钮连接
    connect(m_startButton, &QPushButton::clicked, this, &MainWindow::onStartButtonClicked);
    connect(m_stopButton, &QPushButton::clicked, this, &MainWindow::onStopButtonClicked);
    connect(m_settingsButton, &QPushButton::clicked, this, &MainWindow::onSettingsButtonClicked);
    
    // 相机管理器连接
    connect(m_cameraManager.get(), &CameraManager::frameReady, 
            this, &MainWindow::onCameraFrameReady);
    connect(m_cameraManager.get(), &CameraManager::statusChanged,
            this, [this](CameraManager::CameraStatus status) {
                QString statusText;
                switch(status) {
                case CameraManager::Connected: statusText = "相机已连接"; break;
                case CameraManager::Streaming: statusText = "视频流中"; break;
                case CameraManager::Error: statusText = "相机错误"; break;
                default: statusText = "相机未连接"; break;
                }
                qCInfo(mainWindow) << "Camera status changed:" << statusText;
            });
    
    // AI检测器连接
    connect(m_bambooDetector.get(), &BambooDetector::detectionReady,
            this, &MainWindow::onDetectionResult);
    connect(m_bambooDetector.get(), &BambooDetector::statusChanged,
            this, [this](BambooDetector::DetectionStatus status) {
                QString statusText;
                switch(status) {
                case BambooDetector::Ready: statusText = "AI就绪"; break;
                case BambooDetector::Processing: statusText = "AI处理中"; break;
                case BambooDetector::Error: statusText = "AI错误"; break;
                default: statusText = "AI未就绪"; break;
                }
                qCInfo(mainWindow) << "Detection status changed:" << statusText;
            });
    
    // 系统控制器连接
    connect(m_systemController.get(), &SystemController::systemStatusChanged,
            this, &MainWindow::onSystemStatusUpdate);
    connect(m_systemController.get(), &SystemController::errorOccurred,
            this, [this](const QString& error) {
                QMessageBox::critical(this, "系统错误", error);
                qCCritical(mainWindow) << "System error:" << error;
            });
}

void MainWindow::onStartButtonClicked()
{
    qCInfo(mainWindow) << "Start button clicked";
    
    if (m_isRunning) {
        return;
    }
    
    // 启动相机
    if (!m_cameraManager->start()) {
        QMessageBox::warning(this, "警告", "无法启动摄像头，请检查设备连接");
        return;
    }
    
    // 启动系统控制
    m_systemController->startOperation();
    
    switchToOperationMode();
}

void MainWindow::onStopButtonClicked()
{
    qCInfo(mainWindow) << "Stop button clicked";
    
    if (!m_isRunning) {
        return;
    }
    
    stopOperation();
    switchToStandbyMode();
}

void MainWindow::stopOperation()
{
    // 停止相机
    m_cameraManager->stop();
    
    // 停止系统控制
    m_systemController->stopOperation();
    
    m_isRunning = false;
}

void MainWindow::onSettingsButtonClicked()
{
    qCInfo(mainWindow) << "Settings button clicked";
    
    // 通过QML显示设置对话框
    if (m_videoWidget && m_videoWidget->rootObject()) {
        QObject* rootObj = m_videoWidget->rootObject();
        QMetaObject::invokeMethod(rootObj, "showSettingsDialog", Qt::QueuedConnection);
    }
}

void MainWindow::onSystemStatusUpdate()
{
    auto status = m_systemController->status();
    auto info = m_systemController->systemInfo();
    
    QString statusText;
    switch(status) {
    case SystemController::Operating: statusText = "运行中"; break;
    case SystemController::Standby: statusText = "待机"; break;
    case SystemController::Error: statusText = "错误"; break;
    case SystemController::Maintenance: statusText = "维护"; break;
    default: statusText = "离线"; break;
    }
    
    m_statusLabel->setText(QString("系统状态: %1").arg(statusText));
    
    // 更新其他系统信息
    qCDebug(mainWindow) << "System info updated - CPU:" << info.cpuUsage 
                       << "% Memory:" << info.memoryUsage << "% Temp:" << info.temperature << "°C";
}

void MainWindow::onDetectionResult(const BambooDetector::DetectionResult& result)
{
    if (result.isValid) {
        // 更新视频渲染器的检测结果
        m_videoRenderer->setDetectionResult(result.boundingBox, result.confidence);
        
        // 更新状态显示
        int confidence = static_cast<int>(result.confidence * 100);
        m_detectionLabel->setText(QString("检测精度: %1%").arg(confidence));
        
        // 如果是运行模式且检测到竹子，可以触发切割
        if (m_isRunning && result.confidence > 0.8) {
            // 这里可以添加自动切割逻辑
            qCInfo(mainWindow) << "High confidence detection, triggering cut sequence";
        }
    }
}

void MainWindow::onCameraFrameReady()
{
    auto frame = m_cameraManager->getCurrentFrame();
    if (!frame.empty()) {
        // 更新帧计数和帧率
        m_frameCount++;
        qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
        if (m_lastFrameTime > 0) {
            qint64 timeDiff = currentTime - m_lastFrameTime;
            if (timeDiff > 1000) { // 每秒更新一次帧率
                double fps = (m_frameCount * 1000.0) / timeDiff;
                m_frameRateLabel->setText(QString("帧率: %1 FPS").arg(fps, 0, 'f', 1));
                m_frameCount = 0;
                m_lastFrameTime = currentTime;
            }
        } else {
            m_lastFrameTime = currentTime;
        }
        
        // 发送到视频渲染器
        m_videoRenderer->setFrame(frame);
        
        // 发送到AI检测器
        if (m_isRunning) {
            m_bambooDetector->processFrame(frame);
        }
    }
}

void MainWindow::updateStatusBar()
{
    // 定期更新状态栏信息
    onSystemStatusUpdate();
}

void MainWindow::switchToOperationMode()
{
    m_isRunning = true;
    m_startButton->setEnabled(false);
    m_stopButton->setEnabled(true);
    m_settingsButton->setEnabled(false);
    
    m_statusLabel->setText("系统状态: 运行中");
    m_progressBar->setVisible(true);
    
    qCInfo(mainWindow) << "Switched to operation mode";
}

void MainWindow::switchToStandbyMode()
{
    m_isRunning = false;
    m_startButton->setEnabled(true);
    m_stopButton->setEnabled(false);
    m_settingsButton->setEnabled(true);
    
    m_statusLabel->setText("系统状态: 待机");
    m_progressBar->setVisible(false);
    
    qCInfo(mainWindow) << "Switched to standby mode";
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    
    // 通知触摸控制器更新触摸区域
    if (m_touchController && m_videoWidget) {
        m_touchController->setTouchArea(m_videoWidget->geometry());
    }
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    qCInfo(mainWindow) << "Application closing...";
    
    if (m_isRunning) {
        auto reply = QMessageBox::question(this, "确认退出", 
                                         "系统正在运行中，确定要退出吗？",
                                         QMessageBox::Yes | QMessageBox::No);
        if (reply == QMessageBox::No) {
            event->ignore();
            return;
        }
        
        stopOperation();
    }
    
    // 保存配置
    if (m_configManager) {
        m_configManager->save();
    }
    
    event->accept();
}