#include "cameramanager.h"
#include <QtCore/QTimer>
#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>
#include <QtCore/QQueue>
#include <QtCore/QLoggingCategory>
#include <QtCore/QStandardPaths>
#include <QtCore/QProcess>
#include <opencv2/opencv.hpp>

Q_LOGGING_CATEGORY(cameraManager, "app.cameramanager")

// CameraWorker 实现
CameraWorker::CameraWorker(QObject *parent)
    : QObject(parent)
    , m_captureTimer(nullptr)
    , m_running(false)
    , m_initialized(false)
    , m_needsResize(false)
    , m_targetSize(640, 480)
{
    m_captureTimer = new QTimer(this);
    connect(m_captureTimer, &QTimer::timeout, this, &CameraWorker::captureLoop);
}

CameraWorker::~CameraWorker()
{
    if (m_running) {
        stop();
    }
    releaseCamera();
}

void CameraWorker::setConfig(const CameraManager::CameraConfig& config)
{
    QMutexLocker locker(&m_captureMutex);
    m_config = config;
    m_targetSize = cv::Size(config.width, config.height);
    m_needsResize = true;
}

void CameraWorker::start()
{
    qCInfo(cameraManager) << "Starting camera worker...";
    
    if (m_running) {
        qCWarning(cameraManager) << "Camera worker already running";
        return;
    }
    
    if (!initializeCamera()) {
        emit errorOccurred("Failed to initialize camera");
        return;
    }
    
    m_running = true;
    
    // 根据帧率设置定时器间隔
    int interval = 1000 / qMax(1, m_config.fps);
    m_captureTimer->start(interval);
    
    emit started();
    qCInfo(cameraManager) << "Camera worker started successfully";
}

void CameraWorker::stop()
{
    qCInfo(cameraManager) << "Stopping camera worker...";
    
    if (!m_running) {
        return;
    }
    
    m_running = false;
    m_captureTimer->stop();
    
    releaseCamera();
    
    emit stopped();
    qCInfo(cameraManager) << "Camera worker stopped";
}

void CameraWorker::setParameter(const QString& parameter, const QVariant& value)
{
    if (!m_initialized || !m_capture.isOpened()) {
        qCWarning(cameraManager) << "Camera not initialized, cannot set parameter";
        return;
    }
    
    QMutexLocker locker(&m_captureMutex);
    
    // 设置OpenCV参数
    if (parameter == "brightness") {
        m_capture.set(cv::CAP_PROP_BRIGHTNESS, value.toDouble());
    } else if (parameter == "contrast") {
        m_capture.set(cv::CAP_PROP_CONTRAST, value.toDouble());
    } else if (parameter == "exposure") {
        if (value.toInt() >= 0) {
            m_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25); // 手动模式
            m_capture.set(cv::CAP_PROP_EXPOSURE, value.toDouble());
        } else {
            m_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.75); // 自动模式
        }
    } else if (parameter == "gain") {
        m_capture.set(cv::CAP_PROP_GAIN, value.toDouble());
    }
}

bool CameraWorker::initializeCamera()
{
    qCInfo(cameraManager) << "Initializing camera with config:" << m_config.deviceId;
    
    releaseCamera(); // 确保之前的资源已释放
    
    try {
        if (m_config.type == CameraManager::CSI_Camera) {
            // Jetson CSI 摄像头使用 GStreamer 管道
            if (!m_config.pipeline.isEmpty()) {
                qCInfo(cameraManager) << "Using GStreamer pipeline:" << m_config.pipeline;
                m_capture.open(m_config.pipeline.toStdString(), cv::CAP_GSTREAMER);
            } else {
                // 默认 CSI 管道（针对 Jetson 优化）
                QString defaultPipeline = QString(
                    "nvarguscamerasrc sensor_id=%1 ! "
                    "video/x-raw(memory:NVMM), width=%2, height=%3, framerate=%4/1, format=NV12 ! "
                    "nvvidconv ! video/x-raw, format=BGRx ! "
                    "videoconvert ! video/x-raw, format=BGR ! appsink"
                ).arg(m_config.deviceId).arg(m_config.width).arg(m_config.height).arg(m_config.fps);
                
                qCInfo(cameraManager) << "Using default CSI pipeline:" << defaultPipeline;
                m_capture.open(defaultPipeline.toStdString(), cv::CAP_GSTREAMER);
            }
        } else if (m_config.type == CameraManager::USB_Camera) {
            // USB 摄像头
            m_capture.open(m_config.deviceId);
        } else {
            // 其他类型的摄像头
            m_capture.open(m_config.deviceId);
        }
        
        if (!m_capture.isOpened()) {
            qCCritical(cameraManager) << "Failed to open camera device" << m_config.deviceId;
            return false;
        }
        
        // 设置摄像头参数
        if (m_config.type != CameraManager::CSI_Camera) {
            // USB摄像头才需要手动设置这些参数，CSI摄像头通过GStreamer管道设置
            m_capture.set(cv::CAP_PROP_FRAME_WIDTH, m_config.width);
            m_capture.set(cv::CAP_PROP_FRAME_HEIGHT, m_config.height);
            m_capture.set(cv::CAP_PROP_FPS, m_config.fps);
        }
        
        // 设置缓冲区大小
        m_capture.set(cv::CAP_PROP_BUFFERSIZE, m_config.bufferSize);
        
        // 自动曝光和白平衡设置
        if (m_config.enableAutoExposure) {
            m_capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.75);
        }
        
        // 测试读取一帧
        cv::Mat testFrame;
        if (!m_capture.read(testFrame) || testFrame.empty()) {
            qCCritical(cameraManager) << "Failed to read test frame from camera";
            releaseCamera();
            return false;
        }
        
        qCInfo(cameraManager) << "Camera initialized successfully. Frame size:" 
                             << testFrame.cols << "x" << testFrame.rows 
                             << "channels:" << testFrame.channels();
        
        m_initialized = true;
        return true;
        
    } catch (const std::exception& e) {
        qCCritical(cameraManager) << "Exception during camera initialization:" << e.what();
        releaseCamera();
        return false;
    }
}

void CameraWorker::releaseCamera()
{
    QMutexLocker locker(&m_captureMutex);
    
    if (m_capture.isOpened()) {
        m_capture.release();
    }
    
    m_initialized = false;
}

void CameraWorker::captureLoop()
{
    if (!m_running || !m_initialized) {
        return;
    }
    
    try {
        QMutexLocker locker(&m_captureMutex);
        
        cv::Mat frame;
        if (!m_capture.read(frame) || frame.empty()) {
            qCWarning(cameraManager) << "Failed to capture frame";
            return;
        }
        
        // 预处理帧
        cv::Mat processedFrame = preprocessFrame(frame);
        
        if (!processedFrame.empty()) {
            emit frameReady(processedFrame);
        }
        
    } catch (const std::exception& e) {
        qCWarning(cameraManager) << "Exception in capture loop:" << e.what();
        emit errorOccurred(QString("Capture error: %1").arg(e.what()));
    }
}

cv::Mat CameraWorker::preprocessFrame(const cv::Mat& rawFrame)
{
    if (rawFrame.empty()) {
        return cv::Mat();
    }
    
    cv::Mat result = rawFrame;
    
    // 如果需要调整大小
    if (m_needsResize && (rawFrame.cols != m_targetSize.width || rawFrame.rows != m_targetSize.height)) {
        cv::resize(rawFrame, result, m_targetSize, 0, 0, cv::INTER_LINEAR);
    }
    
    // 确保是3通道BGR格式
    if (result.channels() == 1) {
        cv::cvtColor(result, result, cv::COLOR_GRAY2BGR);
    } else if (result.channels() == 4) {
        cv::cvtColor(result, result, cv::COLOR_BGRA2BGR);
    }
    
    return result;
}

// CameraManager 实现
CameraManager::CameraManager(QObject *parent)
    : QObject(parent)
    , m_status(Disconnected)
    , m_workerThread(nullptr)
    , m_worker(nullptr)
    , m_statisticsTimer(nullptr)
    , m_frameCount(0)
    , m_totalFrames(0)
    , m_lastStatisticsTime(0)
    , m_currentFps(0.0)
{
    qCInfo(cameraManager) << "Creating CameraManager...";
    
    // 创建统计定时器
    m_statisticsTimer = new QTimer(this);
    connect(m_statisticsTimer, &QTimer::timeout, this, &CameraManager::onStatisticsTimer);
}

CameraManager::~CameraManager()
{
    qCInfo(cameraManager) << "Destroying CameraManager...";
    release();
}

bool CameraManager::initialize(const CameraConfig& config)
{
    qCInfo(cameraManager) << "Initializing CameraManager...";
    
    if (m_status != Disconnected) {
        qCWarning(cameraManager) << "CameraManager already initialized";
        return false;
    }
    
    m_config = config;
    
    // 创建工作线程
    m_workerThread = new QThread(this);
    m_worker = new CameraWorker();
    m_worker->moveToThread(m_workerThread);
    
    // 连接信号
    connect(m_workerThread, &QThread::started, m_worker, &CameraWorker::start);
    connect(m_worker, &CameraWorker::frameReady, this, &CameraManager::onFrameReady);
    connect(m_worker, &CameraWorker::errorOccurred, this, &CameraManager::onWorkerError);
    connect(m_worker, &CameraWorker::started, this, [this]() {
        m_status = Connected;
        emit statusChanged(m_status);
        qCInfo(cameraManager) << "Camera connected";
    });
    connect(m_worker, &CameraWorker::stopped, this, [this]() {
        m_status = Disconnected;
        emit statusChanged(m_status);
        qCInfo(cameraManager) << "Camera disconnected";
    });
    
    // 设置配置
    m_worker->setConfig(config);
    
    m_status = Connecting;
    emit statusChanged(m_status);
    
    qCInfo(cameraManager) << "CameraManager initialized";
    return true;
}

bool CameraManager::start()
{
    qCInfo(cameraManager) << "Starting camera...";
    
    if (m_status == Streaming) {
        qCWarning(cameraManager) << "Camera already streaming";
        return true;
    }
    
    if (m_status != Connected && m_status != Connecting) {
        qCWarning(cameraManager) << "Camera not connected";
        return false;
    }
    
    // 启动工作线程
    if (m_workerThread && !m_workerThread->isRunning()) {
        m_workerThread->start();
    }
    
    // 重置统计
    resetStatistics();
    m_statisticsTimer->start(1000); // 每秒更新统计
    
    m_status = Streaming;
    emit statusChanged(m_status);
    
    qCInfo(cameraManager) << "Camera streaming started";
    return true;
}

bool CameraManager::stop()
{
    qCInfo(cameraManager) << "Stopping camera...";
    
    if (m_status != Streaming) {
        qCWarning(cameraManager) << "Camera not streaming";
        return false;
    }
    
    // 停止统计定时器
    m_statisticsTimer->stop();
    
    // 停止工作线程
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "stop", Qt::QueuedConnection);
    }
    
    if (m_workerThread && m_workerThread->isRunning()) {
        m_workerThread->quit();
        if (!m_workerThread->wait(3000)) {
            qCWarning(cameraManager) << "Worker thread did not finish within timeout";
            m_workerThread->terminate();
            m_workerThread->wait(1000);
        }
    }
    
    qCInfo(cameraManager) << "Camera stopped";
    return true;
}

void CameraManager::release()
{
    qCInfo(cameraManager) << "Releasing camera resources...";
    
    stop();
    
    // 清理工作线程
    if (m_worker) {
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    
    if (m_workerThread) {
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
    }
    
    // 清理帧缓存
    QMutexLocker locker(&m_frameMutex);
    m_frameBuffer.clear();
    m_currentFrame = cv::Mat();
    
    m_status = Disconnected;
    emit statusChanged(m_status);
    
    qCInfo(cameraManager) << "Camera resources released";
}

cv::Mat CameraManager::getCurrentFrame() const
{
    QMutexLocker locker(&m_frameMutex);
    return m_currentFrame.clone();
}

void CameraManager::setResolution(int width, int height)
{
    m_config.width = width;
    m_config.height = height;
    
    if (m_worker) {
        // 更新工作线程配置
        QMetaObject::invokeMethod(m_worker, [this]() {
            m_worker->setConfig(m_config);
        }, Qt::QueuedConnection);
    }
}

void CameraManager::setFrameRate(int fps)
{
    m_config.fps = fps;
    
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, [this]() {
            m_worker->setConfig(m_config);
        }, Qt::QueuedConnection);
    }
}

void CameraManager::setExposure(int exposure)
{
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "setParameter", Qt::QueuedConnection,
                                Q_ARG(QString, "exposure"), Q_ARG(QVariant, exposure));
    }
}

void CameraManager::setGain(int gain)
{
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "setParameter", Qt::QueuedConnection,
                                Q_ARG(QString, "gain"), Q_ARG(QVariant, gain));
    }
}

void CameraManager::setBrightness(int brightness)
{
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "setParameter", Qt::QueuedConnection,
                                Q_ARG(QString, "brightness"), Q_ARG(QVariant, brightness));
    }
}

void CameraManager::setContrast(int contrast)
{
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "setParameter", Qt::QueuedConnection,
                                Q_ARG(QString, "contrast"), Q_ARG(QVariant, contrast));
    }
}

void CameraManager::onFrameReady(const cv::Mat& frame)
{
    if (frame.empty()) {
        return;
    }
    
    // 更新帧缓存
    {
        QMutexLocker locker(&m_frameMutex);
        
        // 保存当前帧
        m_currentFrame = frame.clone();
        
        // 管理帧缓冲区
        if (m_frameBuffer.size() >= MAX_BUFFER_SIZE) {
            m_frameBuffer.dequeue();
        }
        m_frameBuffer.enqueue(frame.clone());
    }
    
    // 更新统计
    m_frameCount++;
    m_totalFrames++;
    
    // 发射信号
    emit frameReady(frame);
}

void CameraManager::onWorkerError(const QString& error)
{
    qCWarning(cameraManager) << "Camera worker error:" << error;
    
    m_status = Error;
    emit statusChanged(m_status);
    emit errorOccurred(error);
}

void CameraManager::onStatisticsTimer()
{
    updateStatistics();
}

void CameraManager::updateStatistics()
{
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
    
    if (m_lastStatisticsTime > 0) {
        qint64 timeDiff = currentTime - m_lastStatisticsTime;
        if (timeDiff > 0) {
            m_currentFps = (m_frameCount * 1000.0) / timeDiff;
            emit statisticsUpdated(m_currentFps, m_totalFrames);
        }
    }
    
    // 重置帧计数器
    m_frameCount = 0;
    m_lastStatisticsTime = currentTime;
}

void CameraManager::resetStatistics()
{
    m_frameCount = 0;
    m_totalFrames = 0;
    m_lastStatisticsTime = 0;
    m_currentFps = 0.0;
}

QString CameraManager::buildGStreamerPipeline() const
{
    // 为不同类型的摄像头构建 GStreamer 管道
    QString pipeline;
    
    switch (m_config.type) {
    case CSI_Camera:
        pipeline = QString(
            "nvarguscamerasrc sensor_id=%1 ! "
            "video/x-raw(memory:NVMM), width=%2, height=%3, framerate=%4/1, format=NV12 ! "
            "nvvidconv ! video/x-raw, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        ).arg(m_config.deviceId).arg(m_config.width).arg(m_config.height).arg(m_config.fps);
        break;
        
    case USB_Camera:
        pipeline = QString(
            "v4l2src device=/dev/video%1 ! "
            "video/x-raw, width=%2, height=%3, framerate=%4/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink"
        ).arg(m_config.deviceId).arg(m_config.width).arg(m_config.height).arg(m_config.fps);
        break;
        
    case IP_Camera:
        // 这里可以添加网络摄像头的管道配置
        break;
        
    default:
        break;
    }
    
    return pipeline;
}