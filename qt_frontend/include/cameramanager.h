#ifndef CAMERAMANAGER_H
#define CAMERAMANAGER_H

#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QWaitCondition>
#include <QtCore/QQueue>
#include <opencv2/opencv.hpp>
#include <memory>

class CameraWorker;

class CameraManager : public QObject
{
    Q_OBJECT

public:
    enum CameraType {
        CSI_Camera,     // Jetson CSI摄像头
        USB_Camera,     // USB摄像头
        IP_Camera,      // 网络摄像头
        Stereo_Camera   // 双目摄像头
    };

    enum CameraStatus {
        Disconnected,
        Connecting,
        Connected,
        Streaming,
        Error
    };

    struct CameraConfig {
        int deviceId;
        int width;
        int height;
        int fps;
        CameraType type;
        QString pipeline;  // GStreamer管道配置
        bool useHardwareAcceleration;
        bool enableAutoExposure;
        bool enableAutoWhiteBalance;
        int bufferSize;
    };

    explicit CameraManager(QObject *parent = nullptr);
    ~CameraManager();

    bool initialize(const CameraConfig& config);
    bool start();
    bool stop();
    void release();

    CameraStatus status() const { return m_status; }
    CameraConfig config() const { return m_config; }
    cv::Mat getCurrentFrame() const;
    
    void setResolution(int width, int height);
    void setFrameRate(int fps);
    void setExposure(int exposure);
    void setGain(int gain);
    void setBrightness(int brightness);
    void setContrast(int contrast);

signals:
    void frameReady(const cv::Mat& frame);
    void statusChanged(CameraStatus status);
    void errorOccurred(const QString& error);
    void statisticsUpdated(double fps, int frameCount);

private slots:
    void onFrameReady(const cv::Mat& frame);
    void onWorkerError(const QString& error);
    void onStatisticsTimer();

private:
    QString buildGStreamerPipeline() const;
    void updateStatistics();
    void resetStatistics();

    CameraConfig m_config;
    CameraStatus m_status;
    
    // 工作线程
    QThread *m_workerThread;
    CameraWorker *m_worker;
    
    // 帧缓存
    mutable QMutex m_frameMutex;
    cv::Mat m_currentFrame;
    QQueue<cv::Mat> m_frameBuffer;
    
    // 统计信息
    QTimer *m_statisticsTimer;
    int m_frameCount;
    int m_totalFrames;
    qint64 m_lastStatisticsTime;
    double m_currentFps;
    
    static const int MAX_BUFFER_SIZE = 5;
};

// 相机工作线程类
class CameraWorker : public QObject
{
    Q_OBJECT

public:
    explicit CameraWorker(QObject *parent = nullptr);
    ~CameraWorker();

    void setConfig(const CameraManager::CameraConfig& config);

public slots:
    void start();
    void stop();
    void setParameter(const QString& parameter, const QVariant& value);

signals:
    void frameReady(const cv::Mat& frame);
    void errorOccurred(const QString& error);
    void started();
    void stopped();

private slots:
    void captureLoop();

private:
    bool initializeCamera();
    void releaseCamera();
    cv::Mat preprocessFrame(const cv::Mat& rawFrame);

    CameraManager::CameraConfig m_config;
    cv::VideoCapture m_capture;
    QTimer *m_captureTimer;
    
    bool m_running;
    bool m_initialized;
    mutable QMutex m_captureMutex;
    
    // 帧预处理
    cv::Mat m_processingBuffer;
    bool m_needsResize;
    cv::Size m_targetSize;
};

#endif // CAMERAMANAGER_H