#ifndef BAMBOODETECTOR_H
#define BAMBOODETECTOR_H

#include <QtCore/QObject>
#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QTimer>
#include <QtCore/QRect>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <memory>

class DetectionWorker;

class BambooDetector : public QObject
{
    Q_OBJECT

public:
    struct DetectionResult {
        QRect boundingBox;
        float confidence;
        QString className;
        cv::Point2f center;
        float area;
        bool isValid;
    };

    struct ModelConfig {
        QString modelPath;
        QString configPath;
        QString classNamesPath;
        float confidenceThreshold;
        float nmsThreshold;
        cv::Size inputSize;
        bool useGPU;
        bool useTensorRT;
        bool useINT8;
        int batchSize;
    };

    enum DetectionStatus {
        Uninitialized,
        Loading,
        Ready,
        Processing,
        Error
    };

    explicit BambooDetector(QObject *parent = nullptr);
    ~BambooDetector();

    bool initialize(const ModelConfig& config);
    bool loadModel(const QString& modelPath);
    void processFrame(const cv::Mat& frame);
    
    DetectionStatus status() const { return m_status; }
    ModelConfig config() const { return m_config; }
    DetectionResult lastResult() const { return m_lastResult; }

    void setConfidenceThreshold(float threshold);
    void setNMSThreshold(float threshold);
    void enableGPU(bool enable);
    void enableTensorRT(bool enable);

signals:
    void detectionReady(const DetectionResult& result);
    void statusChanged(DetectionStatus status);
    void errorOccurred(const QString& error);
    void processingTimeUpdated(double ms);

private slots:
    void onDetectionReady(const DetectionResult& result);
    void onWorkerError(const QString& error);

private:
    void resetDetection();
    bool validateModelConfig(const ModelConfig& config);

    ModelConfig m_config;
    DetectionStatus m_status;
    DetectionResult m_lastResult;

    // 工作线程
    QThread *m_workerThread;
    DetectionWorker *m_worker;

    // 统计信息
    mutable QMutex m_resultMutex;
    double m_averageProcessingTime;
    int m_processedFrames;
};

// 检测工作线程类
class DetectionWorker : public QObject
{
    Q_OBJECT

public:
    explicit DetectionWorker(QObject *parent = nullptr);
    ~DetectionWorker();

    void setConfig(const BambooDetector::ModelConfig& config);

public slots:
    void initialize();
    void processFrame(const cv::Mat& frame);
    void updateThreshold(float confidence, float nms);

signals:
    void detectionReady(const BambooDetector::DetectionResult& result);
    void errorOccurred(const QString& error);
    void initialized();

private:
    bool loadYOLOModel();
    bool loadTensorRTModel();
    void preprocessFrame(const cv::Mat& input, cv::Mat& output);
    std::vector<BambooDetector::DetectionResult> postprocessDetections(
        const std::vector<cv::Mat>& outputs, const cv::Size& frameSize);
    void applyNMS(std::vector<BambooDetector::DetectionResult>& detections);
    
    BambooDetector::ModelConfig m_config;
    cv::dnn::Net m_net;
    std::vector<std::string> m_classNames;
    std::vector<std::string> m_outputNames;
    
    bool m_initialized;
    mutable QMutex m_processingMutex;
    
    // 预处理缓冲区
    cv::Mat m_blob;
    cv::Mat m_resizedFrame;
    
    // TensorRT优化
    void* m_tensorRTEngine;
    void* m_tensorRTContext;
    bool m_useTensorRT;
    
    // 性能统计
    qint64 m_lastProcessTime;
};

Q_DECLARE_METATYPE(BambooDetector::DetectionResult)

#endif // BAMBOODETECTOR_H