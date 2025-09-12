#include "bamboodetector.h"
#include <QtCore/QThread>
#include <QtCore/QMutex>
#include <QtCore/QTimer>
#include <QtCore/QLoggingCategory>
#include <QtCore/QStandardPaths>
#include <QtCore/QFileInfo>
#include <QtCore/QDir>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <fstream>

Q_LOGGING_CATEGORY(bambooDetector, "app.bamboodetector")

// DetectionWorker 实现
DetectionWorker::DetectionWorker(QObject *parent)
    : QObject(parent)
    , m_initialized(false)
    , m_tensorRTEngine(nullptr)
    , m_tensorRTContext(nullptr)
    , m_useTensorRT(false)
    , m_lastProcessTime(0)
{
}

DetectionWorker::~DetectionWorker()
{
    QMutexLocker locker(&m_processingMutex);
    
    // 清理TensorRT资源
    if (m_tensorRTContext) {
        // TensorRT清理代码将在实际部署时添加
    }
    
    if (m_tensorRTEngine) {
        // TensorRT清理代码将在实际部署时添加
    }
}

void DetectionWorker::setConfig(const BambooDetector::ModelConfig& config)
{
    QMutexLocker locker(&m_processingMutex);
    m_config = config;
}

void DetectionWorker::initialize()
{
    qCInfo(bambooDetector) << "Initializing detection worker...";
    
    if (m_initialized) {
        qCWarning(bambooDetector) << "Detection worker already initialized";
        return;
    }
    
    try {
        // 检查模型文件是否存在
        QFileInfo modelFile(m_config.modelPath);
        if (!modelFile.exists()) {
            emit errorOccurred(QString("Model file not found: %1").arg(m_config.modelPath));
            return;
        }
        
        // 根据配置加载模型
        bool success = false;
        if (m_config.useTensorRT && m_config.useGPU) {
            qCInfo(bambooDetector) << "Attempting to load TensorRT model...";
            success = loadTensorRTModel();
            if (!success) {
                qCWarning(bambooDetector) << "TensorRT model loading failed, falling back to YOLO";
            }
        }
        
        if (!success) {
            qCInfo(bambooDetector) << "Loading YOLO model...";
            success = loadYOLOModel();
        }
        
        if (!success) {
            emit errorOccurred("Failed to load any detection model");
            return;
        }
        
        // 加载类别名称
        if (!m_config.classNamesPath.isEmpty()) {
            QFileInfo classFile(m_config.classNamesPath);
            if (classFile.exists()) {
                std::ifstream ifs(m_config.classNamesPath.toStdString());
                std::string line;
                while (std::getline(ifs, line)) {
                    m_classNames.push_back(line);
                }
                qCInfo(bambooDetector) << "Loaded" << m_classNames.size() << "class names";
            }
        }
        
        if (m_classNames.empty()) {
            // 默认竹子检测类别
            m_classNames = {"bamboo", "竹子"};
            qCInfo(bambooDetector) << "Using default class names";
        }
        
        m_initialized = true;
        emit initialized();
        qCInfo(bambooDetector) << "Detection worker initialized successfully";
        
    } catch (const std::exception& e) {
        emit errorOccurred(QString("Detection initialization error: %1").arg(e.what()));
    }
}

bool DetectionWorker::loadYOLOModel()
{
    qCInfo(bambooDetector) << "Loading YOLO model from:" << m_config.modelPath;
    
    try {
        // 加载YOLO模型
        m_net = cv::dnn::readNet(m_config.modelPath.toStdString());
        
        if (m_net.empty()) {
            qCCritical(bambooDetector) << "Failed to load YOLO model";
            return false;
        }
        
        // 设置后端和目标设备
        if (m_config.useGPU) {
            qCInfo(bambooDetector) << "Setting CUDA backend for GPU acceleration";
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } else {
            qCInfo(bambooDetector) << "Using CPU backend";
            m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
        
        // 获取输出层名称
        m_outputNames = m_net.getUnconnectedOutLayersNames();
        
        qCInfo(bambooDetector) << "YOLO model loaded successfully with" << m_outputNames.size() << "output layers";
        return true;
        
    } catch (const std::exception& e) {
        qCCritical(bambooDetector) << "Exception loading YOLO model:" << e.what();
        return false;
    }
}

bool DetectionWorker::loadTensorRTModel()
{
    qCInfo(bambooDetector) << "TensorRT model loading not yet implemented";
    
    // TensorRT模型加载将在实际部署时实现
    // 这里预留接口，实际实现需要：
    // 1. 创建TensorRT引擎
    // 2. 分配GPU内存
    // 3. 创建执行上下文
    
    m_useTensorRT = false;
    return false;
}

void DetectionWorker::processFrame(const cv::Mat& frame)
{
    if (!m_initialized || frame.empty()) {
        return;
    }
    
    QMutexLocker locker(&m_processingMutex);
    
    qint64 startTime = QDateTime::currentMSecsSinceEpoch();
    
    try {
        cv::Mat processedFrame;
        
        // 预处理
        preprocessFrame(frame, processedFrame);
        
        // 推理
        std::vector<cv::Mat> outputs;
        m_net.setInput(processedFrame);
        m_net.forward(outputs, m_outputNames);
        
        // 后处理
        std::vector<BambooDetector::DetectionResult> detections = 
            postprocessDetections(outputs, frame.size());
        
        // 应用NMS
        applyNMS(detections);
        
        // 返回最佳检测结果
        BambooDetector::DetectionResult bestResult;
        bestResult.isValid = false;
        
        if (!detections.empty()) {
            // 选择置信度最高的检测结果
            auto bestIt = std::max_element(detections.begin(), detections.end(),
                [](const BambooDetector::DetectionResult& a, const BambooDetector::DetectionResult& b) {
                    return a.confidence < b.confidence;
                });
            
            if (bestIt != detections.end() && bestIt->confidence >= m_config.confidenceThreshold) {
                bestResult = *bestIt;
                bestResult.isValid = true;
            }
        }
        
        // 记录处理时间
        m_lastProcessTime = QDateTime::currentMSecsSinceEpoch() - startTime;
        
        emit detectionReady(bestResult);
        
    } catch (const std::exception& e) {
        qCWarning(bambooDetector) << "Detection processing error:" << e.what();
        emit errorOccurred(QString("Detection error: %1").arg(e.what()));
    }
}

void DetectionWorker::updateThreshold(float confidence, float nms)
{
    QMutexLocker locker(&m_processingMutex);
    m_config.confidenceThreshold = confidence;
    m_config.nmsThreshold = nms;
}

void DetectionWorker::preprocessFrame(const cv::Mat& input, cv::Mat& output)
{
    // YOLO预处理：调整大小、归一化、转换为blob
    cv::Mat resized;
    
    // 调整到模型输入尺寸，保持宽高比
    cv::resize(input, resized, m_config.inputSize);
    
    // 转换为blob格式 (1, 3, height, width)
    cv::dnn::blobFromImage(resized, output, 1.0/255.0, m_config.inputSize, 
                          cv::Scalar(0, 0, 0), true, false, CV_32F);
}

std::vector<BambooDetector::DetectionResult> DetectionWorker::postprocessDetections(
    const std::vector<cv::Mat>& outputs, const cv::Size& frameSize)
{
    std::vector<BambooDetector::DetectionResult> detections;
    
    for (const auto& output : outputs) {
        // YOLO输出格式：[batch, num_detections, 85] (4个坐标 + 1个置信度 + 80个类别)
        int numDetections = output.size[1];
        int numElements = output.size[2];
        
        for (int i = 0; i < numDetections; i++) {
            const float* data = output.ptr<float>(0, i);
            
            float confidence = data[4];
            if (confidence < m_config.confidenceThreshold) {
                continue;
            }
            
            // 获取类别置信度
            float maxClassScore = 0;
            int classId = -1;
            for (int j = 5; j < numElements; j++) {
                if (data[j] > maxClassScore) {
                    maxClassScore = data[j];
                    classId = j - 5;
                }
            }
            
            float finalConfidence = confidence * maxClassScore;
            if (finalConfidence < m_config.confidenceThreshold) {
                continue;
            }
            
            // 解析边界框（YOLO格式：center_x, center_y, width, height）
            float centerX = data[0];
            float centerY = data[1];
            float width = data[2];
            float height = data[3];
            
            // 转换为像素坐标
            float scaleX = static_cast<float>(frameSize.width) / m_config.inputSize.width;
            float scaleY = static_cast<float>(frameSize.height) / m_config.inputSize.height;
            
            float x = (centerX - width / 2) * scaleX;
            float y = (centerY - height / 2) * scaleY;
            float w = width * scaleX;
            float h = height * scaleY;
            
            // 创建检测结果
            BambooDetector::DetectionResult detection;
            detection.boundingBox = QRect(static_cast<int>(x), static_cast<int>(y), 
                                        static_cast<int>(w), static_cast<int>(h));
            detection.confidence = finalConfidence;
            detection.center = cv::Point2f(centerX * scaleX, centerY * scaleY);
            detection.area = w * h;
            detection.isValid = true;
            
            if (classId >= 0 && classId < static_cast<int>(m_classNames.size())) {
                detection.className = QString::fromStdString(m_classNames[classId]);
            } else {
                detection.className = QString("object_%1").arg(classId);
            }
            
            detections.push_back(detection);
        }
    }
    
    return detections;
}

void DetectionWorker::applyNMS(std::vector<BambooDetector::DetectionResult>& detections)
{
    if (detections.size() <= 1) {
        return;
    }
    
    // 构建OpenCV所需的向量
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> indices;
    
    for (const auto& detection : detections) {
        boxes.push_back(cv::Rect(detection.boundingBox.x(), detection.boundingBox.y(),
                                detection.boundingBox.width(), detection.boundingBox.height()));
        confidences.push_back(detection.confidence);
    }
    
    // 应用NMS
    cv::dnn::NMSBoxes(boxes, confidences, m_config.confidenceThreshold, 
                     m_config.nmsThreshold, indices);
    
    // 保留NMS后的检测结果
    std::vector<BambooDetector::DetectionResult> nmsDetections;
    for (int idx : indices) {
        if (idx >= 0 && idx < static_cast<int>(detections.size())) {
            nmsDetections.push_back(detections[idx]);
        }
    }
    
    detections = std::move(nmsDetections);
}

// BambooDetector 实现
BambooDetector::BambooDetector(QObject *parent)
    : QObject(parent)
    , m_status(Uninitialized)
    , m_workerThread(nullptr)
    , m_worker(nullptr)
    , m_averageProcessingTime(0.0)
    , m_processedFrames(0)
{
    qCInfo(bambooDetector) << "Creating BambooDetector...";
}

BambooDetector::~BambooDetector()
{
    qCInfo(bambooDetector) << "Destroying BambooDetector...";
    
    if (m_worker) {
        m_worker->deleteLater();
        m_worker = nullptr;
    }
    
    if (m_workerThread) {
        m_workerThread->quit();
        if (!m_workerThread->wait(3000)) {
            m_workerThread->terminate();
            m_workerThread->wait(1000);
        }
        m_workerThread->deleteLater();
        m_workerThread = nullptr;
    }
}

bool BambooDetector::initialize(const ModelConfig& config)
{
    qCInfo(bambooDetector) << "Initializing BambooDetector with model:" << config.modelPath;
    
    if (m_status != Uninitialized) {
        qCWarning(bambooDetector) << "BambooDetector already initialized";
        return false;
    }
    
    if (!validateModelConfig(config)) {
        return false;
    }
    
    m_config = config;
    m_status = Loading;
    emit statusChanged(m_status);
    
    // 创建工作线程
    m_workerThread = new QThread(this);
    m_worker = new DetectionWorker();
    m_worker->moveToThread(m_workerThread);
    
    // 连接信号
    connect(m_workerThread, &QThread::started, m_worker, &DetectionWorker::initialize);
    connect(m_worker, &DetectionWorker::detectionReady, this, &BambooDetector::onDetectionReady);
    connect(m_worker, &DetectionWorker::errorOccurred, this, &BambooDetector::onWorkerError);
    connect(m_worker, &DetectionWorker::initialized, this, [this]() {
        m_status = Ready;
        emit statusChanged(m_status);
        qCInfo(bambooDetector) << "BambooDetector ready";
    });
    
    // 设置配置
    m_worker->setConfig(config);
    
    // 启动工作线程
    m_workerThread->start();
    
    return true;
}

bool BambooDetector::loadModel(const QString& modelPath)
{
    ModelConfig newConfig = m_config;
    newConfig.modelPath = modelPath;
    
    return initialize(newConfig);
}

void BambooDetector::processFrame(const cv::Mat& frame)
{
    if (m_status != Ready || !m_worker) {
        return;
    }
    
    m_status = Processing;
    emit statusChanged(m_status);
    
    // 异步处理帧
    QMetaObject::invokeMethod(m_worker, "processFrame", Qt::QueuedConnection,
                            Q_ARG(cv::Mat, frame));
}

void BambooDetector::setConfidenceThreshold(float threshold)
{
    m_config.confidenceThreshold = qBound(0.0f, threshold, 1.0f);
    
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "updateThreshold", Qt::QueuedConnection,
                                Q_ARG(float, m_config.confidenceThreshold),
                                Q_ARG(float, m_config.nmsThreshold));
    }
}

void BambooDetector::setNMSThreshold(float threshold)
{
    m_config.nmsThreshold = qBound(0.0f, threshold, 1.0f);
    
    if (m_worker) {
        QMetaObject::invokeMethod(m_worker, "updateThreshold", Qt::QueuedConnection,
                                Q_ARG(float, m_config.confidenceThreshold),
                                Q_ARG(float, m_config.nmsThreshold));
    }
}

void BambooDetector::enableGPU(bool enable)
{
    m_config.useGPU = enable;
    // GPU设置更改需要重新初始化模型
    qCInfo(bambooDetector) << "GPU setting changed, model needs reinitialization";
}

void BambooDetector::enableTensorRT(bool enable)
{
    m_config.useTensorRT = enable;
    // TensorRT设置更改需要重新初始化模型
    qCInfo(bambooDetector) << "TensorRT setting changed, model needs reinitialization";
}

void BambooDetector::onDetectionReady(const DetectionResult& result)
{
    QMutexLocker locker(&m_resultMutex);
    
    m_lastResult = result;
    m_processedFrames++;
    
    // 更新处理时间统计
    if (m_processedFrames > 0) {
        m_averageProcessingTime = (m_averageProcessingTime * (m_processedFrames - 1) + 
                                 m_worker->m_lastProcessTime) / m_processedFrames;
        emit processingTimeUpdated(m_averageProcessingTime);
    }
    
    m_status = Ready;
    emit statusChanged(m_status);
    emit detectionReady(result);
}

void BambooDetector::onWorkerError(const QString& error)
{
    qCWarning(bambooDetector) << "Detection worker error:" << error;
    
    m_status = Error;
    emit statusChanged(m_status);
    emit errorOccurred(error);
}

bool BambooDetector::validateModelConfig(const ModelConfig& config)
{
    // 检查模型文件
    QFileInfo modelFile(config.modelPath);
    if (!modelFile.exists()) {
        emit errorOccurred(QString("Model file does not exist: %1").arg(config.modelPath));
        return false;
    }
    
    // 检查文件格式
    QString suffix = modelFile.suffix().toLower();
    if (suffix != "pt" && suffix != "onnx" && suffix != "trt" && suffix != "engine") {
        emit errorOccurred(QString("Unsupported model format: %1").arg(suffix));
        return false;
    }
    
    // 检查参数范围
    if (config.confidenceThreshold < 0.0f || config.confidenceThreshold > 1.0f) {
        emit errorOccurred("Confidence threshold must be between 0.0 and 1.0");
        return false;
    }
    
    if (config.nmsThreshold < 0.0f || config.nmsThreshold > 1.0f) {
        emit errorOccurred("NMS threshold must be between 0.0 and 1.0");
        return false;
    }
    
    if (config.inputSize.width <= 0 || config.inputSize.height <= 0) {
        emit errorOccurred("Invalid input size");
        return false;
    }
    
    return true;
}

void BambooDetector::resetDetection()
{
    QMutexLocker locker(&m_resultMutex);
    m_lastResult = DetectionResult();
    m_lastResult.isValid = false;
}