#ifndef VIDEORENDERER_H
#define VIDEORENDERER_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLTexture>
#include <QtGui/QMatrix4x4>
#include <QtCore/QTimer>
#include <opencv2/opencv.hpp>

class VideoRenderer : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit VideoRenderer(QWidget *parent = nullptr);
    ~VideoRenderer();

    void setFrame(const cv::Mat& frame);
    void setDetectionResult(const QRect& bambooRect, float confidence);
    void setZoomFactor(float zoom);
    void setPanOffset(const QPointF& offset);

protected:
    void initializeGL() override;
    void resizeGL(int width, int height) override;
    void paintGL() override;
    
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void wheelEvent(QWheelEvent *event) override;

signals:
    void frameRendered();
    void touchPoint(const QPointF& point);
    void zoomChanged(float zoom);

private slots:
    void onRenderTimer();

private:
    void setupShaders();
    void setupBuffers();
    void updateTexture(const cv::Mat& frame);
    void renderFrame();
    void renderDetectionOverlay();
    void calculateFPS();

    // OpenGL资源
    QOpenGLShaderProgram *m_shaderProgram;
    QOpenGLBuffer m_vertexBuffer;
    QOpenGLBuffer m_indexBuffer;
    QOpenGLVertexArrayObject m_vao;
    QOpenGLTexture *m_frameTexture;
    
    // 着色器属性位置
    int m_positionAttribute;
    int m_texCoordAttribute;
    int m_mvpMatrixUniform;
    int m_textureUniform;
    
    // 变换矩阵
    QMatrix4x4 m_projectionMatrix;
    QMatrix4x4 m_viewMatrix;
    QMatrix4x4 m_modelMatrix;
    
    // 视频帧数据
    cv::Mat m_currentFrame;
    bool m_frameUpdated;
    
    // 检测结果
    QRect m_detectionRect;
    float m_detectionConfidence;
    bool m_hasDetection;
    
    // 交互控制
    float m_zoomFactor;
    QPointF m_panOffset;
    QPoint m_lastMousePos;
    bool m_isPanning;
    
    // 渲染控制
    QTimer *m_renderTimer;
    int m_frameCount;
    qint64 m_lastFpsTime;
    float m_currentFps;
    
    // 纹理格式
    bool m_useYUV;
    GLuint m_textureIds[3]; // Y, U, V分量纹理
};

#endif // VIDEORENDERER_H