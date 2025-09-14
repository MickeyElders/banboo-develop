#include "videorenderer.h"
#include <QtOpenGL/QOpenGLShaderProgram>
#include <QtOpenGL/QOpenGLBuffer>
#include <QtOpenGL/QOpenGLVertexArrayObject>
#include <QtOpenGL/QOpenGLTexture>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QMatrix4x4>
#include <QtCore/QTimer>
#include <QtCore/QLoggingCategory>
#include <QtCore/QDateTime>
#include <QtGui/QMouseEvent>
#include <QtGui/QWheelEvent>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <opencv2/opencv.hpp>

Q_LOGGING_CATEGORY(videoRenderer, "app.videorenderer")

// 顶点数据 (位置 + 纹理坐标)
static const float vertices[] = {
    // 位置        // 纹理坐标
    -1.0f, -1.0f,  0.0f, 1.0f,  // 左下
     1.0f, -1.0f,  1.0f, 1.0f,  // 右下
     1.0f,  1.0f,  1.0f, 0.0f,  // 右上
    -1.0f,  1.0f,  0.0f, 0.0f   // 左上
};

static const unsigned int indices[] = {
    0, 1, 2,  // 第一个三角形
    2, 3, 0   // 第二个三角形
};

VideoRenderer::VideoRenderer(QWidget *parent)
    : QOpenGLWidget(parent)
    , m_shaderProgram(nullptr)
    , m_frameTexture(nullptr)
    , m_positionAttribute(-1)
    , m_texCoordAttribute(-1)
    , m_mvpMatrixUniform(-1)
    , m_textureUniform(-1)
    , m_frameUpdated(false)
    , m_detectionConfidence(0.0f)
    , m_hasDetection(false)
    , m_zoomFactor(1.0f)
    , m_panOffset(0.0f, 0.0f)
    , m_isPanning(false)
    , m_renderTimer(nullptr)
    , m_frameCount(0)
    , m_lastFpsTime(0)
    , m_currentFps(0.0f)
    , m_useYUV(false)
{
    qCInfo(videoRenderer) << "Initializing VideoRenderer...";
    
    // 设置OpenGL格式
    QSurfaceFormat format;
    format.setRenderableType(QSurfaceFormat::OpenGLES);
    format.setVersion(2, 0);
    format.setProfile(QSurfaceFormat::NoProfile);
    format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    format.setSamples(4); // 4x MSAA
    setFormat(format);
    
    // 启用鼠标追踪
    setMouseTracking(true);
    
    // 初始化纹理ID
    memset(m_textureIds, 0, sizeof(m_textureIds));
    
    // 创建渲染定时器
    m_renderTimer = new QTimer(this);
    connect(m_renderTimer, &QTimer::timeout, this, &VideoRenderer::onRenderTimer);
    m_renderTimer->start(16); // ~60 FPS
    
    qCInfo(videoRenderer) << "VideoRenderer initialized";
}

VideoRenderer::~VideoRenderer()
{
    qCInfo(videoRenderer) << "Destroying VideoRenderer...";
    
    makeCurrent();
    
    // 清理OpenGL资源
    if (m_shaderProgram) {
        delete m_shaderProgram;
    }
    
    if (m_frameTexture) {
        delete m_frameTexture;
    }
    
    if (m_textureIds[0]) {
        glDeleteTextures(3, m_textureIds);
    }
    
    doneCurrent();
}

void VideoRenderer::initializeGL()
{
    qCInfo(videoRenderer) << "Initializing OpenGL...";
    
    initializeOpenGLFunctions();
    
    // 设置背景色
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    
    // 启用深度测试和混合
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // 设置着色器和缓冲区
    setupShaders();
    setupBuffers();
    
    // 初始化变换矩阵
    m_projectionMatrix.setToIdentity();
    m_viewMatrix.setToIdentity();
    m_modelMatrix.setToIdentity();
    
    qCInfo(videoRenderer) << "OpenGL initialized successfully";
    qCInfo(videoRenderer) << "OpenGL Version:" << (char*)glGetString(GL_VERSION);
    qCInfo(videoRenderer) << "GLSL Version:" << (char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
}

void VideoRenderer::resizeGL(int width, int height)
{
    qCDebug(videoRenderer) << "Resizing to" << width << "x" << height;
    
    glViewport(0, 0, width, height);
    
    // 更新投影矩阵
    m_projectionMatrix.setToIdentity();
    float aspectRatio = static_cast<float>(width) / static_cast<float>(height);
    m_projectionMatrix.ortho(-aspectRatio, aspectRatio, -1.0f, 1.0f, -1.0f, 1.0f);
}

void VideoRenderer::paintGL()
{
    // 清除颜色和深度缓冲区
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    if (!m_shaderProgram || m_currentFrame.empty()) {
        return;
    }
    
    // 更新纹理（如果有新帧）
    if (m_frameUpdated) {
        updateTexture(m_currentFrame);
        m_frameUpdated = false;
    }
    
    // 渲染视频帧
    renderFrame();
    
    // 渲染检测结果覆盖层
    if (m_hasDetection) {
        renderDetectionOverlay();
    }
    
    // 更新FPS计算
    calculateFPS();
    
    emit frameRendered();
}

void VideoRenderer::setupShaders()
{
    qCInfo(videoRenderer) << "Setting up shaders...";
    
    m_shaderProgram = new QOpenGLShaderProgram(this);
    
    // 顶点着色器
    QString vertexShaderSource = R"(
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat4 uMVPMatrix;
        varying vec2 vTexCoord;
        
        void main() {
            gl_Position = uMVPMatrix * aPosition;
            vTexCoord = aTexCoord;
        }
    )";
    
    // 片段着色器（BGR格式）
    QString fragmentShaderSource = R"(
        precision mediump float;
        varying vec2 vTexCoord;
        uniform sampler2D uTexture;
        
        void main() {
            vec4 color = texture2D(uTexture, vTexCoord);
            // BGR to RGB conversion
            gl_FragColor = vec4(color.b, color.g, color.r, color.a);
        }
    )";
    
    // 编译着色器
    if (!m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource)) {
        qCCritical(videoRenderer) << "Failed to compile vertex shader:" << m_shaderProgram->log();
        return;
    }
    
    if (!m_shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource)) {
        qCCritical(videoRenderer) << "Failed to compile fragment shader:" << m_shaderProgram->log();
        return;
    }
    
    // 链接着色器程序
    if (!m_shaderProgram->link()) {
        qCCritical(videoRenderer) << "Failed to link shader program:" << m_shaderProgram->log();
        return;
    }
    
    // 获取属性和uniform位置
    m_positionAttribute = m_shaderProgram->attributeLocation("aPosition");
    m_texCoordAttribute = m_shaderProgram->attributeLocation("aTexCoord");
    m_mvpMatrixUniform = m_shaderProgram->uniformLocation("uMVPMatrix");
    m_textureUniform = m_shaderProgram->uniformLocation("uTexture");
    
    qCInfo(videoRenderer) << "Shaders compiled and linked successfully";
}

void VideoRenderer::setupBuffers()
{
    qCInfo(videoRenderer) << "Setting up buffers...";
    
    // 创建VAO
    if (!m_vao.create()) {
        qCCritical(videoRenderer) << "Failed to create VAO";
        return;
    }
    m_vao.bind();
    
    // 创建顶点缓冲区
    if (!m_vertexBuffer.create()) {
        qCCritical(videoRenderer) << "Failed to create vertex buffer";
        return;
    }
    m_vertexBuffer.bind();
    m_vertexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vertexBuffer.allocate(vertices, sizeof(vertices));
    
    // 创建索引缓冲区
    if (!m_indexBuffer.create()) {
        qCCritical(videoRenderer) << "Failed to create index buffer";
        return;
    }
    m_indexBuffer.bind();
    m_indexBuffer.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_indexBuffer.allocate(indices, sizeof(indices));
    
    // 设置顶点属性
    glEnableVertexAttribArray(m_positionAttribute);
    glVertexAttribPointer(m_positionAttribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);
    
    glEnableVertexAttribArray(m_texCoordAttribute);
    glVertexAttribPointer(m_texCoordAttribute, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 
                         reinterpret_cast<void*>(2 * sizeof(float)));
    
    m_vao.release();
    
    qCInfo(videoRenderer) << "Buffers set up successfully";
}

void VideoRenderer::updateTexture(const cv::Mat& frame)
{
    if (frame.empty()) {
        return;
    }
    
    // 创建或更新纹理
    if (!m_frameTexture) {
        m_frameTexture = new QOpenGLTexture(QOpenGLTexture::Target2D);
        m_frameTexture->setMinificationFilter(QOpenGLTexture::Linear);
        m_frameTexture->setMagnificationFilter(QOpenGLTexture::Linear);
        m_frameTexture->setWrapMode(QOpenGLTexture::ClampToEdge);
    }
    
    // 确保数据是连续的
    cv::Mat continuousFrame;
    if (!frame.isContinuous()) {
        frame.copyTo(continuousFrame);
    } else {
        continuousFrame = frame;
    }
    
    // 上传纹理数据
    GLenum format = GL_BGR;
    if (continuousFrame.channels() == 4) {
        format = GL_BGRA;
    } else if (continuousFrame.channels() == 1) {
        format = GL_LUMINANCE;
    }
    
    m_frameTexture->setSize(continuousFrame.cols, continuousFrame.rows);
    m_frameTexture->setFormat(QOpenGLTexture::RGB8_UNorm);
    m_frameTexture->allocateStorage();
    
    glBindTexture(GL_TEXTURE_2D, m_frameTexture->textureId());
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, continuousFrame.cols, continuousFrame.rows, 
                 0, format, GL_UNSIGNED_BYTE, continuousFrame.ptr());
    glBindTexture(GL_TEXTURE_2D, 0);
}

void VideoRenderer::renderFrame()
{
    if (!m_frameTexture || !m_shaderProgram) {
        return;
    }
    
    // 使用着色器程序
    m_shaderProgram->bind();
    
    // 计算MVP矩阵
    QMatrix4x4 mvpMatrix = m_projectionMatrix * m_viewMatrix * m_modelMatrix;
    
    // 应用缩放和平移
    QMatrix4x4 transformMatrix;
    transformMatrix.translate(m_panOffset.x() / width(), -m_panOffset.y() / height(), 0.0f);
    transformMatrix.scale(m_zoomFactor, m_zoomFactor, 1.0f);
    mvpMatrix = mvpMatrix * transformMatrix;
    
    // 设置uniform
    m_shaderProgram->setUniformValue(m_mvpMatrixUniform, mvpMatrix);
    m_shaderProgram->setUniformValue(m_textureUniform, 0);
    
    // 绑定纹理
    glActiveTexture(GL_TEXTURE0);
    m_frameTexture->bind();
    
    // 绘制
    m_vao.bind();
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
    m_vao.release();
    
    m_frameTexture->release();
    m_shaderProgram->release();
}

void VideoRenderer::renderDetectionOverlay()
{
    // 这里可以实现检测结果的覆盖层渲染
    // 例如绘制边界框、置信度文本等
    // 由于OpenGL ES的限制，这里可能需要使用额外的着色器或QPainter
    
    // 暂时使用QPainter实现简单的覆盖层
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    
    // 绘制检测框
    if (m_hasDetection) {
        QColor boxColor = m_detectionConfidence > 0.7 ? Qt::green : Qt::yellow;
        QPen pen(boxColor, 3);
        painter.setPen(pen);
        
        // 将归一化坐标转换为屏幕坐标
        int x = static_cast<int>(m_detectionRect.x() * width());
        int y = static_cast<int>(m_detectionRect.y() * height());
        int w = static_cast<int>(m_detectionRect.width() * width());
        int h = static_cast<int>(m_detectionRect.height() * height());
        
        painter.drawRect(x, y, w, h);
        
        // 绘制置信度文本
        QString confidenceText = QString("竹子 %1%").arg(static_cast<int>(m_detectionConfidence * 100));
        QFont font = painter.font();
        font.setPointSize(12);
        font.setBold(true);
        painter.setFont(font);
        
        QRect textRect = painter.fontMetrics().boundingRect(confidenceText);
        textRect.translate(x, y - 5);
        
        painter.fillRect(textRect.adjusted(-5, -2, 5, 2), boxColor);
        painter.setPen(Qt::black);
        painter.drawText(textRect, confidenceText);
    }
    
    // 绘制FPS信息
    if (m_currentFps > 0) {
        painter.setPen(Qt::white);
        QFont font = painter.font();
        font.setPointSize(10);
        painter.setFont(font);
        painter.drawText(10, 20, QString("FPS: %1").arg(m_currentFps, 0, 'f', 1));
    }
}

void VideoRenderer::calculateFPS()
{
    m_frameCount++;
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
    
    if (m_lastFpsTime == 0) {
        m_lastFpsTime = currentTime;
        return;
    }
    
    qint64 timeDiff = currentTime - m_lastFpsTime;
    if (timeDiff >= 1000) { // 每秒更新一次FPS
        m_currentFps = (m_frameCount * 1000.0f) / timeDiff;
        m_frameCount = 0;
        m_lastFpsTime = currentTime;
    }
}

void VideoRenderer::setFrame(const cv::Mat& frame)
{
    if (frame.empty()) {
        return;
    }
    
    // 在主线程中更新帧数据
    QMetaObject::invokeMethod(this, [this, frame]() {
        m_currentFrame = frame.clone();
        m_frameUpdated = true;
        update(); // 触发重绘
    }, Qt::QueuedConnection);
}

void VideoRenderer::setDetectionResult(const QRect& bambooRect, float confidence)
{
    m_detectionRect = bambooRect;
    m_detectionConfidence = confidence;
    m_hasDetection = (confidence > 0.0f);
    update(); // 触发重绘
}

void VideoRenderer::setZoomFactor(float zoom)
{
    m_zoomFactor = qBound(0.1f, zoom, 10.0f);
    emit zoomChanged(m_zoomFactor);
    update();
}

void VideoRenderer::setPanOffset(const QPointF& offset)
{
    m_panOffset = offset;
    update();
}

void VideoRenderer::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        m_lastMousePos = event->pos();
        m_isPanning = true;
        setCursor(Qt::ClosedHandCursor);
    }
    
    QOpenGLWidget::mousePressEvent(event);
}

void VideoRenderer::mouseMoveEvent(QMouseEvent *event)
{
    if (m_isPanning && (event->buttons() & Qt::LeftButton)) {
        QPoint delta = event->pos() - m_lastMousePos;
        m_panOffset += QPointF(delta.x(), delta.y());
        m_lastMousePos = event->pos();
        update();
    }
    
    QOpenGLWidget::mouseMoveEvent(event);
}

void VideoRenderer::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        m_isPanning = false;
        setCursor(Qt::ArrowCursor);
        
        // 发射触摸点信号
        emit touchPoint(event->pos());
    }
    
    QOpenGLWidget::mouseReleaseEvent(event);
}

void VideoRenderer::wheelEvent(QWheelEvent *event)
{
    // 缩放
    float scaleFactor = 1.0f + (event->angleDelta().y() / 1200.0f);
    setZoomFactor(m_zoomFactor * scaleFactor);
    
    QOpenGLWidget::wheelEvent(event);
}

void VideoRenderer::onRenderTimer()
{
    // 定期更新渲染，即使没有新帧也保持界面响应
    if (m_frameUpdated || m_hasDetection) {
        update();
    }
}