#include "touchcontroller.h"
#include <QtCore/QDateTime>
#include <QtCore/QLoggingCategory>
#include <QtCore/QVariantMap>
#include <QtGui/QTouchEvent>
#include <QtWidgets/QGestureEvent>
#include <qmath.h>

Q_LOGGING_CATEGORY(touchController, "app.touchcontroller")

// 静态常量定义
const float TouchController::MIN_PAN_DISTANCE = 20.0f;
const float TouchController::MIN_PINCH_DISTANCE = 50.0f;
const float TouchController::MIN_SWIPE_VELOCITY = 0.5f;

TouchController::TouchController(QObject *parent)
    : QObject(parent)
    , m_currentState(Idle)
    , m_multiTouchEnabled(true)
    , m_gestureEnabled(true)
    , m_touchStartTime(0)
    , m_lastTouchTime(0)
    , m_initialDistance(0.0f)
    , m_currentDistance(0.0f)
    , m_currentGesture(None)
    , m_waitingForDoubleTap(false)
{
    qCInfo(touchController) << "Creating TouchController...";
    
    // 创建定时器
    m_longPressTimer = new QTimer(this);
    m_longPressTimer->setSingleShot(true);
    connect(m_longPressTimer, &QTimer::timeout, this, &TouchController::onLongPressTimer);
    
    m_doubleTapTimer = new QTimer(this);
    m_doubleTapTimer->setSingleShot(true);
    connect(m_doubleTapTimer, &QTimer::timeout, this, &TouchController::onDoubleTapTimer);
    
    // 设置默认触摸区域
    m_touchArea = QRect(0, 0, 1920, 1080);
}

TouchController::~TouchController()
{
    qCInfo(touchController) << "Destroying TouchController...";
}

bool TouchController::handleTouchEvent(QTouchEvent *event)
{
    if (!event || !m_gestureEnabled) {
        return false;
    }
    
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
    const QList<QTouchEvent::TouchPoint> touchPoints = event->touchPoints();
    
    // 过滤有效触摸点
    QList<QTouchEvent::TouchPoint> validPoints;
    for (const auto& point : touchPoints) {
        if (isValidTouchPoint(point.position())) {
            validPoints.append(point);
        }
    }
    
    if (validPoints.isEmpty()) {
        return false;
    }
    
    // 更新时间戳
    m_lastTouchTime = currentTime;
    
    // 处理触摸事件
    switch (event->type()) {
    case QEvent::TouchBegin:
        m_touchStartTime = currentTime;
        m_firstTouchPoint = validPoints.first().position();
        m_lastTouchPoint = m_firstTouchPoint;
        m_touchHistory.clear();
        m_touchHistory.append(m_firstTouchPoint);
        
        if (validPoints.size() == 1) {
            m_currentState = SingleTouch;
            emit touchPointPressed(m_firstTouchPoint);
            
            // 启动长按定时器
            m_longPressTimer->start(LONG_PRESS_DURATION);
            
        } else if (validPoints.size() > 1 && m_multiTouchEnabled) {
            m_currentState = MultiTouch;
            m_activeTouchPoints.clear();
            for (const auto& point : validPoints) {
                m_activeTouchPoints.append(point.position());
            }
            
            // 计算初始距离（用于缩放手势）
            if (m_activeTouchPoints.size() >= 2) {
                m_initialDistance = calculateDistance(m_activeTouchPoints[0], m_activeTouchPoints[1]);
                m_currentDistance = m_initialDistance;
                m_gestureCenter = (m_activeTouchPoints[0] + m_activeTouchPoints[1]) / 2;
            }
        }
        break;
        
    case QEvent::TouchUpdate:
        processTouch(validPoints);
        break;
        
    case QEvent::TouchEnd:
        m_longPressTimer->stop();
        
        if (m_currentState == SingleTouch) {
            emit touchPointReleased(m_lastTouchPoint);
            
            // 检测点击手势
            qint64 touchDuration = currentTime - m_touchStartTime;
            float touchDistance = calculateDistance(m_firstTouchPoint, m_lastTouchPoint);
            
            if (touchDuration < LONG_PRESS_DURATION && touchDistance < MIN_PAN_DISTANCE) {
                if (m_waitingForDoubleTap) {
                    // 双击
                    m_doubleTapTimer->stop();
                    m_waitingForDoubleTap = false;
                    emit doubleTapGesture(m_firstTouchPoint);
                    emit gestureDetected(DoubleTap, m_firstTouchPoint, QVariantMap());
                } else {
                    // 可能是单击，等待双击
                    m_waitingForDoubleTap = true;
                    m_doubleTapTimer->start(DOUBLE_TAP_INTERVAL);
                }
            }
        }
        
        resetState();
        break;
        
    default:
        break;
    }
    
    return true;
}

bool TouchController::handleGestureEvent(QGestureEvent *event)
{
    if (!event || !m_gestureEnabled) {
        return false;
    }
    
    if (QPinchGesture *pinch = static_cast<QPinchGesture*>(event->gesture(Qt::PinchGesture))) {
        QVariantMap data;
        data["scaleFactor"] = pinch->scaleFactor();
        data["totalScaleFactor"] = pinch->totalScaleFactor();
        
        emit pinchGesture(pinch->scaleFactor(), pinch->centerPoint());
        emit gestureDetected(Pinch, pinch->centerPoint(), data);
        return true;
    }
    
    if (QPanGesture *pan = static_cast<QPanGesture*>(event->gesture(Qt::PanGesture))) {
        QVariantMap data;
        data["delta"] = pan->delta();
        data["acceleration"] = pan->acceleration();
        
        emit panGesture(pan->delta(), pan->lastOffset());
        emit gestureDetected(Pan, pan->lastOffset(), data);
        return true;
    }
    
    if (QTapGesture *tap = static_cast<QTapGesture*>(event->gesture(Qt::TapGesture))) {
        emit tapGesture(tap->position());
        emit gestureDetected(Tap, tap->position(), QVariantMap());
        return true;
    }
    
    return false;
}

void TouchController::setTouchArea(const QRect& area)
{
    m_touchArea = area;
    qCDebug(touchController) << "Touch area set to:" << area;
}

void TouchController::setMultiTouchEnabled(bool enabled)
{
    m_multiTouchEnabled = enabled;
    qCDebug(touchController) << "Multi-touch enabled:" << enabled;
}

void TouchController::setGestureEnabled(bool enabled)
{
    m_gestureEnabled = enabled;
    qCDebug(touchController) << "Gesture enabled:" << enabled;
}

void TouchController::processTouch(const QList<QTouchEvent::TouchPoint>& touchPoints)
{
    if (touchPoints.isEmpty()) {
        return;
    }
    
    QPointF currentPoint = touchPoints.first().position();
    m_lastTouchPoint = currentPoint;
    m_touchHistory.append(currentPoint);
    
    // 限制历史记录长度
    if (m_touchHistory.size() > 10) {
        m_touchHistory.removeFirst();
    }
    
    emit touchPointMoved(currentPoint);
    
    if (m_currentState == SingleTouch) {
        // 检测拖拽手势
        float distance = calculateDistance(m_firstTouchPoint, currentPoint);
        if (distance > MIN_PAN_DISTANCE) {
            m_longPressTimer->stop(); // 停止长按检测
            
            QPointF delta = currentPoint - m_firstTouchPoint;
            emit panGesture(delta, currentPoint);
            
            QVariantMap data;
            data["delta"] = delta;
            data["distance"] = distance;
            emit gestureDetected(Pan, currentPoint, data);
        }
    } else if (m_currentState == MultiTouch && touchPoints.size() >= 2) {
        // 更新多点触摸数据
        m_activeTouchPoints.clear();
        for (const auto& point : touchPoints) {
            if (isValidTouchPoint(point.position())) {
                m_activeTouchPoints.append(point.position());
            }
        }
        
        if (m_activeTouchPoints.size() >= 2) {
            // 计算当前距离
            float newDistance = calculateDistance(m_activeTouchPoints[0], m_activeTouchPoints[1]);
            float scaleFactor = newDistance / m_initialDistance;
            
            if (qAbs(scaleFactor - 1.0f) > 0.1f) { // 10%的变化阈值
                m_currentDistance = newDistance;
                m_gestureCenter = (m_activeTouchPoints[0] + m_activeTouchPoints[1]) / 2;
                
                emit pinchGesture(scaleFactor, m_gestureCenter);
                
                QVariantMap data;
                data["scaleFactor"] = scaleFactor;
                data["initialDistance"] = m_initialDistance;
                data["currentDistance"] = m_currentDistance;
                emit gestureDetected(Pinch, m_gestureCenter, data);
            }
        }
    }
}

void TouchController::detectGesture()
{
    if (m_touchHistory.size() < 2) {
        return;
    }
    
    QPointF start = m_touchHistory.first();
    QPointF end = m_touchHistory.last();
    
    float distance = calculateDistance(start, end);
    qint64 timeDiff = m_lastTouchTime - m_touchStartTime;
    
    if (timeDiff > 0) {
        float velocity = distance / timeDiff;
        
        if (velocity > MIN_SWIPE_VELOCITY && distance > MIN_PAN_DISTANCE * 2) {
            QPointF direction = (end - start).normalized();
            
            emit swipeGesture(direction, velocity);
            
            QVariantMap data;
            data["direction"] = direction;
            data["velocity"] = velocity;
            data["distance"] = distance;
            emit gestureDetected(Swipe, end, data);
            
            m_currentGesture = Swipe;
        }
    }
}

void TouchController::resetState()
{
    m_currentState = Idle;
    m_currentGesture = None;
    m_activeTouchPoints.clear();
    m_touchHistory.clear();
    m_initialDistance = 0.0f;
    m_currentDistance = 0.0f;
}

bool TouchController::isValidTouchPoint(const QPointF& point) const
{
    return m_touchArea.contains(point.toPoint());
}

float TouchController::calculateDistance(const QPointF& p1, const QPointF& p2) const
{
    QPointF delta = p2 - p1;
    return qSqrt(delta.x() * delta.x() + delta.y() * delta.y());
}

float TouchController::calculateAngle(const QPointF& p1, const QPointF& p2) const
{
    QPointF delta = p2 - p1;
    return qAtan2(delta.y(), delta.x()) * 180.0f / M_PI;
}

void TouchController::onLongPressTimer()
{
    if (m_currentState == SingleTouch) {
        // 检查是否仍然在原位置附近
        float distance = calculateDistance(m_firstTouchPoint, m_lastTouchPoint);
        if (distance < MIN_PAN_DISTANCE) {
            emit longPressGesture(m_lastTouchPoint);
            emit gestureDetected(LongPress, m_lastTouchPoint, QVariantMap());
            m_currentGesture = LongPress;
        }
    }
}

void TouchController::onDoubleTapTimer()
{
    if (m_waitingForDoubleTap) {
        // 超时了，发送单击事件
        m_waitingForDoubleTap = false;
        emit tapGesture(m_firstTouchPoint);
        emit gestureDetected(Tap, m_firstTouchPoint, QVariantMap());
    }
}