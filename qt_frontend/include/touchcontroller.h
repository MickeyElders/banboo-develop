#ifndef TOUCHCONTROLLER_H
#define TOUCHCONTROLLER_H

#include <QtCore/QObject>
#include <QtCore/QTimer>
#include <QtCore/QPointF>
#include <QtCore/QRect>
#include <QtGui/QTouchEvent>
#include <QtWidgets/QGestureEvent>
#include <QtWidgets/QPinchGesture>
#include <QtWidgets/QPanGesture>
#include <QtWidgets/QTapGesture>

class TouchController : public QObject
{
    Q_OBJECT

public:
    enum TouchState {
        Idle,
        SingleTouch,
        MultiTouch,
        Gesture
    };

    enum GestureType {
        None,
        Tap,
        DoubleTap,
        LongPress,
        Pan,
        Pinch,
        Swipe
    };

    explicit TouchController(QObject *parent = nullptr);
    ~TouchController();

    bool handleTouchEvent(QTouchEvent *event);
    bool handleGestureEvent(QGestureEvent *event);
    
    void setTouchArea(const QRect& area);
    void setMultiTouchEnabled(bool enabled);
    void setGestureEnabled(bool enabled);
    
    TouchState currentState() const { return m_currentState; }
    QPointF lastTouchPoint() const { return m_lastTouchPoint; }

signals:
    void touchPointPressed(const QPointF& point);
    void touchPointMoved(const QPointF& point);
    void touchPointReleased(const QPointF& point);
    
    void gestureDetected(GestureType type, const QPointF& center, const QVariantMap& data);
    void tapGesture(const QPointF& point);
    void doubleTapGesture(const QPointF& point);
    void longPressGesture(const QPointF& point);
    void panGesture(const QPointF& delta, const QPointF& center);
    void pinchGesture(float scaleFactor, const QPointF& center);
    void swipeGesture(const QPointF& direction, float velocity);

private slots:
    void onLongPressTimer();
    void onDoubleTapTimer();

private:
    void processTouch(const QList<QTouchEvent::TouchPoint>& touchPoints);
    void detectGesture();
    void resetState();
    bool isValidTouchPoint(const QPointF& point) const;
    float calculateDistance(const QPointF& p1, const QPointF& p2) const;
    float calculateAngle(const QPointF& p1, const QPointF& p2) const;

    // 触摸状态
    TouchState m_currentState;
    QRect m_touchArea;
    bool m_multiTouchEnabled;
    bool m_gestureEnabled;

    // 触摸点数据
    QPointF m_lastTouchPoint;
    QPointF m_firstTouchPoint;
    QList<QPointF> m_touchHistory;
    qint64 m_touchStartTime;
    qint64 m_lastTouchTime;

    // 多点触摸数据
    QList<QPointF> m_activeTouchPoints;
    float m_initialDistance;
    float m_currentDistance;
    QPointF m_gestureCenter;

    // 手势识别
    GestureType m_currentGesture;
    QTimer *m_longPressTimer;
    QTimer *m_doubleTapTimer;
    bool m_waitingForDoubleTap;
    
    // 配置参数
    static const int LONG_PRESS_DURATION = 800;  // ms
    static const int DOUBLE_TAP_INTERVAL = 300;  // ms
    static const float MIN_PAN_DISTANCE;         // pixels
    static const float MIN_PINCH_DISTANCE;       // pixels
    static const float MIN_SWIPE_VELOCITY;       // pixels/ms
};

#endif // TOUCHCONTROLLER_H