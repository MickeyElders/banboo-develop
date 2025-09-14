import QtQuick 6.0
import QtQuick.Controls 6.0

Rectangle {
    id: root
    color: "#000000"
    border.color: "#333333"
    border.width: 1

    property real zoomFactor: 1.0
    property point panOffset: Qt.point(0, 0)
    property bool hasDetection: false
    property rect detectionRect: Qt.rect(0, 0, 0, 0)
    property real detectionConfidence: 0.0

    signal touchPoint(point position)
    signal zoomChanged(real zoom)

    function setZoomFactor(zoom) {
        root.zoomFactor = Math.max(0.5, Math.min(5.0, zoom))
        zoomChanged(root.zoomFactor)
    }

    function resetView() {
        root.zoomFactor = 1.0
        root.panOffset = Qt.point(0, 0)
        zoomChanged(root.zoomFactor)
    }

    // 视频渲染区域
    Rectangle {
        id: videoArea
        anchors.centerIn: parent
        width: parent.width * root.zoomFactor
        height: parent.height * root.zoomFactor
        color: "#1a1a1a"
        
        transform: Translate {
            x: root.panOffset.x
            y: root.panOffset.y
        }

        // 占位符 - 实际项目中会被C++的VideoRenderer替换
        Column {
            anchors.centerIn: parent
            spacing: 10
            visible: !root.hasDetection

            Rectangle {
                width: 60
                height: 60
                radius: 30
                color: "#333333"
                anchors.horizontalCenter: parent.horizontalCenter

                Text {
                    anchors.centerIn: parent
                    text: "📹"
                    font.pixelSize: 24
                    color: "#666666"
                }
            }

            Text {
                text: "等待视频信号..."
                color: "#666666"
                font.pixelSize: 16
                anchors.horizontalCenter: parent.horizontalCenter
            }
        }

        // 检测结果覆盖层
        Rectangle {
            id: detectionOverlay
            visible: root.hasDetection
            x: root.detectionRect.x * parent.width
            y: root.detectionRect.y * parent.height
            width: root.detectionRect.width * parent.width
            height: root.detectionRect.height * parent.height
            color: "transparent"
            border.color: root.detectionConfidence > 0.7 ? "#00ff00" : "#ffaa00"
            border.width: 3

            Rectangle {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.topMargin: -25
                width: confidenceText.width + 10
                height: 25
                color: parent.border.color
                
                Text {
                    id: confidenceText
                    anchors.centerIn: parent
                    text: "竹子 " + Math.round(root.detectionConfidence * 100) + "%"
                    color: "black"
                    font.pixelSize: 12
                    font.bold: true
                }
            }
        }

        // 十字准线
        Rectangle {
            anchors.centerIn: parent
            width: 20
            height: 2
            color: "#ff0000"
            opacity: 0.7
        }

        Rectangle {
            anchors.centerIn: parent
            width: 2
            height: 20
            color: "#ff0000"
            opacity: 0.7
        }
    }

    // 触摸处理
    MultiPointTouchArea {
        anchors.fill: parent
        maximumTouchPoints: 2

        property point lastPanPoint
        property real lastPinchScale: 1.0

        onPressed: function(touchPoints) {
            if (touchPoints.length === 1) {
                // 单点触摸 - 开始平移
                lastPanPoint = touchPoints[0].position
            } else if (touchPoints.length === 2) {
                // 双点触摸 - 开始缩放
                var distance = Math.sqrt(
                    Math.pow(touchPoints[1].x - touchPoints[0].x, 2) +
                    Math.pow(touchPoints[1].y - touchPoints[0].y, 2)
                )
                lastPinchScale = distance
            }
        }

        onUpdated: function(touchPoints) {
            if (touchPoints.length === 1) {
                // 平移
                var delta = Qt.point(
                    touchPoints[0].position.x - lastPanPoint.x,
                    touchPoints[0].position.y - lastPanPoint.y
                )
                root.panOffset = Qt.point(
                    root.panOffset.x + delta.x,
                    root.panOffset.y + delta.y
                )
                lastPanPoint = touchPoints[0].position
            } else if (touchPoints.length === 2) {
                // 缩放
                var distance = Math.sqrt(
                    Math.pow(touchPoints[1].x - touchPoints[0].x, 2) +
                    Math.pow(touchPoints[1].y - touchPoints[0].y, 2)
                )
                var scale = distance / lastPinchScale
                setZoomFactor(root.zoomFactor * scale)
                lastPinchScale = distance
            }
        }

        onReleased: function(touchPoints) {
            if (touchPoints.length === 1) {
                // 发送触摸点信号
                root.touchPoint(touchPoints[0].position)
            }
        }
    }

    // 鼠标处理（PC测试用）
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton

        property point lastPosition

        onPressed: function(mouse) {
            lastPosition = Qt.point(mouse.x, mouse.y)
        }

        onPositionChanged: function(mouse) {
            if (pressed && mouse.buttons & Qt.LeftButton) {
                var delta = Qt.point(mouse.x - lastPosition.x, mouse.y - lastPosition.y)
                root.panOffset = Qt.point(
                    root.panOffset.x + delta.x,
                    root.panOffset.y + delta.y
                )
                lastPosition = Qt.point(mouse.x, mouse.y)
            }
        }

        onClicked: function(mouse) {
            root.touchPoint(Qt.point(mouse.x, mouse.y))
        }

        onWheel: function(wheel) {
            var factor = 1.0 + (wheel.angleDelta.y / 1200.0)
            setZoomFactor(root.zoomFactor * factor)
        }
    }

    // 状态指示器
    Rectangle {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 10
        width: 80
        height: 30
        color: "#80000000"
        radius: 15

        Row {
            anchors.centerIn: parent
            spacing: 5

            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: root.hasDetection ? "#00ff00" : "#ff0000"
                
                SequentialAnimation on opacity {
                    running: true
                    loops: Animation.Infinite
                    NumberAnimation { to: 0.3; duration: 1000 }
                    NumberAnimation { to: 1.0; duration: 1000 }
                }
            }

            Text {
                text: root.hasDetection ? "检测中" : "待机"
                color: "white"
                font.pixelSize: 10
            }
        }
    }
}