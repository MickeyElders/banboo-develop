import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15
import QtMultimedia 5.15
import "components"

ApplicationWindow {
    id: root
    width: 1024
    height: 768
    visible: true
    title: "智能切竹机控制系统"

    property bool isOperating: false
    property real detectionConfidence: 0.0
    property string systemStatus: "待机"
    property real frameRate: 0.0

    // 全屏切换
    function toggleFullscreen() {
        if (root.visibility === Window.FullScreen) {
            root.showNormal()
        } else {
            root.showFullScreen()
        }
    }

    // 主布局
    RowLayout {
        anchors.fill: parent
        spacing: 0

        // 左侧视频显示区域
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width * 0.7
            color: "#1e1e1e"

            VideoDisplay {
                id: videoDisplay
                anchors.fill: parent
                anchors.margins: 10
                
                onTouchPoint: function(point) {
                    console.log("Touch at:", point.x, point.y)
                    // 处理触摸事件
                }

                onZoomChanged: function(zoom) {
                    zoomSlider.value = zoom
                }
            }

            // 视频控制叠加层
            Rectangle {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                height: 80
                color: "#80000000"
                visible: !root.isOperating

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 10

                    Label {
                        text: "缩放:"
                        color: "white"
                    }

                    Slider {
                        id: zoomSlider
                        from: 0.5
                        to: 5.0
                        value: 1.0
                        Layout.fillWidth: true
                        
                        onValueChanged: {
                            videoDisplay.setZoomFactor(value)
                        }
                    }

                    Label {
                        text: Math.round(zoomSlider.value * 100) + "%"
                        color: "white"
                        Layout.minimumWidth: 50
                    }

                    Button {
                        text: "重置"
                        onClicked: {
                            zoomSlider.value = 1.0
                            videoDisplay.resetView()
                        }
                    }
                }
            }
        }

        // 右侧控制面板
        Rectangle {
            Layout.fillHeight: true
            Layout.preferredWidth: parent.width * 0.3
            Layout.minimumWidth: 300
            color: "#f5f5f5"

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 10

                // 状态栏
                StatusBar {
                    id: statusBar
                    Layout.fillWidth: true
                    systemStatus: root.systemStatus
                    frameRate: root.frameRate
                    detectionConfidence: root.detectionConfidence
                }

                // 控制面板
                ControlPanel {
                    id: controlPanel
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    isOperating: root.isOperating

                    onStartOperation: {
                        root.isOperating = true
                        // 启动操作逻辑
                    }

                    onStopOperation: {
                        root.isOperating = false
                        // 停止操作逻辑
                    }

                    onEmergencyStop: {
                        root.isOperating = false
                        // 紧急停止逻辑
                    }

                    onOpenSettings: {
                        settingsDialog.open()
                    }
                }
            }
        }
    }

    // 设置对话框
    SettingsDialog {
        id: settingsDialog
        
        onAccepted: {
            // 应用设置
            console.log("Settings applied")
        }
    }

    // 全屏提示
    Rectangle {
        id: fullscreenHint
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: 20
        width: hintText.width + 20
        height: hintText.height + 10
        color: "#80000000"
        radius: 5
        visible: root.visibility === Window.FullScreen

        Text {
            id: hintText
            anchors.centerIn: parent
            text: "按 ESC 退出全屏 | 双击切换全屏"
            color: "white"
            font.pixelSize: 14
        }

        Timer {
            running: parent.visible
            interval: 3000
            onTriggered: parent.visible = false
        }
    }

    // 键盘事件处理
    Keys.onEscapePressed: {
        if (root.visibility === Window.FullScreen) {
            root.showNormal()
        }
    }

    Keys.onPressed: function(event) {
        switch(event.key) {
        case Qt.Key_F11:
            toggleFullscreen()
            event.accepted = true
            break
        case Qt.Key_Space:
            if (root.isOperating) {
                controlPanel.stopOperation()
            } else {
                controlPanel.startOperation()
            }
            event.accepted = true
            break
        }
    }

    // 双击全屏
    TapHandler {
        acceptedButtons: Qt.LeftButton
        onDoubleTapped: toggleFullscreen()
    }

    Component.onCompleted: {
        // 初始化完成
        console.log("Main QML loaded")
    }
}