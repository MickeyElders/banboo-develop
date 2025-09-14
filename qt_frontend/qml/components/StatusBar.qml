import QtQuick 6.0
import QtQuick.Controls 6.0
import QtQuick.Layouts 6.0

Rectangle {
    id: root
    height: 120
    color: "#e9ecef"
    border.color: "#adb5bd"
    border.width: 1
    radius: 5

    property string systemStatus: "待机"
    property real frameRate: 0.0
    property real detectionConfidence: 0.0
    property int processedFrames: 0
    property string currentTime: ""

    function updateTime() {
        var now = new Date()
        root.currentTime = Qt.formatTime(now, "hh:mm:ss")
    }

    Timer {
        running: true
        repeat: true
        interval: 1000
        onTriggered: root.updateTime()
    }

    Component.onCompleted: updateTime()

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        // 标题栏
        RowLayout {
            Layout.fillWidth: true

            Text {
                text: "系统状态"
                font.pixelSize: 16
                font.bold: true
                color: "#495057"
            }

            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#adb5bd"
            }

            Text {
                text: root.currentTime
                font.pixelSize: 14
                color: "#6c757d"
            }
        }

        // 主要状态信息
        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 15
            rowSpacing: 8

            // 系统状态
            Rectangle {
                Layout.fillWidth: true
                Layout.columnSpan: 2
                height: 40
                color: {
                    switch(root.systemStatus) {
                    case "运行中": return "#d4edda"
                    case "错误": return "#f8d7da"
                    case "维护": return "#fff3cd"
                    default: return "#d1ecf1"
                    }
                }
                border.color: {
                    switch(root.systemStatus) {
                    case "运行中": return "#c3e6cb"
                    case "错误": return "#f5c6cb"
                    case "维护": return "#ffeaa7"
                    default: return "#bee5eb"
                    }
                }
                border.width: 1
                radius: 3

                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 10

                    Rectangle {
                        width: 12
                        height: 12
                        radius: 6
                        color: {
                            switch(root.systemStatus) {
                            case "运行中": return "#28a745"
                            case "错误": return "#dc3545"
                            case "维护": return "#ffc107"
                            default: return "#17a2b8"
                            }
                        }

                        SequentialAnimation on opacity {
                            running: root.systemStatus === "运行中"
                            loops: Animation.Infinite
                            NumberAnimation { to: 0.3; duration: 800 }
                            NumberAnimation { to: 1.0; duration: 800 }
                        }
                    }

                    Text {
                        text: "状态: " + root.systemStatus
                        font.pixelSize: 14
                        font.bold: true
                        color: {
                            switch(root.systemStatus) {
                            case "运行中": return "#155724"
                            case "错误": return "#721c24"
                            case "维护": return "#856404"
                            default: return "#0c5460"
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                    }

                    Text {
                        text: root.processedFrames + " 帧"
                        font.pixelSize: 12
                        color: "#6c757d"
                    }
                }
            }

            // 帧率信息
            ColumnLayout {
                spacing: 2

                Text {
                    text: "帧率"
                    font.pixelSize: 12
                    color: "#6c757d"
                }

                Text {
                    text: root.frameRate.toFixed(1) + " FPS"
                    font.pixelSize: 14
                    font.bold: true
                    color: root.frameRate > 25 ? "#28a745" : 
                          root.frameRate > 15 ? "#ffc107" : "#dc3545"
                }

                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 30
                    value: root.frameRate
                    Material.accent: root.frameRate > 25 ? Material.Green : 
                                   root.frameRate > 15 ? Material.Orange : Material.Red
                }
            }

            // 检测置信度
            ColumnLayout {
                spacing: 2

                Text {
                    text: "检测精度"
                    font.pixelSize: 12
                    color: "#6c757d"
                }

                Text {
                    text: Math.round(root.detectionConfidence * 100) + "%"
                    font.pixelSize: 14
                    font.bold: true
                    color: root.detectionConfidence > 0.8 ? "#28a745" : 
                          root.detectionConfidence > 0.6 ? "#ffc107" : "#dc3545"
                }

                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 1
                    value: root.detectionConfidence
                    Material.accent: root.detectionConfidence > 0.8 ? Material.Green : 
                                   root.detectionConfidence > 0.6 ? Material.Orange : Material.Red
                }
            }
        }
    }

    // 连接状态指示器
    Row {
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: 5
        spacing: 5

        Rectangle {
            width: 8
            height: 8
            radius: 4
            color: "#28a745"
            
            ToolTip.visible: connectionMouseArea.containsMouse
            ToolTip.text: "摄像头连接正常"

            MouseArea {
                id: connectionMouseArea
                anchors.fill: parent
                hoverEnabled: true
            }
        }

        Rectangle {
            width: 8
            height: 8
            radius: 4
            color: root.detectionConfidence > 0 ? "#28a745" : "#6c757d"
            
            ToolTip.visible: detectionMouseArea.containsMouse
            ToolTip.text: root.detectionConfidence > 0 ? "AI检测正常" : "等待检测"

            MouseArea {
                id: detectionMouseArea
                anchors.fill: parent
                hoverEnabled: true
            }
        }

        Rectangle {
            width: 8
            height: 8
            radius: 4
            color: root.systemStatus === "运行中" ? "#28a745" : "#ffc107"
            
            ToolTip.visible: systemMouseArea.containsMouse
            ToolTip.text: "系统状态: " + root.systemStatus

            MouseArea {
                id: systemMouseArea
                anchors.fill: parent
                hoverEnabled: true
            }
        }
    }
}