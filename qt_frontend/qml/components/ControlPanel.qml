import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Rectangle {
    id: root
    color: "#f8f9fa"
    border.color: "#dee2e6"
    border.width: 1
    radius: 5

    property bool isOperating: false
    property string operationMode: "自动"
    property int cutSpeed: 50
    property int cutDepth: 30
    property real systemTemperature: 35.2
    property bool emergencyStopEnabled: true

    signal startOperation()
    signal stopOperation()
    signal emergencyStop()
    signal openSettings()
    signal modeChanged(string mode)

    ScrollView {
        anchors.fill: parent
        anchors.margins: 15
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 15

            // 操作控制区域
            GroupBox {
                title: "操作控制"
                Layout.fillWidth: true
                
                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    // 主操作按钮
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10

                        Button {
                            id: startButton
                            text: "开始作业"
                            enabled: !root.isOperating
                            Layout.fillWidth: true
                            Material.background: Material.Green
                            Material.foreground: "white"
                            font.pixelSize: 16
                            font.bold: true
                            
                            onClicked: {
                                root.startOperation()
                            }
                        }

                        Button {
                            id: stopButton
                            text: "停止作业"
                            enabled: root.isOperating
                            Layout.fillWidth: true
                            Material.background: Material.Orange
                            Material.foreground: "white"
                            font.pixelSize: 16
                            font.bold: true
                            
                            onClicked: {
                                root.stopOperation()
                            }
                        }
                    }

                    // 紧急停止按钮
                    Button {
                        id: emergencyButton
                        text: "🚨 紧急停止 🚨"
                        enabled: root.emergencyStopEnabled
                        Layout.fillWidth: true
                        Layout.preferredHeight: 60
                        Material.background: Material.Red
                        Material.foreground: "white"
                        font.pixelSize: 18
                        font.bold: true
                        
                        onClicked: {
                            root.emergencyStop()
                        }

                        // 紧急按钮闪烁效果
                        SequentialAnimation on opacity {
                            running: root.isOperating
                            loops: Animation.Infinite
                            NumberAnimation { to: 0.6; duration: 500 }
                            NumberAnimation { to: 1.0; duration: 500 }
                        }
                    }
                }
            }

            // 操作模式选择
            GroupBox {
                title: "操作模式"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    ButtonGroup {
                        id: modeGroup
                        onClicked: function(button) {
                            root.operationMode = button.text
                            root.modeChanged(button.text)
                        }
                    }

                    RadioButton {
                        text: "自动"
                        checked: root.operationMode === "自动"
                        ButtonGroup.group: modeGroup
                        enabled: !root.isOperating
                    }

                    RadioButton {
                        text: "半自动"
                        checked: root.operationMode === "半自动"
                        ButtonGroup.group: modeGroup
                        enabled: !root.isOperating
                    }

                    RadioButton {
                        text: "手动"
                        checked: root.operationMode === "手动"
                        ButtonGroup.group: modeGroup
                        enabled: !root.isOperating
                    }
                }
            }

            // 切割参数
            GroupBox {
                title: "切割参数"
                Layout.fillWidth: true

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 15
                    rowSpacing: 10

                    Label {
                        text: "切割速度:"
                        font.pixelSize: 14
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Slider {
                            id: speedSlider
                            from: 10
                            to: 100
                            value: root.cutSpeed
                            stepSize: 5
                            Layout.fillWidth: true
                            enabled: !root.isOperating

                            onValueChanged: {
                                root.cutSpeed = value
                            }
                        }

                        Label {
                            text: root.cutSpeed + "%"
                            font.pixelSize: 14
                            Layout.minimumWidth: 40
                        }
                    }

                    Label {
                        text: "切割深度:"
                        font.pixelSize: 14
                    }

                    RowLayout {
                        Layout.fillWidth: true

                        Slider {
                            id: depthSlider
                            from: 10
                            to: 80
                            value: root.cutDepth
                            stepSize: 5
                            Layout.fillWidth: true
                            enabled: !root.isOperating

                            onValueChanged: {
                                root.cutDepth = value
                            }
                        }

                        Label {
                            text: root.cutDepth + "mm"
                            font.pixelSize: 14
                            Layout.minimumWidth: 50
                        }
                    }
                }
            }

            // 手动控制 (仅在手动模式下显示)
            GroupBox {
                title: "手动控制"
                Layout.fillWidth: true
                visible: root.operationMode === "手动"

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 10
                    rowSpacing: 10

                    Button {
                        text: "刀片上升"
                        Layout.fillWidth: true
                        enabled: root.isOperating
                    }

                    Button {
                        text: "刀片下降"
                        Layout.fillWidth: true
                        enabled: root.isOperating
                    }

                    Button {
                        text: "启动切割"
                        Layout.fillWidth: true
                        enabled: root.isOperating
                        Material.background: Material.Teal
                        Material.foreground: "white"
                    }

                    Button {
                        text: "收回刀片"
                        Layout.fillWidth: true
                        enabled: root.isOperating
                    }
                }
            }

            // 系统状态
            GroupBox {
                title: "系统状态"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    RowLayout {
                        Layout.fillWidth: true

                        Label {
                            text: "系统温度:"
                            font.pixelSize: 14
                        }

                        Label {
                            text: root.systemTemperature.toFixed(1) + "°C"
                            font.pixelSize: 14
                            color: root.systemTemperature > 60 ? "red" : "green"
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            height: 1
                        }

                        Rectangle {
                            width: 12
                            height: 12
                            radius: 6
                            color: {
                                if (root.systemTemperature > 70) return "red"
                                if (root.systemTemperature > 50) return "orange"
                                return "green"
                            }
                        }
                    }

                    ProgressBar {
                        Layout.fillWidth: true
                        from: 0
                        to: 100
                        value: root.systemTemperature
                        Material.accent: {
                            if (value > 70) return Material.Red
                            if (value > 50) return Material.Orange
                            return Material.Green
                        }
                    }
                }
            }

            // 设置和校准
            GroupBox {
                title: "系统设置"
                Layout.fillWidth: true

                ColumnLayout {
                    anchors.fill: parent
                    spacing: 10

                    Button {
                        text: "系统设置"
                        Layout.fillWidth: true
                        enabled: !root.isOperating
                        
                        onClicked: {
                            root.openSettings()
                        }
                    }

                    Button {
                        text: "校准系统"
                        Layout.fillWidth: true
                        enabled: !root.isOperating
                        Material.background: Material.BlueGrey
                        Material.foreground: "white"
                    }

                    Button {
                        text: "系统重置"
                        Layout.fillWidth: true
                        enabled: !root.isOperating
                        Material.background: Material.Red
                        Material.foreground: "white"
                        
                        onClicked: {
                            resetConfirmDialog.open()
                        }
                    }
                }
            }

            // 填充空间
            Item {
                Layout.fillHeight: true
            }
        }
    }

    // 重置确认对话框
    Dialog {
        id: resetConfirmDialog
        title: "确认重置"
        modal: true
        anchors.centerIn: parent

        Column {
            spacing: 20
            Text {
                text: "确定要重置系统吗？这将重启所有设备。"
                wrapMode: Text.WordWrap
            }

            RowLayout {
                Button {
                    text: "确定"
                    Material.background: Material.Red
                    Material.foreground: "white"
                    onClicked: {
                        resetConfirmDialog.close()
                        // 执行重置逻辑
                        console.log("System reset confirmed")
                    }
                }

                Button {
                    text: "取消"
                    onClicked: resetConfirmDialog.close()
                }
            }
        }
    }
}