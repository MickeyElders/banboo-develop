import QtQuick 6.0
import QtQuick.Controls 6.0
import QtQuick.Layouts 6.0
import QtQuick.Dialogs 6.0

Dialog {
    id: root
    title: "系统设置"
    modal: true
    width: 600
    height: 500
    
    property alias cameraDeviceId: cameraSettings.deviceId
    property alias cameraResolutionWidth: cameraSettings.resolutionWidth
    property alias cameraResolutionHeight: cameraSettings.resolutionHeight
    property alias cameraFrameRate: cameraSettings.frameRate
    property alias detectionModelPath: detectionSettings.modelPath
    property alias detectionConfidenceThreshold: detectionSettings.confidenceThreshold
    property alias enableGPU: detectionSettings.enableGPU
    property alias enableTensorRT: detectionSettings.enableTensorRT

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 20

            // 摄像头设置
            GroupBox {
                title: "摄像头设置"
                Layout.fillWidth: true

                GridLayout {
                    id: cameraSettings
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 15
                    rowSpacing: 10

                    property int deviceId: 0
                    property int resolutionWidth: 1920
                    property int resolutionHeight: 1080
                    property int frameRate: 30

                    Label { text: "设备ID:" }
                    SpinBox {
                        from: 0
                        to: 10
                        value: cameraSettings.deviceId
                        onValueChanged: cameraSettings.deviceId = value
                    }

                    Label { text: "分辨率:" }
                    RowLayout {
                        ComboBox {
                            model: ["1920x1080", "1280x720", "640x480", "自定义"]
                            currentIndex: 0
                            onActivated: function(index) {
                                switch(index) {
                                case 0: 
                                    cameraSettings.resolutionWidth = 1920
                                    cameraSettings.resolutionHeight = 1080
                                    break
                                case 1:
                                    cameraSettings.resolutionWidth = 1280
                                    cameraSettings.resolutionHeight = 720
                                    break
                                case 2:
                                    cameraSettings.resolutionWidth = 640
                                    cameraSettings.resolutionHeight = 480
                                    break
                                }
                            }
                        }
                    }

                    Label { text: "帧率:" }
                    SpinBox {
                        from: 5
                        to: 60
                        value: cameraSettings.frameRate
                        suffix: " FPS"
                        onValueChanged: cameraSettings.frameRate = value
                    }

                    Label { text: "硬件加速:" }
                    Switch {
                        checked: true
                    }

                    Label { text: "自动曝光:" }
                    Switch {
                        checked: true
                    }

                    Label { text: "自动白平衡:" }
                    Switch {
                        checked: true
                    }
                }
            }

            // 检测设置
            GroupBox {
                title: "AI检测设置"
                Layout.fillWidth: true

                GridLayout {
                    id: detectionSettings
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 15
                    rowSpacing: 10

                    property string modelPath: "/models/best.pt"
                    property real confidenceThreshold: 0.7
                    property bool enableGPU: true
                    property bool enableTensorRT: true

                    Label { text: "模型文件:" }
                    RowLayout {
                        TextField {
                            Layout.fillWidth: true
                            text: detectionSettings.modelPath
                            onTextChanged: detectionSettings.modelPath = text
                        }
                        Button {
                            text: "浏览"
                            onClicked: modelFileDialog.open()
                        }
                    }

                    Label { text: "置信度阈值:" }
                    RowLayout {
                        Slider {
                            Layout.fillWidth: true
                            from: 0.1
                            to: 1.0
                            value: detectionSettings.confidenceThreshold
                            onValueChanged: detectionSettings.confidenceThreshold = value
                        }
                        Label {
                            text: Math.round(detectionSettings.confidenceThreshold * 100) + "%"
                            Layout.minimumWidth: 40
                        }
                    }

                    Label { text: "启用GPU:" }
                    Switch {
                        checked: detectionSettings.enableGPU
                        onCheckedChanged: detectionSettings.enableGPU = checked
                    }

                    Label { text: "启用TensorRT:" }
                    Switch {
                        checked: detectionSettings.enableTensorRT
                        enabled: detectionSettings.enableGPU
                        onCheckedChanged: detectionSettings.enableTensorRT = checked
                    }

                    Label { text: "INT8量化:" }
                    Switch {
                        checked: true
                        enabled: detectionSettings.enableTensorRT
                    }

                    Label { text: "批处理大小:" }
                    SpinBox {
                        from: 1
                        to: 8
                        value: 1
                    }
                }
            }

            // 系统设置
            GroupBox {
                title: "系统设置"
                Layout.fillWidth: true

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 15
                    rowSpacing: 10

                    Label { text: "语言:" }
                    ComboBox {
                        model: ["简体中文", "English"]
                        currentIndex: 0
                    }

                    Label { text: "主题:" }
                    ComboBox {
                        model: ["浅色", "深色", "自动"]
                        currentIndex: 0
                    }

                    Label { text: "全屏启动:" }
                    Switch {
                        checked: false
                    }

                    Label { text: "自动启动:" }
                    Switch {
                        checked: false
                    }

                    Label { text: "启用日志:" }
                    Switch {
                        checked: true
                    }

                    Label { text: "调试模式:" }
                    Switch {
                        checked: false
                    }
                }
            }

            // 网络设置
            GroupBox {
                title: "网络设置"
                Layout.fillWidth: true

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 15
                    rowSpacing: 10

                    Label { text: "服务器地址:" }
                    TextField {
                        Layout.fillWidth: true
                        text: "192.168.1.100"
                        placeholderText: "输入服务器IP地址"
                    }

                    Label { text: "端口:" }
                    SpinBox {
                        from: 1024
                        to: 65535
                        value: 8080
                    }

                    Label { text: "串口:" }
                    ComboBox {
                        model: ["/dev/ttyUSB0", "/dev/ttyACM0", "/dev/ttyS0"]
                        currentIndex: 0
                        editable: true
                    }

                    Label { text: "波特率:" }
                    ComboBox {
                        model: ["9600", "19200", "38400", "57600", "115200"]
                        currentIndex: 4
                    }

                    Label { text: "连接超时:" }
                    SpinBox {
                        from: 1
                        to: 60
                        value: 5
                        suffix: " 秒"
                    }

                    Label { text: "启用SSL:" }
                    Switch {
                        checked: false
                    }
                }
            }

            // 硬件设置
            GroupBox {
                title: "硬件设置"
                Layout.fillWidth: true

                GridLayout {
                    anchors.fill: parent
                    columns: 2
                    columnSpacing: 15
                    rowSpacing: 10

                    Label { text: "控制器类型:" }
                    ComboBox {
                        model: ["Modbus RTU", "Modbus TCP", "串口通信"]
                        currentIndex: 0
                    }

                    Label { text: "默认切割速度:" }
                    SpinBox {
                        from: 10
                        to: 100
                        value: 50
                        suffix: "%"
                    }

                    Label { text: "默认切割深度:" }
                    SpinBox {
                        from: 10
                        to: 80
                        value: 30
                        suffix: " mm"
                    }

                    Label { text: "自动收回:" }
                    Switch {
                        checked: true
                    }

                    Label { text: "收回延时:" }
                    SpinBox {
                        from: 100
                        to: 5000
                        value: 1000
                        suffix: " ms"
                    }

                    Label { text: "力阈值:" }
                    SpinBox {
                        from: 10
                        to: 100
                        value: 80
                        suffix: " N"
                    }
                }
            }
        }
    }

    // 对话框按钮
    standardButtons: Dialog.Ok | Dialog.Cancel | Dialog.Apply | Dialog.Reset

    onAccepted: {
        console.log("Settings saved")
        // 应用设置逻辑
    }

    onApplied: {
        console.log("Settings applied")
        // 应用设置逻辑（不关闭对话框）
    }

    onReset: {
        console.log("Settings reset to defaults")
        // 重置为默认设置
    }

    // 文件选择对话框
    FileDialog {
        id: modelFileDialog
        title: "选择模型文件"
        folder: shortcuts.home
        nameFilters: ["模型文件 (*.pt *.onnx *.trt)", "所有文件 (*)"]
        
        onAccepted: {
            detectionSettings.modelPath = fileUrl.toString()
        }
    }
}