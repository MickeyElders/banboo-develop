import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtMultimedia

ApplicationWindow {
    id: root
    visible: true
    width: 1366
    height: 768
    color: "#0f131a"
    title: qsTr("AI竹节识别切割系统 v2.1 - Qt Preview")

    readonly property color bgPanel: "#161b24"
    readonly property color bgSpot: "#1f2632"
    readonly property color border: "#2a3342"
    readonly property color accent: "#ff7a3d"
    readonly property color success: "#4caf50"
    readonly property color warning: "#f2c14f"
    readonly property color error: "#f44336"
    readonly property color textPrimary: "#e8ecf3"
    readonly property color textSecondary: "#9aa4b5"

    Component.onCompleted: {
        // Modbus connect: default 127.0.0.1:1502 (set 502 for PLC)
        modbus.configure("127.0.0.1", 1502, 1)
    }

    Rectangle {
        anchors.fill: parent
        color: "transparent"
        GridLayout {
            anchors.fill: parent
            anchors.margins: 10
            columns: 2
            rows: 3
            rowSpacing: 8
            columnSpacing: 8
            // header
            Rectangle {
                Layout.columnSpan: 2
                Layout.fillWidth: true
                Layout.preferredHeight: 68
                radius: 8
                color: bgPanel
                border.color: border
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 12
                    RowLayout {
                        spacing: 10
                        Label { text: "AI竹节识别切割系统 v2.1"; color: textPrimary; font.bold: true; font.pixelSize: 18 }
                        Rectangle {
                            radius: 6; color: "#1b88ff"
                            anchors.verticalCenter: parent.verticalCenter
                            width: implicitWidth; height: 20; anchors.margins: 0
                            Row {
                                anchors.centerIn: parent
                                spacing: 4
                                anchors.margins: 6
                                Label { text: "Jetson Orin NX · Modbus TCP"; color: "#031422"; font.pixelSize: 11; font.bold: true }
                            }
                        }
                    }
                    Item { Layout.fillWidth: true }
                    RowLayout {
                        spacing: 6
                        Repeater {
                            model: 5
                            delegate: Rectangle {
                                property int plcStep: modbus.plcCommand > 0 ? modbus.plcCommand : 1
                                property bool active: (index + 1) === plcStep
                                property bool done: (index + 1) < plcStep
                                radius: 10
                                color: done ? Qt.rgba(0,1,0,0.18) : active ? Qt.rgba(1,0.47,0.24,0.2) : bgSpot
                                border.color: done ? success : active ? accent : border
                                width: 90; height: 28
                                Label {
                                    anchors.centerIn: parent
                                    color: done ? success : active ? textPrimary : textSecondary
                                    font.pixelSize: 12
                                    text: ["进料检�?,"视觉识别","坐标传输","切割准备","执行切割"][index]
                                }
                            }
                        }
                    }
                    Button {
                        text: "⚙️ 设置"
                        onClicked: settingsDialog.open()
                    }
                    RowLayout {
                        spacing: 6
                        Rectangle { width: 12; height: 12; radius: 6; color: success; anchors.verticalCenter: parent.verticalCenter; opacity: 0.8 }
                        Label { text: "心跳 " + modbus.heartbeat; color: success; font.pixelSize: 12 }
                        Label { text: "响应 12ms"; color: textSecondary; font.pixelSize: 12 }
                    }
                }
            }
            // camera + coords
            Rectangle {
                Layout.row: 1
                Layout.column: 0
                Layout.fillHeight: true
                Layout.fillWidth: true
                radius: 10
                color: bgPanel
                border.color: border
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 8
                    RowLayout {
                        Layout.fillWidth: true
                        Label { text: "📹 实时检测画�?; color: accent; font.pixelSize: 16; font.bold: true }
                        Item { Layout.fillWidth: true }
                        Label { text: "1280x720 | YOLOv8 | 0.1mm"; color: textSecondary; font.pixelSize: 12 }
                    }
                    Rectangle {
                        Layout.fillHeight: true
                        Layout.fillWidth: true
                        radius: 8
                        color: "#000000"
                        border.color: border
                        anchors.margins: 4
                        VideoOutput {
                            id: videoOut
                            anchors.fill: parent
                            source: mediaPlayer
                            fillMode: VideoOutput.PreserveAspectFit
                        }
                        Rectangle {
                            id: rail
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.bottom: parent.bottom
                            anchors.bottomMargin: 12
                            height: 28
                            radius: 6
                            color: Qt.rgba(0.13, 0.59, 0.95, 0.12)
                            border.color: "#2196f3"
                            Label { anchors.centerIn: parent; text: "X轴导�?(0 - 1000.0 mm)"; color: "#2196f3"; font.pixelSize: 12 }
                            Rectangle {
                                width: 2; color: error; anchors.top: parent.top; anchors.bottom: parent.bottom
                                x: parent.width * ((modbus.visionTargetCoord || systemState.xCoordinate) / 1000.0)
                            }
                        }
                    }
                    Rectangle {
                        Layout.fillWidth: true
                        radius: 8
                        color: bgSpot
                        border.color: border
                        GridLayout {
                            anchors.fill: parent
                            anchors.margins: 8
                            columns: 3
                            Repeater {
                                model: [
                                    {label: "X坐标", value: Qt.formatNumber(modbus.visionTargetCoord || systemState.xCoordinate, 'f', 1) + "mm"},
                                    {label: "PLC接收状�?, value: modbus.plcReceiveState === 1 ? "可接�? : (modbus.plcReceiveState === 2 ? "送料�? : "未知")},
                                    {label: "送料伺服当前位置", value: modbus.plcServoPosition.toFixed(1) + " mm"}
                                ]
                                delegate: Rectangle {
                                    color: Qt.rgba(1,1,1,0.02)
                                    border.color: border
                                    radius: 6
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: 50
                                    Column {
                                        anchors.fill: parent
                                        anchors.margins: 6
                                        spacing: 4
                                        Label { text: modelData.label; color: textSecondary; font.pixelSize: 12 }
                                        Label { text: modelData.value; color: accent; font.pixelSize: 16; font.bold: true }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // right panel
            Flickable {
                Layout.row: 1
                Layout.column: 1
                Layout.fillHeight: true
                Layout.preferredWidth: 420
                contentHeight: col.implicitHeight
                clip: true
                Column {
                    id: col
                    width: parent.width
                    spacing: 8
                    function card(title, contentItem) {
                        return contentItem
                    }
                    Rectangle {
                        width: parent.width; radius: 10; color: bgPanel; border.color: border
                        Column {
                            anchors.fill: parent; anchors.margins: 8; spacing: 6
                            RowLayout {
                                Layout.fillWidth: true
                                Label { text: "📊 Modbus"; color: textPrimary; font.bold: true }
                                Item { Layout.fillWidth: true }
                                Label { text: modbus.connected ? "已连�? : "未连�?; color: modbus.connected ? success : error; font.pixelSize: 12 }
                            }
                            Label { text: "PLC �?相机 (D2100 系列)"; color: textSecondary; font.pixelSize: 12 }
                            GridLayout {
                                columns: 2; rowSpacing: 4; columnSpacing: 6
                                Repeater {
                                    model: [
                                        {k:"D2100 通讯请求",v: modbus.plcPowerRequest},
                                        {k:"D2101 接收状�?,v: modbus.plcReceiveState},
                                        {k:"D2102/3 送料伺服",v: modbus.plcServoPosition.toFixed(1)},
                                        {k:"D2104/5 坐标反馈",v: modbus.plcCoordFeedback.toFixed(1)}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 11 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: accent; font.pixelSize: 12; font.family: "Consolas" }
                                    }
                                }
                            }
                            Label { text: "Camera -> PLC (D2000 series)", color: textSecondary; font.pixelSize: 12; anchors.margins: 6 }
                            GridLayout {
                                columns: 2; rowSpacing: 4; columnSpacing: 6
                                Repeater {
                                    model: [
                                        {k:"D2000 通讯应答",v: modbus.visionCommAck},
                                        {k:"D2001 相机状态",v: modbus.visionStatus},
                                        {k:"D2002/3 目标坐标",v: modbus.visionTargetCoord.toFixed(1)},
                                        {k:"D2004 传输结果",v: modbus.visionTransferResult}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 11 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: accent; font.pixelSize: 12; font.family: "Consolas" }
                                    }
                                }
                            }
                        }
                    }
                    Rectangle {
                        width: parent.width; radius: 10; color: bgPanel; border.color: border
                        Column {
                            anchors.fill: parent; anchors.margins: 8; spacing: 6
                            Label { text: "🟢 Jetson Orin NX"; color: textPrimary; font.bold: true }
                            Column {
                                spacing: 4
                                function bar(label, value, color) {
                                    return Column {
                                        spacing: 2
                                        Row {
                                            spacing: 6
                                            Label { text: label; color: textSecondary; font.pixelSize: 11 }
                                            Item { Layout.fillWidth: true }
                                            Label { text: value; color: textPrimary; font.pixelSize: 11 }
                                        }
                                        Rectangle {
                                            height: 8; radius: 4; color: border
                                            Rectangle { anchors.left: parent.left; anchors.top: parent.top; anchors.bottom: parent.bottom; width: parent.width * parseFloat(value)/100.0; radius: 4; color: color }
                                        }
                                    }
                                }
                                bar("CPU", jetsonState.cpuUsage.toFixed(1)+"%", "#76b900")
                                bar("GPU", jetsonState.gpuUsage.toFixed(1)+"%", accent)
                                bar("内存", (jetsonState.memUsed/jetsonState.memTotal*100).toFixed(1)+"%", warning)
                                bar("温度", jetsonState.temp.toFixed(0)+"%", error)
                            }
                            GridLayout {
                                columns: 2; rowSpacing: 4; columnSpacing: 6
                                Repeater {
                                    model: [
                                        {k:"功�?, v: jetsonState.cpuUsage.toFixed(0)+" W"},
                                        {k:"风扇", v: jetsonState.fanRpm.toFixed(0)+" RPM"},
                                        {k:"性能模式", v: jetsonState.perfMode},
                                        {k:"X11", v:"已禁�?}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 11 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: textPrimary; font.pixelSize: 12 }
                                    }
                                }
                            }
                        }
                    }
                    Rectangle {
                        width: parent.width; radius: 10; color: bgPanel; border.color: border
                        Column {
                            anchors.fill: parent; anchors.margins: 8; spacing: 6
                            Label { text: "🧠 模型 & 检�?; color: textPrimary; font.bold: true }
                            GridLayout {
                                columns: 2; rowSpacing: 4; columnSpacing: 6
                                Repeater {
                                    model: [
                                        {k:"模型版本", v:"YOLOv8n"},
                                        {k:"推理时间", v: aiState.inferenceMs.toFixed(1)+"ms"},
                                        {k:"FPS", v: aiState.fps.toFixed(1)},
                                        {k:"检测精�?, v: aiState.accuracy.toFixed(1)+"%"},
                                        {k:"总检测数", v: aiState.total},
                                        {k:"今日检�?, v: aiState.today}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 11 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: textPrimary; font.pixelSize: 12 }
                                    }
                                }
                            }
                        }
                    }
                    Rectangle {
                        width: parent.width; radius: 10; color: bgPanel; border.color: border
                        Column {
                            anchors.fill: parent; anchors.margins: 8; spacing: 6
                            Label { text: "📈 通信统计"; color: textPrimary; font.bold: true }
                            GridLayout {
                                columns: 2; rowSpacing: 4; columnSpacing: 6
                                Repeater {
                                    model: [
                                        {k:"连接时长", v:"2h 15m"},
                                        {k:"数据�?, v:"15,432"},
                                        {k:"错误�?, v:"0.02%"},
                                        {k:"吞吐", v:"1.2KB/s"},
                                        {k:"Wi-Fi", v:`${wifiState.ssid} (${wifiState.mode})`},
                                        {k:"信号", v: wifiState.rssi + " dBm"}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 11 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: textPrimary; font.pixelSize: 12 }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // footer
            Rectangle {
                Layout.row: 2
                Layout.columnSpan: 2
                Layout.fillWidth: true
                Layout.preferredHeight: 80
                radius: 10
                color: bgPanel
                border.color: border
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 10
                    RowLayout {
                        spacing: 8
                        Button { text: "启动系统"; onClicked: { modbus.setVisionCommAck(1); modbus.setVisionStatus(1); modbus.setVisionTransferResult(0); } }
                        Button { text: "暂停"; onClicked: { modbus.setVisionStatus(3); } }
                        Button { text: "停止"; onClicked: { modbus.setVisionStatus(3); modbus.setVisionCommAck(0); } }
                    }
                    Item { Layout.fillWidth: true }
                    ColumnLayout {
                        spacing: 4
                        Label {
                            text: "当前工序: " + ["进料检�?,"视觉识别","坐标传输","切割准备","执行切割"][Math.max(0, Math.min(4, (modbus.plcCommand||1)-1))]
                            color: textSecondary; font.pixelSize: 13
                        }
                        Label { text: "上次切割 14:25:33 | 今日切割 " + aiState.today + " | 效率 94.2%"; color: textSecondary; font.pixelSize: 12 }
                    }
                    RowLayout {
                        spacing: 8
                        Button { text: "🚨 紧急停�?; onClicked: { modbus.setVisionStatus(2); modbus.setVisionCommAck(0); modbus.setVisionTransferResult(2); }; background: Rectangle { color: error; radius: 6 } }
                        Button { text: "�?关机"; onClicked: settingsDialog.open() }
                    }
                }
            }
        }
    }

    MediaPlayer {
        id: mediaPlayer
        autoPlay: true
        loops: MediaPlayer.Infinite
        source: deepStream.sourceUrl
    }

    Dialog {
        id: settingsDialog
        modal: true
        focus: true
        x: (root.width - width) / 2
        y: (root.height - height) / 2
        standardButtons: Dialog.Close
        title: "系统设置"
        contentItem: ColumnLayout {
            spacing: 8
            width: 700
            GroupBox {
                title: "电源控制"
                Layout.fillWidth: true
                RowLayout {
                    spacing: 8
                    Button { text: "重启 Jetson"; onClicked: console.log("reboot (todo: hook systemd)") }
                    Button { text: "安全关机"; onClicked: console.log("shutdown (todo: hook systemd)") }
                    Button { text: "重启推理/Modbus 服务"; onClicked: console.log("restart services") }
                }
            }
            GroupBox {
                title: "性能模式"
                Layout.fillWidth: true
                RowLayout {
                    spacing: 8
                    Button { text: "10W"; onClicked: jetsonState.setPerfMode("10W") }
                    Button { text: "15W"; onClicked: jetsonState.setPerfMode("15W") }
                    Button { text: "MAXN"; onClicked: jetsonState.setPerfMode("MAXN") }
                    Label { text: "当前: " + jetsonState.perfMode; color: textSecondary }
                }
            }
            GroupBox {
                title: "Modbus 连接"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2
                    rowSpacing: 6
                    columnSpacing: 10
                    Label { text: "IP"; color: textSecondary }
                    TextField { id: mbHost; text: "127.0.0.1" }
                    Label { text: "端口"; color: textSecondary }
                    TextField { id: mbPort; text: "1502" }
                    Label { text: "Slave ID"; color: textSecondary }
                    TextField { id: mbSlave; text: "1" }
                }
                RowLayout {
                    spacing: 8
                    Button { text: "连接"; onClicked: modbus.configure(mbHost.text, parseInt(mbPort.text), parseInt(mbSlave.text)) }
                    Label { text: modbus.connected ? "已连�? : "未连�?; color: modbus.connected ? success : error }
                }
            }
            GroupBox {
                title: "Wi-Fi 配置"
                Layout.fillWidth: true
                GridLayout {
                    columns: 2
                    rowSpacing: 6
                    columnSpacing: 10
                    Label { text: "SSID"; color: textSecondary }
                    TextField { id: ssidField; text: wifiState.ssid; placeholderText: "SSID" }
                    Label { text: "密码"; color: textSecondary }
                    TextField { id: pwdField; text: "******"; echoMode: TextInput.Password }
                    Label { text: "模式"; color: textSecondary }
                    ComboBox { id: modeBox; model: ["DHCP","STATIC"]; currentIndex: wifiState.mode === "STATIC" ? 1 : 0 }
                    Label { text: "IP"; color: textSecondary }
                    TextField { id: ipField; text: "192.168.1.120"; enabled: modeBox.currentText === "STATIC" }
                    Label { text: "掩码"; color: textSecondary }
                    TextField { id: maskField; text: "255.255.255.0"; enabled: modeBox.currentText === "STATIC" }
                    Label { text: "网关"; color: textSecondary }
                    TextField { id: gwField; text: "192.168.1.1"; enabled: modeBox.currentText === "STATIC" }
                    Label { text: "DNS"; color: textSecondary }
                    TextField { id: dnsField; text: "223.5.5.5"; enabled: modeBox.currentText === "STATIC" }
                }
                RowLayout {
                    spacing: 8
                    Button { text: "应用"; onClicked: wifiState.apply(ssidField.text, pwdField.text, modeBox.currentText, ipField.text, maskField.text, gwField.text, dnsField.text) }
                    Button { text: "检�?; onClicked: wifiState.check() }
                    Label { text: "状�? " + wifiState.status + " | RSSI " + wifiState.rssi + " dBm"; color: textSecondary }
                }
            }
        }
    }
}

