import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtMultimedia 5.15

ApplicationWindow {
    id: root
    visible: true
    width: 1366
    height: 768
    color: "#0f131a"
    title: qsTr("AI Bamboo Cutter UI")

    readonly property color bgPanel: "#161b24"
    readonly property color bgSpot: "#1f2632"
    readonly property color borderColor: "#2a3342"
    readonly property color accent: "#ff7a3d"
    readonly property color success: "#4caf50"
    readonly property color warning: "#f2c14f"
    readonly property color error: "#f44336"
    readonly property color textPrimary: "#e8ecf3"
    readonly property color textSecondary: "#9aa4b5"

    Component.onCompleted: {
        modbus.configure("127.0.0.1", 1502, 1)
    }

    Rectangle {
        anchors.fill: parent
        color: "transparent"
        anchors.margins: 12

        ColumnLayout {
            anchors.fill: parent
            spacing: 10

            // Header
            RowLayout {
                Layout.fillWidth: true
                spacing: 10
                Label { text: "AI Bamboo Cutter"; color: textPrimary; font.pixelSize: 20; font.bold: true }
                Item { Layout.fillWidth: true }
                Label { text: "Heartbeat: " + modbus.heartbeat; color: success; font.pixelSize: 12 }
                Label { text: modbus.connected ? "Modbus Connected" : "Modbus Disconnected"; color: modbus.connected ? success : error; font.pixelSize: 12 }
                Button { text: "Settings"; onClicked: settingsDialog.open() }
            }

            // Video / stats row
            RowLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 10

                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    radius: 8
                    color: bgPanel
                    border.color: borderColor
                    ColumnLayout {
                        anchors.fill: parent
                        anchors.margins: 10
                        spacing: 8
                        Label { text: "Live camera feed (" + deepStream.sourceUrl + ")"; color: accent; font.pixelSize: 16; font.bold: true }
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            radius: 6
                            color: "#000"
                            border.color: borderColor
                            // Placeholder for embedded video; DeepStreamRunner provides RTSP URL for external player
                            Label { anchors.centerIn: parent; text: "Video preview (RTSP)"; color: textSecondary }
                        }
                    }
                }

                Rectangle {
                    Layout.preferredWidth: 420
                    Layout.fillHeight: true
                    radius: 8
                    color: bgPanel
                    border.color: borderColor
                    Flickable {
                        anchors.fill: parent
                        contentWidth: parent.width
                        contentHeight: infoColumn.implicitHeight
                        clip: true
                        Column {
                            id: infoColumn
                            width: parent.width
                            spacing: 10
                            anchors.margins: 10

                            // PLC -> Camera
                            Column {
                                spacing: 4
                                Label { text: "PLC -> Camera (D2100 block)"; color: textPrimary; font.bold: true }
                                Repeater {
                                    model: [
                                        {k:"D2100 power request", v: modbus.plcPowerRequest},
                                        {k:"D2101 receive state", v: modbus.plcReceiveState},
                                        {k:"D2102/3 servo pos", v: modbus.plcServoPosition.toFixed(1)},
                                        {k:"D2104/5 coord fb", v: modbus.plcCoordFeedback.toFixed(1)}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 12 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: accent; font.pixelSize: 12; font.family: "Consolas" }
                                    }
                                }
                            }

                            // Camera -> PLC
                            Column {
                                spacing: 4
                                Label { text: "Camera -> PLC (D2000 block)"; color: textPrimary; font.bold: true }
                                Repeater {
                                    model: [
                                        {k:"D2000 comm ack", v: modbus.visionCommAck},
                                        {k:"D2001 camera status", v: modbus.visionStatus},
                                        {k:"D2002/3 target X", v: modbus.visionTargetCoord.toFixed(1)},
                                        {k:"D2004 transfer result", v: modbus.visionTransferResult}
                                    ]
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Label { text: modelData.k; color: textSecondary; font.pixelSize: 12 }
                                        Item { Layout.fillWidth: true }
                                        Label { text: modelData.v; color: accent; font.pixelSize: 12; font.family: "Consolas" }
                                    }
                                }
                                RowLayout {
                                    spacing: 8
                                    Button {
                                        text: "Ack OK"
                                        onClicked: {
                                            modbus.setVisionCommAck(1)
                                            modbus.setVisionStatus(1)
                                            modbus.setVisionTransferResult(0)
                                        }
                                    }
                                    Button {
                                        text: "Set Idle"
                                        onClicked: modbus.setVisionStatus(3)
                                    }
                                    Button {
                                        text: "Fail"
                                        onClicked: {
                                            modbus.setVisionStatus(3)
                                            modbus.setVisionCommAck(0)
                                            modbus.setVisionTransferResult(2)
                                        }
                                    }
                                }
                            }

                            // Workflow
                            Column {
                                spacing: 4
                                Label { text: "Workflow"; color: textPrimary; font.bold: true }
                                Label { text: "Workflow step: " + ["Step1","Step2","Step3","Step4","Step5"][Math.max(0, Math.min(4, (modbus.plcCommand||1)-1))]; color: textSecondary; font.pixelSize: 13 }
                                Label { text: "Last event placeholder"; color: textSecondary; font.pixelSize: 12 }
                                RowLayout {
                                    spacing: 8
                                    Button {
                                        text: "Mark NG"
                                        onClicked: {
                                            modbus.setVisionStatus(2)
                                            modbus.setVisionCommAck(0)
                                            modbus.setVisionTransferResult(2)
                                        }
                                        background: Rectangle { color: error; radius: 6 }
                                    }
                                    Button { text: "Settings"; onClicked: settingsDialog.open() }
                                }
                            }

                            // WiFi
                            Column {
                                spacing: 4
                                Label { text: "Wi-Fi"; color: textPrimary; font.bold: true }
                                GridLayout {
                                    columns: 2
                                    columnSpacing: 8
                                    rowSpacing: 6
                                    Label { text: "SSID"; color: textSecondary }
                                    TextField { id: ssidField; text: wifiState.ssid; placeholderText: "SSID" }
                                    Label { text: "Password"; color: textSecondary }
                                    TextField { id: pwdField; text: "******"; echoMode: TextInput.Password }
                                    Label { text: "Mode"; color: textSecondary }
                                    ComboBox { id: modeBox; model: ["DHCP","STATIC"]; currentIndex: 0 }
                                    Label { text: "IP"; color: textSecondary }
                                    TextField { id: ipField; text: "192.168.1.120"; enabled: modeBox.currentText === "STATIC" }
                                    Label { text: "Mask"; color: textSecondary }
                                    TextField { id: maskField; text: "255.255.255.0"; enabled: modeBox.currentText === "STATIC" }
                                    Label { text: "Gateway"; color: textSecondary }
                                    TextField { id: gwField; text: "192.168.1.1"; enabled: modeBox.currentText === "STATIC" }
                                    Label { text: "DNS"; color: textSecondary }
                                    TextField { id: dnsField; text: "223.5.5.5"; enabled: modeBox.currentText === "STATIC" }
                                    Button { text: "Apply WiFi"; onClicked: wifiState.apply(ssidField.text, pwdField.text, modeBox.currentText, ipField.text, maskField.text, gwField.text, dnsField.text) }
                                    Button { text: "Check WiFi"; onClicked: wifiState.check() }
                                    Label { text: "Status " + wifiState.status + " | RSSI " + wifiState.rssi + " dBm"; color: textSecondary }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Settings dialog placeholder
    Dialog {
        id: settingsDialog
        modal: true
        standardButtons: Dialog.Ok
        title: "Settings"
        contentItem: ColumnLayout {
            anchors.fill: parent
            anchors.margins: 10
            Label { text: "Settings placeholder"; color: textSecondary }
        }
    }
}
