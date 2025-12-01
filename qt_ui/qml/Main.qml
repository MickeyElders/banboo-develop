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

    title: qsTr("AI绔硅妭璇嗗埆鍒囧壊绯荤粺 v2.1 - Qt Preview")



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

                        Label { text: "AI绔硅妭璇嗗埆鍒囧壊绯荤粺 v2.1"; color: textPrimary; font.bold: true; font.pixelSize: 18 }

                        Rectangle {

                            radius: 6; color: "#1b88ff"

                            anchors.verticalCenter: parent.verticalCenter

                            width: implicitWidth; height: 20; anchors.margins: 0

                            Row {

                                anchors.centerIn: parent

                                spacing: 4

                                anchors.margins: 6

                                Label { text: "Jetson Orin NX 路 Modbus TCP"; color: "#031422"; font.pixelSize: 11; font.bold: true }

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

                                    text: ["Step1 杩涙枡妫€娴?,"Step2 瑙嗚璇嗗埆","Step3 鍧愭爣浼犺緭","Step4 鍒囧壊鍑嗗","Step5 鎵ц鍒囧壊"][index]
                                }

                            }

                        }

                    }

                    Button {

                        text: "鈿欙笍 璁剧疆"

                        onClicked: settingsDialog.open()

                    }

                    RowLayout {

                        spacing: 6

                        Rectangle { width: 12; height: 12; radius: 6; color: success; anchors.verticalCenter: parent.verticalCenter; opacity: 0.8 }

                        Label { text: "蹇冭烦 " + modbus.heartbeat; color: success; font.pixelSize: 12 }

                        Label { text: "鍝嶅簲 12ms"; color: textSecondary; font.pixelSize: 12 }

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
                        Label { text: "Live camera feed"; color: accent; font.pixelSize: 16; font.bold: true }
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

                            Label { anchors.centerIn: parent; text: "X杞村锟?(0 - 1000.0 mm)"; color: "#2196f3"; font.pixelSize: 12 }

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

                                    {label: "X鍧愭爣", value: Qt.formatNumber(modbus.visionTargetCoord || systemState.xCoordinate, 'f', 1) + "mm"},

                                    {label: "PLC鎺ユ敹鐘讹拷?, value: modbus.plcReceiveState === 1 ? "鍙帴锟? : (modbus.plcReceiveState === 2 ? "閫佹枡锟? : "鏈煡")},

                                    {label: "閫佹枡浼烘湇褰撳墠浣嶇疆", value: modbus.plcServoPosition.toFixed(1) + " mm"}

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

                                Label { text: "馃搳 Modbus"; color: textPrimary; font.bold: true }

                                Item { Layout.fillWidth: true }

                                Label { text: modbus.connected ? "宸茶繛锟? : "鏈繛锟?; color: modbus.connected ? success : error; font.pixelSize: 12 }

                            }

                            Label { text: "PLC 锟?鐩告満 (D2100 绯诲垪)"; color: textSecondary; font.pixelSize: 12 }

                            GridLayout {

                                columns: 2; rowSpacing: 4; columnSpacing: 6

                                Repeater {

                                    model: [

                                        {k:"D2100 閫氳璇锋眰",v: modbus.plcPowerRequest},

                                        {k:"D2101 鎺ユ敹鐘讹拷?,v: modbus.plcReceiveState},

                                        {k:"D2102/3 閫佹枡浼烘湇",v: modbus.plcServoPosition.toFixed(1)},

                                        {k:"D2104/5 鍧愭爣鍙嶉",v: modbus.plcCoordFeedback.toFixed(1)}

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
                                        {k:"D2000 閫氳搴旂瓟",v: modbus.visionCommAck},
                                        {k:"D2001 鐩告満鐘舵€?,v: modbus.visionStatus},
                                        {k:"D2002/3 鐩爣鍧愭爣",v: modbus.visionTargetCoord.toFixed(1)},
                                        {k:"D2004 浼犺緭缁撴灉",v: modbus.visionTransferResult}
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

                            Label { text: "馃煝 Jetson Orin NX"; color: textPrimary; font.bold: true }

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

                                bar("鍐呭瓨", (jetsonState.memUsed/jetsonState.memTotal*100).toFixed(1)+"%", warning)

                                bar("娓╁害", jetsonState.temp.toFixed(0)+"%", error)

                            }

                            GridLayout {

                                columns: 2; rowSpacing: 4; columnSpacing: 6

                                Repeater {

                                    model: [

                                        {k:"鍔燂拷?, v: jetsonState.cpuUsage.toFixed(0)+" W"},

                                        {k:"椋庢墖", v: jetsonState.fanRpm.toFixed(0)+" RPM"},

                                        {k:"鎬ц兘妯″紡", v: jetsonState.perfMode},

                                        {k:"X11", v:"宸茬锟?}

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

                            Label { text: "馃 妯″瀷 & 妫€锟?; color: textPrimary; font.bold: true }

                            GridLayout {

                                columns: 2; rowSpacing: 4; columnSpacing: 6

                                Repeater {

                                    model: [

                                        {k:"妯″瀷鐗堟湰", v:"YOLOv8n"},

                                        {k:"鎺ㄧ悊鏃堕棿", v: aiState.inferenceMs.toFixed(1)+"ms"},

                                        {k:"FPS", v: aiState.fps.toFixed(1)},

                                        {k:"妫€娴嬬簿锟?, v: aiState.accuracy.toFixed(1)+"%"},

                                        {k:"鎬绘娴嬫暟", v: aiState.total},

                                        {k:"浠婃棩妫€锟?, v: aiState.today}

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

                            Label { text: "馃搱 閫氫俊缁熻"; color: textPrimary; font.bold: true }

                            GridLayout {

                                columns: 2; rowSpacing: 4; columnSpacing: 6

                                Repeater {

                                    model: [

                                        {k:"杩炴帴鏃堕暱", v:"2h 15m"},

                                        {k:"鏁版嵁锟?, v:"15,432"},

                                        {k:"閿欒锟?, v:"0.02%"},

                                        {k:"鍚炲悙", v:"1.2KB/s"},

                                        {k:"Wi-Fi", v:`${wifiState.ssid} (${wifiState.mode})`},

                                        {k:"淇″彿", v: wifiState.rssi + " dBm"}

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

                        Button { text: "鍚姩绯荤粺"; onClicked: { modbus.setVisionCommAck(1); modbus.setVisionStatus(1); modbus.setVisionTransferResult(0); } }

                        Button { text: "鏆傚仠"; onClicked: { modbus.setVisionStatus(3); } }

                        Button { text: "鍋滄"; onClicked: { modbus.setVisionStatus(3); modbus.setVisionCommAck(0); } }

                    }

                    Item { Layout.fillWidth: true }

                    ColumnLayout {

                        spacing: 4

                        Label {

                            text: "褰撳墠宸ュ簭: " + ["杩涙枡妫€锟?,"瑙嗚璇嗗埆","鍧愭爣浼犺緭","鍒囧壊鍑嗗","鎵ц鍒囧壊"][Math.max(0, Math.min(4, (modbus.plcCommand||1)-1))]

                            color: textSecondary; font.pixelSize: 13

                        }

                        Label { text: "涓婃鍒囧壊 14:25:33 | 浠婃棩鍒囧壊 " + aiState.today + " | 鏁堢巼 94.2%"; color: textSecondary; font.pixelSize: 12 }

                    }

                    RowLayout {

                        spacing: 8

                        Button { text: "馃毃 绱ф€ュ仠锟?; onClicked: { modbus.setVisionStatus(2); modbus.setVisionCommAck(0); modbus.setVisionTransferResult(2); }; background: Rectangle { color: error; radius: 6 } }

                        Button { text: "锟?鍏虫満"; onClicked: settingsDialog.open() }

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

        title: "绯荤粺璁剧疆"

        contentItem: ColumnLayout {

            spacing: 8

            width: 700

            GroupBox {

                title: "鐢垫簮鎺у埗"

                Layout.fillWidth: true

                RowLayout {

                    spacing: 8

                    Button { text: "閲嶅惎 Jetson"; onClicked: console.log("reboot (todo: hook systemd)") }

                    Button { text: "瀹夊叏鍏虫満"; onClicked: console.log("shutdown (todo: hook systemd)") }

                    Button { text: "閲嶅惎鎺ㄧ悊/Modbus 鏈嶅姟"; onClicked: console.log("restart services") }

                }

            }

            GroupBox {

                title: "鎬ц兘妯″紡"

                Layout.fillWidth: true

                RowLayout {

                    spacing: 8

                    Button { text: "10W"; onClicked: jetsonState.setPerfMode("10W") }

                    Button { text: "15W"; onClicked: jetsonState.setPerfMode("15W") }

                    Button { text: "MAXN"; onClicked: jetsonState.setPerfMode("MAXN") }

                    Label { text: "褰撳墠: " + jetsonState.perfMode; color: textSecondary }

                }

            }

            GroupBox {

                title: "Modbus 杩炴帴"

                Layout.fillWidth: true

                GridLayout {

                    columns: 2

                    rowSpacing: 6

                    columnSpacing: 10

                    Label { text: "IP"; color: textSecondary }

                    TextField { id: mbHost; text: "127.0.0.1" }

                    Label { text: "绔彛"; color: textSecondary }

                    TextField { id: mbPort; text: "1502" }

                    Label { text: "Slave ID"; color: textSecondary }

                    TextField { id: mbSlave; text: "1" }

                }

                RowLayout {

                    spacing: 8

                    Button { text: "杩炴帴"; onClicked: modbus.configure(mbHost.text, parseInt(mbPort.text), parseInt(mbSlave.text)) }

                    Label { text: modbus.connected ? "宸茶繛锟? : "鏈繛锟?; color: modbus.connected ? success : error }

                }

            }

            GroupBox {

                title: "Wi-Fi 閰嶇疆"

                Layout.fillWidth: true

                GridLayout {

                    columns: 2

                    rowSpacing: 6

                    columnSpacing: 10

                    Label { text: "SSID"; color: textSecondary }

                    TextField { id: ssidField; text: wifiState.ssid; placeholderText: "SSID" }

                    Label { text: "瀵嗙爜"; color: textSecondary }

                    TextField { id: pwdField; text: "******"; echoMode: TextInput.Password }

                    Label { text: "妯″紡"; color: textSecondary }

                    ComboBox { id: modeBox; model: ["DHCP","STATIC"]; currentIndex: wifiState.mode === "STATIC" ? 1 : 0 }

                    Label { text: "IP"; color: textSecondary }

                    TextField { id: ipField; text: "192.168.1.120"; enabled: modeBox.currentText === "STATIC" }

                    Label { text: "鎺╃爜"; color: textSecondary }

                    TextField { id: maskField; text: "255.255.255.0"; enabled: modeBox.currentText === "STATIC" }

                    Label { text: "缃戝叧"; color: textSecondary }

                    TextField { id: gwField; text: "192.168.1.1"; enabled: modeBox.currentText === "STATIC" }

                    Label { text: "DNS"; color: textSecondary }

                    TextField { id: dnsField; text: "223.5.5.5"; enabled: modeBox.currentText === "STATIC" }

                }

                RowLayout {

                    spacing: 8

                    Button { text: "搴旂敤"; onClicked: wifiState.apply(ssidField.text, pwdField.text, modeBox.currentText, ipField.text, maskField.text, gwField.text, dnsField.text) }

                    Button { text: "妫€锟?; onClicked: wifiState.check() }

                    Label { text: "鐘讹拷? " + wifiState.status + " | RSSI " + wifiState.rssi + " dBm"; color: textSecondary }

                }

            }

        }

    }

}




