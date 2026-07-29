import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import EBts 1.0

// Live force/torque view for the Wittenstein HEX21 sensor: a numeric readout
// plus two stacked rolling graphs (forces in N, torques in mNm). Data arrives
// via Main.qml's Connections on ftEvents at the worker's ~60 Hz display rate;
// the CSV recording is driven separately, in lockstep with the camera .raw.
Rectangle {
    id: root
    color: theme.background

    signal closed()
    property bool connected: false

    function addSample(fx, fy, fz, mx, my, mz, temp) {
        forcesGraph.addSample(fx, fy, fz)
        torquesGraph.addSample(mx, my, mz)
        readout.text =
            "Fx " + fx.toFixed(2) + "  Fy " + fy.toFixed(2) + "  Fz " + fz.toFixed(2) + " N        " +
            "Mx " + mx.toFixed(1) + "  My " + my.toFixed(1) + "  Mz " + mz.toFixed(1) + " mNm        " +
            "T " + temp.toFixed(1) + " °C"
    }

    function setConnected(isConnected, info) {
        root.connected = isConnected
        statusLabel.text = (info && info.length > 0) ? info : (isConnected ? "connected" : "not connected")
    }

    Theme { id: theme }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            height: 30
            color: theme.surface

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 10
                anchors.rightMargin: 4
                spacing: 8

                Rectangle {
                    width: 10
                    height: 10
                    radius: 5
                    color: root.connected ? theme.accent : theme.danger
                }
                Label {
                    text: "Force / Torque"
                    color: theme.text
                    font.bold: true
                }
                Label {
                    id: statusLabel
                    text: "not connected"
                    color: theme.mutedText
                }
                Item { Layout.fillWidth: true }
                ToolButton {
                    text: "✕"
                    onClicked: root.closed()
                }
            }
        }

        Label {
            id: readout
            Layout.fillWidth: true
            leftPadding: 10
            topPadding: 4
            bottomPadding: 4
            text: "waiting for data…"
            color: theme.mutedText
            font.family: "monospace"
            font.pixelSize: 12
        }

        FtGraphView {
            id: forcesGraph
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Force (N)"
            seriesNames: ["Fx", "Fy", "Fz"]
        }

        FtGraphView {
            id: torquesGraph
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Torque (mNm)"
            seriesNames: ["Mx", "My", "Mz"]
        }
    }
}
