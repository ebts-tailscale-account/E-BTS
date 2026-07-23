import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.12

// Always-on-top toolbar, VSCode-style: never disappears once a source is
// open, gains contextual buttons depending on which sources are open.
Rectangle {
    id: root
    color: theme.surfaceElevated

    property bool connected: false
    property bool cameraOpen: false
    property bool circleTrackingOpen: false
    property bool sequenceRecordingOpen: false
    property bool recording: false

    signal toggleSource(string name)
    signal checkConnection()
    signal toggleRecording()
    signal accumulationTimeChanged(int value)
    signal detectionPercentChanged(real value)

    Theme { id: theme }

    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 1
        color: theme.background
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 14
        anchors.rightMargin: 14
        spacing: 10

        Rectangle {
            width: 10
            height: 10
            radius: 5
            color: root.connected ? theme.accent : theme.danger
        }

        Label {
            text: "E-BTS GUI"
            font.bold: true
            font.pixelSize: 16
            color: theme.text
        }

        ToolSeparator {
            Layout.fillHeight: false
            Layout.preferredHeight: 24
        }

        Button {
            id: addSourceButton
            text: "+ Add Source"
            enabled: root.connected
            highlighted: true
            onClicked: addSourceMenu.open()

            AddSourceMenu {
                id: addSourceMenu
                y: addSourceButton.height
                cameraOpen: root.cameraOpen
                circleTrackingOpen: root.circleTrackingOpen
                sequenceRecordingOpen: root.sequenceRecordingOpen
                onToggleSource: root.toggleSource(name)
            }
        }

        Button {
            visible: root.cameraOpen
            text: root.recording ? "● Stop Recording" : "● Record Sequence"
            highlighted: root.recording
            Material.background: root.recording ? theme.danger : undefined
            onClicked: root.toggleRecording()
        }

        Button {
            visible: root.cameraOpen
            text: "Accumulation Time"
            onClicked: accumulationDialog.open()

            AccumulationTimeDialog {
                id: accumulationDialog
                onAccepted: root.accumulationTimeChanged(value)
            }
        }

        Button {
            visible: root.circleTrackingOpen
            text: "Detection %"
            onClicked: detectionDialog.open()

            DetectionPercentDialog {
                id: detectionDialog
                onAccepted: root.detectionPercentChanged(value)
            }
        }

        Item {
            Layout.fillWidth: true
        }

        Button {
            visible: !root.connected
            text: "Check Connection"
            highlighted: true
            onClicked: root.checkConnection()
        }
    }
}
