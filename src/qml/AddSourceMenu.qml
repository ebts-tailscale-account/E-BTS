import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12

// Persistent popup-style menu opened from the ribbon's "+ Add Source"
// button. Not modal and not a one-shot dialog: it stays a checklist you can
// reopen at any time to toggle any of the 3 sources on/off, mirroring how a
// VSCode-style toolbar button behaves rather than a wizard.
Popup {
    id: root
    modal: false
    focus: true
    closePolicy: Popup.CloseOnPressOutside | Popup.CloseOnEscape
    width: 260

    property bool cameraOpen: false
    property bool circleTrackingOpen: false
    property bool sequenceRecordingOpen: false

    signal toggleSource(string name)

    Theme { id: theme }

    contentItem: Column {
        spacing: 0

        ItemDelegate {
            width: root.width - 2
            text: (root.cameraOpen ? "✓  " : "     ") + "Camera"
            onClicked: root.toggleSource("camera")
        }
        ItemDelegate {
            width: root.width - 2
            text: (root.circleTrackingOpen ? "✓  " : "     ") + "Circle Tracking"
            onClicked: root.toggleSource("circleTracking")
        }
        ItemDelegate {
            width: root.width - 2
            text: (root.sequenceRecordingOpen ? "✓  " : "     ") + "Sequence Recording"
            onClicked: root.toggleSource("sequenceRecording")
        }
    }
}
