import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12

Dialog {
    id: root
    title: "Accumulation Time"
    modal: true
    standardButtons: Dialog.Ok | Dialog.Cancel
    width: 340
    anchors.centerIn: Overlay.overlay

    // Same shared range as circle_tracker_config.h's kMinimumCollectionTimeUs
    // / kMaximumCollectionTimeUs -- the C++ side clamps to this range too, so
    // keep both in sync if either changes.
    property alias value: spinBox.value

    Theme { id: theme }

    contentItem: Column {
        spacing: 12
        width: root.width - 48

        Label {
            text: "Microseconds per accumulation window (shared by the Camera feed and Circle Tracking):"
            color: theme.text
            wrapMode: Text.WordWrap
            width: parent.width
        }

        SpinBox {
            id: spinBox
            from: 1000
            to: 100000
            stepSize: 1000
            value: 10000
            editable: true
            width: parent.width
        }
    }
}
