import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12

Dialog {
    id: root
    title: "Detection Percent"
    modal: true
    standardButtons: Dialog.Ok | Dialog.Cancel
    width: 340
    anchors.centerIn: Overlay.overlay

    // Real-valued 0-100; the C++ side (CircleDetector) clamps to
    // circle_tracker_config.h's kMinimumDetectionDensity/kMaximumDetectionDensity.
    property alias value: spinBox.value

    Theme { id: theme }

    contentItem: Column {
        spacing: 12
        width: root.width - 48

        Label {
            text: "Minimum occupied-pixel density required for a marker to count as detected:"
            color: theme.text
            wrapMode: Text.WordWrap
            width: parent.width
        }

        SpinBox {
            id: spinBox
            from: 1
            to: 100
            stepSize: 1
            value: 50
            editable: true
            width: parent.width

            textFromValue: function(value) { return value + " %" }
            valueFromText: function(text) { return parseInt(text); }
        }
    }
}
