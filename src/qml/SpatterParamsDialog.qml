import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.12

// Tuning for the spatter tracker. Every field maps one-to-one onto a command
// line option of Prophesee's metavision_spatter_tracking sample, and the
// defaults below are that sample's, so its documentation reads across directly.
// The C++ side clamps each value to the range in spatter_tracker_config.h.
Dialog {
    id: root
    title: "Spatter Tracking Parameters"
    modal: true
    standardButtons: Dialog.Ok | Dialog.Cancel
    width: 460
    anchors.centerIn: Overlay.overlay

    property alias cellWidth: cellWidthBox.value
    property alias cellHeight: cellHeightBox.value
    property alias activationThreshold: activationBox.value
    property alias minSize: minSizeBox.value
    property alias maxSize: maxSizeBox.value
    property alias maxDistance: maxDistanceBox.value
    property alias untrackedThreshold: untrackedBox.value
    property alias roiX: roiXBox.value
    property alias roiY: roiYBox.value
    property alias roiWidth: roiWidthBox.value
    property alias roiHeight: roiHeightBox.value

    Theme { id: theme }

    contentItem: ColumnLayout {
        spacing: 8

        Label {
            Layout.fillWidth: true
            Layout.bottomMargin: 4
            text: "Objects are found by activating grid cells, grouping neighbouring active " +
                  "cells, then matching the groups to the previous window's tracks. " +
                  "Window length comes from the shared Accumulation Time setting."
            color: theme.mutedText
            wrapMode: Text.WordWrap
            font.pixelSize: 12
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 12
            rowSpacing: 6

            Label {
                text: "Cell width (px)"
                color: theme.text
                ToolTip.visible: hoverHandlerCellWidth.hovered
                ToolTip.text: "Grid resolution. Smaller cells separate nearby objects but need a lower activation threshold."
                HoverHandler { id: hoverHandlerCellWidth }
            }
            SpinBox {
                id: cellWidthBox
                Layout.fillWidth: true
                from: 1; to: 128; value: 7; editable: true
            }

            Label { text: "Cell height (px)"; color: theme.text }
            SpinBox {
                id: cellHeightBox
                Layout.fillWidth: true
                from: 1; to: 128; value: 7; editable: true
            }

            Label {
                text: "Activation threshold"
                color: theme.text
                ToolTip.visible: hoverHandlerActivation.hovered
                ToolTip.text: "Distinct pixels that must fire inside a cell for it to count as active. Repeat events on one pixel count once."
                HoverHandler { id: hoverHandlerActivation }
            }
            SpinBox {
                id: activationBox
                Layout.fillWidth: true
                from: 1; to: 4096; value: 5; editable: true
            }

            Label {
                text: "Min object size (px)"
                color: theme.text
                ToolTip.visible: hoverHandlerMinSize.hovered
                ToolTip.text: "Both bounding-box sides must be at least this. A long thin streak is rejected on its short side."
                HoverHandler { id: hoverHandlerMinSize }
            }
            SpinBox {
                id: minSizeBox
                Layout.fillWidth: true
                from: 1; to: 4096; value: 10; editable: true
            }

            Label { text: "Max object size (px)"; color: theme.text }
            SpinBox {
                id: maxSizeBox
                Layout.fillWidth: true
                from: 1; to: 4096; value: 300; editable: true
            }

            Label {
                text: "Max association distance (px)"
                color: theme.text
                ToolTip.visible: hoverHandlerDistance.hovered
                ToolTip.text: "Must exceed how far an object travels in one window, but stay below the spacing between objects or tracks swap ids."
                HoverHandler { id: hoverHandlerDistance }
            }
            SpinBox {
                id: maxDistanceBox
                Layout.fillWidth: true
                from: 1; to: 4096; value: 50; editable: true
            }

            Label {
                text: "Untracked threshold (windows)"
                color: theme.text
                ToolTip.visible: hoverHandlerUntracked.hovered
                ToolTip.text: "How many consecutive windows a track may go unmatched before it is retired."
                HoverHandler { id: hoverHandlerUntracked }
            }
            SpinBox {
                id: untrackedBox
                Layout.fillWidth: true
                from: 0; to: 256; value: 5; editable: true
            }
        }

        // Region of interest. Not a metavision_spatter_tracking option -- it is
        // specific to this rig, where only the marker band matters -- so it is
        // separated from the fields above rather than mixed in with them.
        Label {
            Layout.fillWidth: true
            Layout.topMargin: 8
            text: "Region of interest"
            color: theme.text
            font.bold: true
        }

        Label {
            Layout.fillWidth: true
            text: "Events outside this rectangle are dropped before tracking, so nothing " +
                  "outside it can start a track. Full-sensor coordinates. A rectangle that " +
                  "overhangs an edge is clipped to the sensor, not rejected. Affects Spatter " +
                  "Tracking only -- the live view and Circle Tracking still see everything."
            color: theme.mutedText
            wrapMode: Text.WordWrap
            font.pixelSize: 12
        }

        GridLayout {
            Layout.fillWidth: true
            columns: 2
            columnSpacing: 12
            rowSpacing: 6

            Label { text: "ROI x (px)"; color: theme.text }
            SpinBox {
                id: roiXBox
                Layout.fillWidth: true
                from: 0; to: 4096; value: 6; editable: true
            }

            Label { text: "ROI y (px)"; color: theme.text }
            SpinBox {
                id: roiYBox
                Layout.fillWidth: true
                from: 0; to: 4096; value: 181; editable: true
            }

            Label { text: "ROI width (px)"; color: theme.text }
            SpinBox {
                id: roiWidthBox
                Layout.fillWidth: true
                from: 1; to: 4096; value: 640; editable: true
            }

            Label { text: "ROI height (px)"; color: theme.text }
            SpinBox {
                id: roiHeightBox
                Layout.fillWidth: true
                from: 1; to: 4096; value: 64; editable: true
            }
        }
    }
}
