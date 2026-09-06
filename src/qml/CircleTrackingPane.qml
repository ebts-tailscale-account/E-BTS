import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12
import EBts 1.0

Rectangle {
    id: root
    color: "black"

    signal closed()
    signal rebuildBaseline()
    signal reloadCalibration()
    signal frankaLinkRequested(string url)
    signal frankaUnlinkRequested()

    function setFrame(frame) {
        frameView.setFrame(frame)
    }

    function setContactEstimate(valid, ambiguous, hasMm, xMm, yMm, divergence, trackedMarkers) {
        contactReadout.estimateValid = valid
        contactReadout.estimateAmbiguous = ambiguous
        contactReadout.estimateHasMm = hasMm
        contactReadout.estimateX = xMm
        contactReadout.estimateY = yMm
        contactReadout.divergence = divergence
        contactReadout.trackedMarkers = trackedMarkers
    }

    function setCalibrationStatus(ready, hasOrigin, summary) {
        contactReadout.calibrationReady = ready && hasOrigin
        contactReadout.calibrationSummary = summary
    }

    function setFrankaPose(xMm, yMm) {
        contactReadout.frankaX = xMm
        contactReadout.frankaY = yMm
    }

    function setFrankaStatus(ok, message) {
        contactReadout.frankaOk = ok
        contactReadout.frankaStatus = message
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

                Label {
                    text: "Circle Tracking"
                    color: theme.text
                    font.bold: true
                }
                Item { Layout.fillWidth: true }
                ToolButton {
                    text: "Rebuild Baseline"
                    ToolTip.visible: hovered
                    ToolTip.text: "Recollect the marker baseline; keep the sensor unloaded while it runs."
                    onClicked: root.rebuildBaseline()
                }
                // The Franka link and the URL live in the readout below, beside the
                // value they control -- putting them up here overflowed the header
                // on a narrow window and pushed the close button off screen.
                ToolButton {
                    text: "Reload Cal."
                    ToolTip.visible: hovered
                    ToolTip.text: "Re-read calibration/pixel_to_mm.json and calibration/elastomer_origin.json "
                                  + "(e.g. after teaching a new elastomer origin on tactile)."
                    onClicked: root.reloadCalibration()
                }
                ToolButton {
                    text: "✕"
                    onClicked: root.closed()
                }
            }
        }

        FrameView {
            id: frameView
            Layout.fillWidth: true
            Layout.fillHeight: true
        }

        ContactReadout {
            id: contactReadout
            Layout.fillWidth: true
            onLinkRequested: root.frankaLinkRequested(url)
            onUnlinkRequested: root.frankaUnlinkRequested()
        }
    }
}
