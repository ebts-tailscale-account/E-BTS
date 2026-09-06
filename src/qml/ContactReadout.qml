import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Layouts 1.12

// The contact-location readout: what the camera estimates, what the robot says,
// and the difference. Three columns, always in that order, always in the same
// units and the same frame (robot base millimetres, origin at the taught centre
// of the elastomer) -- so the middle column is the answer, the left is the
// measurement, and the right is how wrong the measurement is.
//
// EVERY FIELD HAS AN EXPLICIT "NO VALUE" STATE. A contact readout that shows
// 0.00, 0.00 when it has found nothing is indistinguishable from one reporting a
// contact at the centre of the pad, and the centre is exactly where a poke is
// most likely to be. So: em dashes, never zeros.
Rectangle {
    id: root

    // Its own Theme instance rather than one passed in: Theme.qml is a QtObject of
    // constants, and a `theme: theme` binding from the parent would have the
    // property shadow the id it is trying to read.
    Theme { id: theme }

    // --- estimate ---
    property bool estimateValid: false
    property bool estimateAmbiguous: false
    property bool estimateHasMm: false
    property real estimateX: 0
    property real estimateY: 0
    property real divergence: 0
    property int trackedMarkers: 0
    property string calibrationSummary: ""
    property bool calibrationReady: false
    // --- franka ---
    property bool frankaLinked: false
    property bool frankaOk: false
    property string frankaStatus: "off"
    property real frankaX: 0
    property real frankaY: 0

    signal linkRequested(string url)
    signal unlinkRequested()

    readonly property bool deltaAvailable: estimateValid && estimateHasMm && frankaLinked && frankaOk
    readonly property real deltaX: frankaX - estimateX
    readonly property real deltaY: frankaY - estimateY
    readonly property real deltaNorm: Math.sqrt(deltaX * deltaX + deltaY * deltaY)

    implicitHeight: layout.implicitHeight + 12
    color: theme.surfaceElevated

    function mm(value) {
        return (value >= 0 ? "+" : "") + value.toFixed(2)
    }

    RowLayout {
        id: layout
        anchors.fill: parent
        anchors.margins: 6
        spacing: 18

        // ---- camera estimate -------------------------------------------------
        ColumnLayout {
            spacing: 1
            Label {
                text: "ESTIMATE (camera)"
                color: theme.mutedText
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Label {
                font.family: "monospace"
                font.pixelSize: 20
                color: root.estimateAmbiguous ? "#FFB300"
                                              : (root.estimateValid ? (theme.text)
                                                                    : (theme.mutedText))
                text: {
                    if (!root.estimateValid)
                        return "—  no contact"
                    if (!root.estimateHasMm)
                        return "—  no mm calibration"
                    return "x " + root.mm(root.estimateX) + "   y " + root.mm(root.estimateY) + " mm"
                }
            }
            Label {
                font.pixelSize: 10
                color: root.estimateAmbiguous ? "#FFB300" : (theme.mutedText)
                text: {
                    if (root.estimateAmbiguous)
                        return "two contacts resolved — this readout is single-point only"
                    if (root.estimateValid)
                        return "divergence " + root.divergence.toFixed(1) + " px/cell, "
                               + root.trackedMarkers + " markers tracked"
                    return root.trackedMarkers > 0
                           ? root.trackedMarkers + " markers tracked, no peak above threshold"
                           : "waiting for the marker baseline"
                }
            }
        }

        Rectangle { Layout.fillHeight: true; width: 1; color: theme.background }

        // ---- franka truth ----------------------------------------------------
        ColumnLayout {
            spacing: 1
            Label {
                text: "FRANKA (robot)"
                color: theme.mutedText
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Label {
                font.family: "monospace"
                font.pixelSize: 20
                color: root.frankaOk ? (theme.text)
                                     : (theme.mutedText)
                text: root.frankaLinked && root.frankaOk
                      ? "x " + root.mm(root.frankaX) + "   y " + root.mm(root.frankaY) + " mm"
                      : "—"
            }
            RowLayout {
                spacing: 6
                TextField {
                    id: urlField
                    text: "http://100.93.60.35:8732"
                    implicitWidth: 170
                    font.pixelSize: 10
                    enabled: !linkButton.checked
                    ToolTip.visible: hovered
                    ToolTip.text: "Root URL of pose_server.py on tactile (~/E-BTS/pose_server.py)."
                }
                Button {
                    id: linkButton
                    text: checked ? "Unlink" : "Link"
                    checkable: true
                    font.pixelSize: 10
                    implicitHeight: 22
                    onToggled: {
                        root.frankaLinked = checked
                        if (checked) {
                            root.linkRequested(urlField.text)
                        } else {
                            root.unlinkRequested()
                            root.frankaOk = false
                            root.frankaStatus = "off"
                        }
                    }
                }
            }
            Label {
                font.pixelSize: 10
                color: root.frankaLinked && !root.frankaOk ? (theme.danger)
                                                           : (theme.mutedText)
                text: root.frankaStatus
                elide: Text.ElideRight
                Layout.maximumWidth: 240
            }
        }

        Rectangle { Layout.fillHeight: true; width: 1; color: theme.background }

        // ---- the error -------------------------------------------------------
        ColumnLayout {
            spacing: 1
            Label {
                text: "Δ  (robot − camera)"
                color: theme.mutedText
                font.pixelSize: 10
                font.letterSpacing: 1
            }
            Label {
                font.family: "monospace"
                font.pixelSize: 20
                color: root.deltaAvailable ? (theme.accent)
                                           : (theme.mutedText)
                text: root.deltaAvailable
                      ? "dx " + root.mm(root.deltaX) + "  dy " + root.mm(root.deltaY) + " mm"
                      : "—"
            }
            Label {
                font.pixelSize: 10
                color: theme.mutedText
                // The magnitude is the number worth quoting, but it is only ever a
                // LIVE glance: the proper error table comes from the recorded
                // <run>_contact.csv against franka.csv (ml/contact_error.py), which
                // aligns the two clocks instead of comparing whatever each side
                // happened to have last.
                text: root.deltaAvailable ? "|Δ| " + root.deltaNorm.toFixed(2) + " mm"
                                          : (root.frankaLinked ? "needs a contact and a live pose"
                                                               : "link the Franka to compare")
            }
        }

        Item { Layout.fillWidth: true }

        Label {
            Layout.maximumWidth: 260
            font.pixelSize: 10
            wrapMode: Text.WordWrap
            color: root.calibrationReady ? (theme.mutedText)
                                         : (theme.danger)
            text: root.calibrationSummary
        }
    }
}
