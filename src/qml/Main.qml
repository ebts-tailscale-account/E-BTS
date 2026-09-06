import QtQuick 2.12
import QtQuick.Controls 2.12
import QtQuick.Controls.Material 2.12
import QtQuick.Layouts 1.12

ApplicationWindow {
    id: window
    visible: true
    width: 1280
    height: 800
    title: "E-BTS GUI"
    color: theme.background

    Material.theme: Material.Dark
    Material.accent: theme.accent
    Material.primary: theme.primary

    Theme { id: theme }

    property bool cameraConnected: false
    property bool cameraOpen: false
    property bool circleTrackingOpen: false
    property bool spatterTrackingOpen: false
    property bool sequenceRecordingOpen: false
    property bool forceTorqueOpen: false
    property bool recording: false

    // FrankaPoseClient lives on the GUI thread (it is just a QTimer and a
    // QNetworkAccessManager), so QML binds to it directly -- the cameraEvents
    // relay below exists only because cameraWorker does not.
    Connections {
        target: frankaPose
        onPoseReady: {
            circleTrackingPane.setFrankaPose(xMm, yMm)
        }
        onStatusChanged: {
            circleTrackingPane.setFrankaStatus(ok, message)
        }
    }

    // Qt 5.12 (this project's target) only supports Connections' older
    // "onSignalName: { ... }" property syntax, where the signal's declared
    // parameter names (e.g. "reason", "frame") are implicitly in scope --
    // NOT the "function onSignalName(args) { ... }" form, which is a Qt
    // 5.15+ addition and is silently never invoked here if used instead.
    //
    // Listens on cameraEvents (a main-thread relay), not cameraWorker
    // directly: cameraWorker lives on its own QThread so circle tracking's
    // OpenCV work never blocks the GUI thread, and QML's Connections type
    // refuses to bind to a target living outside the QML engine's thread.
    // Calling slots on cameraWorker directly (see below) is unaffected --
    // that direction is safe cross-thread.
    Connections {
        target: cameraEvents
        onConnected: {
            window.cameraConnected = true
        }
        onConnectionFailed: {
            window.cameraConnected = false
            connectionBanner.message = reason
        }
        onDisconnected: {
            window.cameraConnected = false
            window.cameraOpen = false
            window.circleTrackingOpen = false
            window.spatterTrackingOpen = false
            window.sequenceRecordingOpen = false
            connectionBanner.message = reason
        }
        onCameraFrameReady: {
            cameraPane.setFrame(frame)
        }
        onTrackingFrameReady: {
            circleTrackingPane.setFrame(frame)
        }
        onSpatterFrameReady: {
            spatterTrackingPane.setFrame(frame)
        }
        onContactEstimateReady: {
            circleTrackingPane.setContactEstimate(valid, ambiguous, hasMm, xMm, yMm, divergence, trackedMarkers)
        }
        onContactCalibrationStatus: {
            circleTrackingPane.setCalibrationStatus(ready, hasOrigin, summary)
        }
        onRecordingLogLine: {
            sequenceRecordingConsole.appendLine(line)
        }
        onRecordingStateChanged: {
            window.recording = active
        }
        onSequenceRecordingWatchStopped: {
            window.sequenceRecordingOpen = false
        }
    }

    // Wittenstein force/torque sensor -- an independent device, so it listens
    // on its own bridge (ftEvents) and is unaffected by camera connect/disconnect.
    // Same Qt 5.12 "onSignalName:" syntax constraint as above applies.
    Connections {
        target: ftEvents
        onSampleReady: {
            if (window.forceTorqueOpen) {
                forceTorquePane.addSample(fx, fy, fz, mx, my, mz, temp)
            }
        }
        onConnectionChanged: {
            if (window.forceTorqueOpen) {
                forceTorquePane.setConnected(connected, info)
            }
        }
        onRecordingStateChanged: {
            // F/T recording is driven by the camera recorder; nothing to do here
            // yet, but this is where a combined status indicator would hook in.
        }
    }

    function toggleSource(name) {
        if (name === "camera") {
            window.cameraOpen = !window.cameraOpen
            cameraWorker.setCameraViewActive(window.cameraOpen)
        } else if (name === "circleTracking") {
            window.circleTrackingOpen = !window.circleTrackingOpen
            cameraWorker.setCircleTrackingActive(window.circleTrackingOpen)
        } else if (name === "spatterTracking") {
            window.spatterTrackingOpen = !window.spatterTrackingOpen
            cameraWorker.setSpatterTrackingActive(window.spatterTrackingOpen)
        } else if (name === "sequenceRecording") {
            window.sequenceRecordingOpen = !window.sequenceRecordingOpen
            cameraWorker.setSequenceRecordingActive(window.sequenceRecordingOpen)
        } else if (name === "forceTorque") {
            window.forceTorqueOpen = !window.forceTorqueOpen
            ftWorker.setActive(window.forceTorqueOpen)
        }
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        Ribbon {
            id: ribbon
            Layout.fillWidth: true
            Layout.preferredHeight: 52
            connected: window.cameraConnected
            cameraOpen: window.cameraOpen
            circleTrackingOpen: window.circleTrackingOpen
            spatterTrackingOpen: window.spatterTrackingOpen
            sequenceRecordingOpen: window.sequenceRecordingOpen
            forceTorqueOpen: window.forceTorqueOpen
            recording: window.recording
            onToggleSource: window.toggleSource(name)
            onCheckConnection: cameraWorker.connectToCamera()
            onToggleRecording: cameraWorker.toggleManualRecording()
            onAccumulationTimeChanged: cameraWorker.setAccumulationTimeUs(value)
            onDetectionPercentChanged: cameraWorker.setDetectionPercent(value)
            onSpatterParamsChanged: cameraWorker.setSpatterParams(cellWidth, cellHeight, activationThreshold,
                                                                 minSize, maxSize, maxDistance,
                                                                 untrackedThreshold, roiX, roiY,
                                                                 roiWidth, roiHeight)
        }

        Item {
            id: content
            Layout.fillWidth: true
            Layout.fillHeight: true

            ColumnLayout {
                anchors.fill: parent
                spacing: 2

                // Camera, Circle Tracking and Spatter Tracking are three views
                // of one shared live event stream (the EVK1 only exposes one
                // open camera handle), so they split this row evenly: 50/50
                // with two open, thirds with all three, and each fills the
                // whole row alone. An invisible child contributes no space in
                // Qt Quick Layouts, which is what makes every combination work
                // out of this one row without separate layout code per case.
                RowLayout {
                    id: videoRow
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    visible: window.cameraOpen || window.circleTrackingOpen || window.spatterTrackingOpen
                    spacing: 2

                    CameraPane {
                        id: cameraPane
                        visible: window.cameraOpen
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        onClosed: window.toggleSource("camera")
                    }
                    CircleTrackingPane {
                        id: circleTrackingPane
                        visible: window.circleTrackingOpen
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        onClosed: window.toggleSource("circleTracking")
                        onRebuildBaseline: cameraWorker.requestBaselineRestart()
                        onReloadCalibration: cameraWorker.reloadContactCalibration()
                        onFrankaLinkRequested: frankaPose.start(url)
                        onFrankaUnlinkRequested: frankaPose.stop()
                    }
                    SpatterTrackingPane {
                        id: spatterTrackingPane
                        visible: window.spatterTrackingOpen
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        onClosed: window.toggleSource("spatterTracking")
                        onResetTracks: cameraWorker.requestSpatterReset()
                    }
                }

                SequenceRecordingConsole {
                    id: sequenceRecordingConsole
                    visible: window.sequenceRecordingOpen
                    Layout.fillWidth: true
                    Layout.fillHeight: !videoRow.visible
                    Layout.preferredHeight: videoRow.visible ? (content.height * 0.25) : -1
                    onClosed: window.toggleSource("sequenceRecording")
                }

                ForceTorquePane {
                    id: forceTorquePane
                    visible: window.forceTorqueOpen
                    Layout.fillWidth: true
                    Layout.fillHeight: !videoRow.visible
                    Layout.preferredHeight: videoRow.visible ? (content.height * 0.4) : -1
                    Layout.minimumHeight: 160
                    onClosed: window.toggleSource("forceTorque")
                }
            }

            Label {
                anchors.fill: parent
                visible: window.cameraConnected && !window.cameraOpen && !window.circleTrackingOpen &&
                        !window.spatterTrackingOpen && !window.sequenceRecordingOpen && !window.forceTorqueOpen
                text: "No sources open. Use \"+ Add Source\" above to begin."
                color: theme.mutedText
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                font.pixelSize: 16
            }

            ConnectionBanner {
                id: connectionBanner
                anchors.fill: parent
                visible: !window.cameraConnected
                onCheckConnection: cameraWorker.connectToCamera()
            }
        }
    }
}
