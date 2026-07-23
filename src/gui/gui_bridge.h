#ifndef E_BTS_GUI_GUI_BRIDGE_H
#define E_BTS_GUI_GUI_BRIDGE_H

#include "camera_session_worker.h"

#include <QImage>
#include <QObject>
#include <QString>

namespace e_bts::gui {

// QML's Connections type refuses to connect to a target QObject that lives
// in a different thread than the QQmlEngine (logs "Illegal attempt to
// connect..." and silently never fires) -- and CameraSessionWorker lives on
// its own QThread on purpose, so circle-tracking's per-window OpenCV work
// never blocks the GUI thread. Calling slots on a cross-thread QObject
// directly from QML *is* safe (Qt auto-queues it), which is why Main.qml
// still calls cameraWorker.connectToCamera() etc. directly -- only the
// signal-listening direction needs this relay. GuiBridge lives on the main
// thread and re-emits each of CameraSessionWorker's signals verbatim, via
// plain QObject::connect (which -- unlike QML's Connections -- does support
// and safely auto-queue cross-thread signal/slot connections); QML listens
// to this object instead of cameraWorker.
class GuiBridge : public QObject {
    Q_OBJECT

public:
    explicit GuiBridge(CameraSessionWorker *worker, QObject *parent = nullptr) : QObject(parent) {
        connect(worker, &CameraSessionWorker::connected, this, &GuiBridge::connected);
        connect(worker, &CameraSessionWorker::connectionFailed, this, &GuiBridge::connectionFailed);
        connect(worker, &CameraSessionWorker::disconnected, this, &GuiBridge::disconnected);
        connect(worker, &CameraSessionWorker::cameraFrameReady, this, &GuiBridge::cameraFrameReady);
        connect(worker, &CameraSessionWorker::trackingFrameReady, this, &GuiBridge::trackingFrameReady);
        connect(worker, &CameraSessionWorker::recordingLogLine, this, &GuiBridge::recordingLogLine);
        connect(worker, &CameraSessionWorker::recordingStateChanged, this, &GuiBridge::recordingStateChanged);
        connect(worker, &CameraSessionWorker::sequenceRecordingWatchStopped, this,
                &GuiBridge::sequenceRecordingWatchStopped);
    }

signals:
    void connected(int width, int height);
    void connectionFailed(QString reason);
    void disconnected(QString reason);
    void cameraFrameReady(QImage frame);
    void trackingFrameReady(QImage frame);
    void recordingLogLine(QString line);
    void recordingStateChanged(bool active);
    void sequenceRecordingWatchStopped();
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_GUI_BRIDGE_H
