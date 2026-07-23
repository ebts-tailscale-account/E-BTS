#ifndef E_BTS_GUI_CAMERA_SESSION_WORKER_H
#define E_BTS_GUI_CAMERA_SESSION_WORKER_H

#include <atomic>
#include <optional>

#include "../camera.h"
#include "../circle_tracking.h"
#include "../sequence_recording_controller.h"

#include <QImage>
#include <QObject>
#include <QString>

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>

class QTimer;

namespace e_bts::gui {

// Owns the single shared Metavision::Camera for the whole application and
// everything that consumes it: CameraSource (live view), CircleTrackingSource
// (marker detection/tracking), and SequenceRecordingController (manual +
// externally-triggered RAW recording). The EVK1 only exposes one open
// handle, so this is the one place that handle is opened/closed; the three
// "sources" the ribbon's Add Source menu toggles are views into this shared
// session, not independent camera connections.
//
// Meant to live on its own QThread (moveToThread by the caller). All public
// slots run on that thread; signals are safe to connect to GUI-thread
// objects with a normal (auto/queued) connection.
class CameraSessionWorker : public QObject {
    Q_OBJECT

public:
    explicit CameraSessionWorker(QObject *parent = nullptr);

public slots:
    // Attempts to open the first available EVK1. Emits connected() or
    // connectionFailed(); safe to call again after a failure (e.g. from a
    // "Check Connection" button) or after disconnected().
    void connectToCamera();

    void setCameraViewActive(bool active);
    void setCircleTrackingActive(bool active);
    void setSequenceRecordingActive(bool active);

    // No-ops with a recordingLogLine() explanation if a recording is already
    // in progress from the other trigger (manual vs. start.cmd/stop.cmd).
    void toggleManualRecording();

    void setAccumulationTimeUs(quint32 accumulation_time_us);
    void setDetectionPercent(double percent);
    void requestBaselineRestart();

signals:
    void connected(int width, int height);
    void connectionFailed(QString reason);
    void disconnected(QString reason);
    void cameraFrameReady(QImage frame);
    void trackingFrameReady(QImage frame);
    void recordingLogLine(QString line);
    void recordingStateChanged(bool active);
    // Emitted when a quit.cmd closes the watcher on its own, so the ribbon's
    // Sequence Recording toggle can reflect that it is no longer open.
    void sequenceRecordingWatchStopped();

private slots:
    void pollTick();

private:
    void teardownCamera(const QString &reason);

    std::optional<Metavision::Camera> camera_;
    std::optional<CameraSource> camera_source_;
    std::optional<CircleTrackingSource> circle_tracking_source_;
    SequenceRecordingController sequence_recording_;

    std::atomic_bool camera_error_{false};
    bool camera_view_active_     = false;
    bool circle_tracking_active_ = false;

    int width_  = 0;
    int height_ = 0;
    Metavision::timestamp accumulation_time_us_;

    QTimer *poll_timer_ = nullptr;
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_CAMERA_SESSION_WORKER_H
