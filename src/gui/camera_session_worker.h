#ifndef E_BTS_GUI_CAMERA_SESSION_WORKER_H
#define E_BTS_GUI_CAMERA_SESSION_WORKER_H

#include <atomic>
#include <chrono>
#include <fstream>
#include <optional>
#include <string>

#include "../camera.h"
#include "../camera_calibration.h"
#include "../circle_tracking.h"
#include "../pixel_to_mm.h"
#include "../sequence_recording_controller.h"
#include "../spatter_tracking.h"

#include <QImage>
#include <QObject>
#include <QString>

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>

class QTimer;

namespace e_bts::gui {

// Owns the single shared Metavision::Camera for the whole application and
// everything that consumes it: CameraSource (live view), CircleTrackingSource
// (marker detection/tracking), SpatterTrackingSource (generic moving-blob
// tracking), and SequenceRecordingController (manual + externally-triggered
// RAW recording). The EVK1 only exposes one open handle, so this is the one
// place that handle is opened/closed; the "sources" the ribbon's Add Source
// menu toggles are views into this shared session, not independent camera
// connections.
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
    void setSpatterTrackingActive(bool active);
    void setSequenceRecordingActive(bool active);

    // No-ops with a recordingLogLine() explanation if a recording is already
    // in progress from the other trigger (manual vs. start.cmd/stop.cmd).
    void toggleManualRecording();

    void setAccumulationTimeUs(quint32 accumulation_time_us);
    void setDetectionPercent(double percent);
    void requestBaselineRestart();

    // One slot rather than eleven setters: the Spatter Params dialog commits all
    // of its fields at once on OK, and applying them individually would let the
    // tracker run for a few windows against a half-updated configuration.
    // Values are clamped tracker-side (see spatter_tracker_config.h); the ROI is
    // clipped to the sensor there too, so a rectangle overhanging an edge is
    // accepted and trimmed rather than rejected.
    // Re-reads calibration/pixel_to_mm.json and calibration/elastomer_origin.json
    // without restarting the GUI, so a freshly taught origin takes effect at once.
    void reloadContactCalibration();

    void setSpatterParams(int cell_width, int cell_height, int activation_threshold, int min_size,
                          int max_size, int max_distance, int untracked_threshold, int roi_x,
                          int roi_y, int roi_width, int roi_height);
    // Drops every live track and restarts id numbering.
    void requestSpatterReset();

signals:
    void connected(int width, int height);
    void connectionFailed(QString reason);
    void disconnected(QString reason);
    void cameraFrameReady(QImage frame);
    void trackingFrameReady(QImage frame);
    void spatterFrameReady(QImage frame);
    void recordingLogLine(QString line);
    void recordingStateChanged(bool active);
    // Emitted when a quit.cmd closes the watcher on its own, so the ribbon's
    // Sequence Recording toggle can reflect that it is no longer open.
    void sequenceRecordingWatchStopped();
    // Fired when the shared recorder starts/stops (either the manual ribbon
    // button or start.cmd/stop.cmd), carrying the .raw path so the Wittenstein
    // worker can log a matching _ft.csv in lockstep. Connected cross-thread to
    // WittensteinWorker in combined_main.cpp.
    void recordingStartedPath(QString rawPath);
    void recordingStopped();
    // Live contact location. `valid` false means no contact was found -- the pane
    // must show that as "no contact", never as (0, 0). `hasMm` false means the
    // point is real but could not be put in millimetres (no calibration loaded, or
    // it fell outside the calibrated field).
    void contactEstimateReady(bool valid, bool ambiguous, bool hasMm, double xMm, double yMm,
                              double divergence, int trackedMarkers);
    // One line describing what calibration is in force, emitted at startup and on
    // every reload so the readout can never silently mean something else.
    void contactCalibrationStatus(bool ready, bool hasOrigin, QString summary);

private slots:
    void pollTick();

private:
    void teardownCamera(const QString &reason);
    void loadContactCalibration();
    void installContactCalibration();
    // Takes the .raw path as a string, not a std::filesystem::path: Qt 5.12's moc
    // cannot parse <filesystem> when it is included at the top of a Q_OBJECT
    // header (it dies in bits/fs_fwd.h), and this class's header is moc'd.
    void startContactLog(const std::string &raw_path);
    void stopContactLog();
    void publishContactReading(const e_bts::ContactReading &reading);
    void reportTrackingStarvation();

    std::optional<Metavision::Camera> camera_;
    std::optional<CameraSource> camera_source_;
    std::optional<CircleTrackingSource> circle_tracking_source_;
    std::optional<SpatterTrackingSource> spatter_tracking_source_;
    SequenceRecordingController sequence_recording_;

    // Biases + hardware ROI pushed to the sensor at open (they do not survive
    // between sessions). Kept so each recording can log the ROI it ran with.
    e_bts::AppliedCalibration applied_calibration_;

    // Pixel -> millimetre map plus the taught elastomer centre. Loaded once at
    // construction (not at camera open) so the pane can report a missing or
    // unusable calibration before anything is ever recorded.
    e_bts::PixelToMm pixel_to_mm_;
    std::ofstream contact_csv_;
    bool contact_logging_ = false;
    std::chrono::steady_clock::time_point last_contact_emit_{};
    std::chrono::steady_clock::time_point last_tracking_window_at_{};
    std::chrono::steady_clock::time_point last_starvation_report_at_{};
    bool tracking_starved_ = false;

    std::atomic_bool camera_error_{false};
    bool camera_view_active_      = false;
    bool circle_tracking_active_  = false;
    bool spatter_tracking_active_ = false;

    // Survives teardown/reconnect so a re-opened camera keeps whatever the
    // Spatter Params dialog was last set to, the way the accumulation time does.
    SpatterTrackerParams spatter_params_;

    int width_  = 0;
    int height_ = 0;
    Metavision::timestamp accumulation_time_us_;

    QTimer *poll_timer_ = nullptr;
};

} // namespace e_bts::gui

#endif // E_BTS_GUI_CAMERA_SESSION_WORKER_H
