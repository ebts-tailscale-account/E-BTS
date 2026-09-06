#include "camera_session_worker.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "../camera_utils.h"
#include "../circle_tracker_config.h"

#include <QTimer>

#include <metavision/sdk/driver/camera_exception.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <opencv2/imgproc.hpp>

namespace e_bts::gui {

namespace {

constexpr int kPollIntervalMs = 5;

// The tracker produces a contact estimate every event window (~100 Hz at the
// default 10 ms accumulation). The CSV gets all of them; the GUI gets 20 Hz,
// because a QML label repainting 100 times a second is unreadable and costs the
// render thread for nothing.
constexpr double kContactDisplayIntervalMs = 50.0;

// A tracking pane with nothing in it is a bug report waiting to be written; say so
// on the console rather than leaving a black rectangle to be interpreted.
constexpr double kTrackingStarvationSeconds        = 1.5;
constexpr double kTrackingStarvationReportIntervalS = 5.0;

// Every diagnostic print from this class uses this prefix so it's easy to
// tell apart from Metavision's own "[HAL] ..." console output when both are
// interleaved in the same terminal.
constexpr const char *kLogPrefix = "[E_BTS_GUI]";

QImage mat_to_qimage(const cv::Mat &frame) {
    // Qt5 on Ubuntu 20.04 predates QImage::Format_BGR888 (Qt 5.14+), so
    // convert to RGB explicitly rather than relying on that enum value.
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

} // namespace

CameraSessionWorker::CameraSessionWorker(QObject *parent) :
    QObject(parent), accumulation_time_us_(kDefaultEventCollectionTimeUs) {
    // Flush std::cout after every write. This class's prints are low-frequency
    // (connection/state-change events, not per-frame), so the performance cost
    // is irrelevant, and it means the debug log survives even if the process
    // is killed abruptly (e.g. closing the window) or output is redirected to
    // a file instead of a terminal.
    std::cout << std::unitbuf;

    sequence_recording_.set_log_callback([this](const std::string &message) {
        std::cout << kLogPrefix << " [recording] " << message << '\n';
        emit recordingLogLine(QString::fromStdString(message));
    });

    // Fan the recorder's start/stop out to the Wittenstein F/T worker so its
    // CSV shares the camera .raw's base name and starts/stops in lockstep.
    // Both paths (manual button, start.cmd) funnel through the controller, so
    // hooking it here covers both. These emit on this worker's thread; the
    // connections to WittensteinWorker (its own thread) auto-queue.
    sequence_recording_.set_recording_started_callback([this](const std::filesystem::path &raw_path) {
        // The SDK writes a .bias sidecar next to every .raw, but nothing records
        // the ROI -- the .raw header always reports the full sensor geometry. Log
        // it ourselves so a run's spatial crop stays recoverable in post.
        std::filesystem::path roi_path = raw_path;
        roi_path.replace_extension(".roi");
        if (!e_bts::write_roi_sidecar(roi_path, applied_calibration_.roi, width_, height_)) {
            std::cout << kLogPrefix << " Could not write ROI sidecar " << roi_path.string() << '\n';
        }
        startContactLog(raw_path.string());
        emit recordingStartedPath(QString::fromStdString(raw_path.string()));
    });
    sequence_recording_.set_recording_stopped_callback([this]() {
        stopContactLog();
        emit recordingStopped();
    });

    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(kPollIntervalMs);
    connect(poll_timer_, &QTimer::timeout, this, &CameraSessionWorker::pollTick);

    // At construction rather than at camera open: a missing or wrong-model
    // calibration should be visible before a single frame is recorded, not
    // discovered afterwards in a run with no coordinates in it.
    loadContactCalibration();
}

void CameraSessionWorker::loadContactCalibration() {
    pixel_to_mm_ = e_bts::load_pixel_to_mm(e_bts::default_pixel_to_mm_path(),
                                           e_bts::default_elastomer_origin_path());
    QString summary;
    if (!pixel_to_mm_.loaded) {
        summary = QStringLiteral("no pixel->mm calibration (%1); contact shown in pixels only")
                      .arg(pixel_to_mm_.error);
    } else if (!pixel_to_mm_.has_origin) {
        // Robot-base millimetres are a real, usable answer -- they are just not the
        // pad-centred ones the readout advertises. Say which is on screen.
        summary = QStringLiteral("pixel->mm from %1, but no elastomer origin (%2); "
                                 "coordinates are ROBOT BASE mm, not pad-centred")
                      .arg(pixel_to_mm_.source_path, pixel_to_mm_.origin_error);
    } else {
        summary = QStringLiteral("pixel->mm from %1; pad centre at robot (%2, %3) mm")
                      .arg(pixel_to_mm_.source_path)
                      .arg(pixel_to_mm_.origin_mm.x, 0, 'f', 2)
                      .arg(pixel_to_mm_.origin_mm.y, 0, 'f', 2);
    }
    std::cout << kLogPrefix << " [contact] " << summary.toStdString() << '\n';
    emit contactCalibrationStatus(pixel_to_mm_.loaded, pixel_to_mm_.has_origin, summary);
}

void CameraSessionWorker::reloadContactCalibration() {
    loadContactCalibration();
    if (circle_tracking_source_) {
        installContactCalibration();
    }
}

void CameraSessionWorker::installContactCalibration() {
    if (!circle_tracking_source_) {
        return;
    }
    if (!pixel_to_mm_.loaded) {
        circle_tracking_source_->set_pixel_to_mm(nullptr);
        return;
    }
    // Captured by value: the tracking source runs this on the poll thread, and a
    // reload replaces the whole callback rather than mutating one it is reading.
    const e_bts::PixelToMm calibration = pixel_to_mm_;
    circle_tracking_source_->set_pixel_to_mm(
        [calibration](const cv::Point2d &pixel, cv::Point2d &mm) { return calibration.pixel_to_pad_mm(pixel, mm); });
}

void CameraSessionWorker::connectToCamera() {
    if (camera_) {
        std::cout << kLogPrefix << " connectToCamera() called while already connected; ignoring.\n";
        return;
    }

    std::cout << kLogPrefix << " Opening first available EVK1...\n";

    try {
        camera_.emplace(Metavision::Camera::from_first_available());
        width_  = camera_->geometry().width();
        height_ = camera_->geometry().height();
        std::cout << kLogPrefix << " Camera opened: " << width_ << "x" << height_ << '\n';

        set_camera_runtime_error_callback(*camera_, camera_error_);

        // Push the tuned biases + hardware ROI BEFORE camera_->start(): the EVK1
        // comes up at sensor defaults every session, so without this a recording
        // silently ignores whatever bias_tuner was used to dial in.
        const e_bts::CameraCalibration calibration =
            e_bts::load_camera_calibration(e_bts::default_camera_bias_path());
        // Absolute paths: the default is cwd-relative, so a GUI launched from the
        // wrong directory would otherwise silently record at sensor defaults.
        std::error_code path_error;
        std::cout << kLogPrefix << " [calib] bias file: "
                  << std::filesystem::absolute(calibration.bias_path, path_error).string() << " ("
                  << (calibration.bias_file_found ? "found" : "MISSING") << ")\n";
        std::cout << kLogPrefix << " [calib] roi file : "
                  << std::filesystem::absolute(calibration.roi_path, path_error).string() << " ("
                  << (calibration.roi_file_found ? "found" : "MISSING") << ")\n";
        if (!calibration.bias_file_found && !calibration.roi_file_found) {
            std::cout << kLogPrefix << " No camera calibration found (override with $E_BTS_CAMERA_BIAS);"
                      << " running at SENSOR DEFAULTS.\n";
        } else if (!calibration.bias_file_found) {
            std::cout << kLogPrefix << " No bias file; biases left at sensor defaults.\n";
        }
        applied_calibration_ = e_bts::apply_camera_calibration(
            *camera_, calibration,
            [](const std::string &message) { std::cout << kLogPrefix << " [calib] " << message << '\n'; });

        camera_source_.emplace(width_, height_, accumulation_time_us_);
        camera_source_->connect_to_camera(*camera_);
        camera_source_->start([this](const cv::Mat &frame) {
            if (camera_view_active_) {
                emit cameraFrameReady(mat_to_qimage(frame));
            }
        });

        circle_tracking_source_.emplace(width_, height_, accumulation_time_us_, kDefaultMinimumCircleDensity);
        circle_tracking_source_->connect_to_camera(*camera_);
        installContactCalibration();

        spatter_tracking_source_.emplace(width_, height_, accumulation_time_us_, spatter_params_);
        spatter_tracking_source_->connect_to_camera(*camera_);

        sequence_recording_.connect_to_camera(*camera_);

        if (!camera_->start()) {
            throw std::runtime_error("Camera did not start.");
        }

        poll_timer_->start();
        std::cout << kLogPrefix << " Camera started; polling for events.\n";
        emit connected(width_, height_);
    } catch (const Metavision::CameraException &error) {
        std::cerr << kLogPrefix << " Metavision camera error: " << error.what() << '\n';
        e_bts::print_available_sources(std::cerr);
        camera_source_.reset();
        circle_tracking_source_.reset();
        spatter_tracking_source_.reset();
        camera_.reset();
        emit connectionFailed(QString::fromStdString(error.what()));
    } catch (const std::exception &error) {
        std::cerr << kLogPrefix << " Error: " << error.what() << '\n';
        e_bts::print_available_sources(std::cerr);
        camera_source_.reset();
        circle_tracking_source_.reset();
        spatter_tracking_source_.reset();
        camera_.reset();
        emit connectionFailed(QString::fromStdString(error.what()));
    }
}

void CameraSessionWorker::setCameraViewActive(bool active) {
    std::cout << kLogPrefix << " Camera view " << (active ? "opened" : "closed") << ".\n";
    camera_view_active_ = active;
}

void CameraSessionWorker::setCircleTrackingActive(bool active) {
    std::cout << kLogPrefix << " Circle tracking " << (active ? "opened" : "closed") << ".\n";
    circle_tracking_active_ = active;
    // Restart the starvation clock, so opening the pane does not immediately
    // inherit however long it happened to sit closed.
    last_tracking_window_at_ = std::chrono::steady_clock::time_point{};
    tracking_starved_        = false;
}

void CameraSessionWorker::setSpatterTrackingActive(bool active) {
    std::cout << kLogPrefix << " Spatter tracking " << (active ? "opened" : "closed") << ".\n";
    spatter_tracking_active_ = active;
    // Re-opening the pane starts from a clean slate: the tracks that were live
    // when it was closed are long gone, and their ids would otherwise reappear
    // as a burst of high numbers with no visible history.
    if (active && spatter_tracking_source_) {
        spatter_tracking_source_->request_reset();
    }
}

void CameraSessionWorker::setSequenceRecordingActive(bool active) {
    if (!camera_) {
        return;
    }
    std::cout << kLogPrefix << " Sequence recording pane " << (active ? "opened" : "closed") << ".\n";
    if (active) {
        sequence_recording_.start_watching();
    } else {
        sequence_recording_.stop_watching();
    }
}

void CameraSessionWorker::toggleManualRecording() {
    if (!camera_) {
        return;
    }

    if (sequence_recording_.is_recording()) {
        sequence_recording_.stop_recording_if_active(*camera_);
        emit recordingStateChanged(false);
        return;
    }

    const bool started = sequence_recording_.start_recording(*camera_, "manual");
    if (!started) {
        std::cout << kLogPrefix
                  << " Manual recording request ignored: a recording is already in progress.\n";
    }
    emit recordingStateChanged(started);
}

void CameraSessionWorker::setAccumulationTimeUs(quint32 accumulation_time_us) {
    std::cout << kLogPrefix << " Accumulation time set to " << accumulation_time_us << " us.\n";
    accumulation_time_us_ = static_cast<Metavision::timestamp>(accumulation_time_us);
    if (camera_source_) {
        camera_source_->set_display_accumulation_time_us(accumulation_time_us_);
    }
    if (circle_tracking_source_) {
        circle_tracking_source_->set_collection_time_us(accumulation_time_us_);
    }
    // Spatter tracking shares the one accumulation time rather than exposing the
    // SDK sample's separate --processing-accumulation-time. The window length is
    // the same physical quantity for both trackers, and two independent controls
    // for it in one ribbon invites setting one and wondering why the other pane
    // did not change.
    if (spatter_tracking_source_) {
        spatter_tracking_source_->set_collection_time_us(accumulation_time_us_);
    }
}

void CameraSessionWorker::setDetectionPercent(double percent) {
    std::cout << kLogPrefix << " Detection percent set to " << percent << "%.\n";
    if (circle_tracking_source_) {
        circle_tracking_source_->set_minimum_circle_density(percent / 100.0);
    }
}

void CameraSessionWorker::requestBaselineRestart() {
    std::cout << kLogPrefix << " Baseline rebuild requested.\n";
    if (circle_tracking_source_) {
        circle_tracking_source_->request_baseline_restart();
    }
}

void CameraSessionWorker::setSpatterParams(int cell_width, int cell_height, int activation_threshold,
                                           int min_size, int max_size, int max_distance,
                                           int untracked_threshold, int roi_x, int roi_y, int roi_width,
                                           int roi_height) {
    spatter_params_.cell_width           = cell_width;
    spatter_params_.cell_height          = cell_height;
    spatter_params_.activation_threshold = activation_threshold;
    spatter_params_.min_size             = min_size;
    spatter_params_.max_size             = max_size;
    spatter_params_.max_distance         = max_distance;
    spatter_params_.untracked_threshold  = untracked_threshold;
    spatter_params_.roi_x                = roi_x;
    spatter_params_.roi_y                = roi_y;
    spatter_params_.roi_width            = roi_width;
    spatter_params_.roi_height           = roi_height;
    if (spatter_tracking_source_) {
        spatter_tracking_source_->set_params(spatter_params_);
        // Read the values back: the tracker clamps them, so logging what was
        // requested would misreport anything out of range.
        spatter_params_ = spatter_tracking_source_->params();
    }
    std::cout << kLogPrefix << " Spatter params: cell " << spatter_params_.cell_width << "x"
              << spatter_params_.cell_height << ", activation " << spatter_params_.activation_threshold
              << ", size " << spatter_params_.min_size << "-" << spatter_params_.max_size << " px, max distance "
              << spatter_params_.max_distance << " px, untracked threshold "
              << spatter_params_.untracked_threshold << ", roi " << spatter_params_.roi_x << ","
              << spatter_params_.roi_y << " " << spatter_params_.roi_width << "x"
              << spatter_params_.roi_height << ".\n";
}

void CameraSessionWorker::requestSpatterReset() {
    std::cout << kLogPrefix << " Spatter track reset requested.\n";
    if (spatter_tracking_source_) {
        spatter_tracking_source_->request_reset();
    }
}

void CameraSessionWorker::pollTick() {
    if (!camera_) {
        return;
    }
    if (camera_error_.exchange(false)) {
        teardownCamera(QStringLiteral("Camera runtime error."));
        return;
    }

    // Non-blocking: the QTimer above already provides the ~5ms cadence the
    // native main-loop variants got from passing a sleep budget straight
    // into poll_and_dispatch(). Blocking here too would needlessly delay
    // delivery of queued slot calls (e.g. ribbon button clicks) on this
    // thread.
    Metavision::EventLoop::poll_and_dispatch(0);

    // Also processed while a recording is running with the pane CLOSED: the
    // contact log is part of the run's data, and it would otherwise be silently
    // empty whenever nobody happened to have the pane open. The rendered frame is
    // only converted and emitted when something is actually watching it.
    if ((circle_tracking_active_ || contact_logging_) && circle_tracking_source_) {
        cv::Mat output;
        if (circle_tracking_source_->process_next_window(output)) {
            if (circle_tracking_active_) {
                emit trackingFrameReady(mat_to_qimage(output));
            }
            publishContactReading(circle_tracking_source_->last_contact_reading());
            last_tracking_window_at_ = std::chrono::steady_clock::now();
            tracking_starved_        = false;
        } else {
            reportTrackingStarvation();
        }
    }

    // Only drained while the pane is open. The buffer still fills from the
    // camera callback either way, but pop_latest() folds every skipped window
    // into the dropped counter, so nothing accumulates unboundedly.
    if (spatter_tracking_active_ && spatter_tracking_source_) {
        cv::Mat output;
        if (spatter_tracking_source_->process_next_window(output)) {
            emit spatterFrameReady(mat_to_qimage(output));
        }
    }

    if (sequence_recording_.is_watching()) {
        const bool quit_requested = sequence_recording_.poll(*camera_);
        if (quit_requested) {
            sequence_recording_.stop_watching();
            emit sequenceRecordingWatchStopped();
        }
    }
}

// A blank tracking pane is silent by construction: the status line, the marker
// overlay and the contact readout are ALL drawn from a processed event window, so
// when no window arrives the pane shows its startup state forever and looks
// identical to "connected but nothing is touching the pad". This says which it is.
//
// The three counters separate the three causes that look the same on screen. Note
// the Camera pane is NOT evidence either way: it is fed by its own CDFrameGenerator
// callback, so it can render a perfectly normal marker field while this branch's
// buffer receives nothing.
void CameraSessionWorker::reportTrackingStarvation() {
    if (!circle_tracking_source_) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (last_tracking_window_at_ == std::chrono::steady_clock::time_point{}) {
        last_tracking_window_at_ = now;   // first tick after opening; nothing is wrong yet
        return;
    }
    const double idle_s = std::chrono::duration<double>(now - last_tracking_window_at_).count();
    if (idle_s < kTrackingStarvationSeconds) {
        return;
    }
    const double since_report_s =
        std::chrono::duration<double>(now - last_starvation_report_at_).count();
    if (tracking_starved_ && since_report_s < kTrackingStarvationReportIntervalS) {
        return;
    }
    last_starvation_report_at_ = now;
    tracking_starved_          = true;

    const e_bts::TrackingBufferStats stats = circle_tracking_source_->buffer_stats();
    std::cout << kLogPrefix << " [tracking] no event window for " << std::fixed << std::setprecision(1)
              << idle_s << " s -- the pane will stay blank. This branch's buffer has received "
              << stats.events_received << " events, has " << stats.queued_windows
              << " window(s) queued, " << stats.dropped_windows << " dropped.\n";
    if (stats.events_received == 0) {
        std::cout << kLogPrefix << " [tracking]   0 events HERE. The camera pane can still look normal"
                     " -- it has its own callback. Reconnect the camera (the tracking callback is"
                     " registered at open).\n";
    } else {
        std::cout << kLogPrefix << " [tracking]   Events are arriving but no window completes."
                     " A window is only queued once it contains an event AND the accumulation time"
                     " has elapsed; check the accumulation time.\n";
    }
}

// Every window's estimate to the CSV; a throttled subset to the GUI.
void CameraSessionWorker::publishContactReading(const e_bts::ContactReading &reading) {
    const e_bts::ContactEstimate &contact = reading.estimate;

    if (contact_logging_ && contact_csv_.is_open()) {
        const double unix_time_s =
            std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
        // Robot-frame millimetres as well as pad-centred ones. The comparison
        // against the Franka happens in the robot frame, and recomputing it later
        // from the pad frame would need the origin that was in force AT THE TIME,
        // which a CSV read six months from now does not have.
        cv::Point2d robot_mm(0.0, 0.0);
        const bool has_robot_mm =
            contact.valid && pixel_to_mm_.loaded && pixel_to_mm_.pixel_to_mm(contact.pixel, robot_mm);

        std::ostringstream line;
        line << std::fixed << std::setprecision(6) << unix_time_s << ',' << reading.window_end_us << ','
             << (contact.valid ? 1 : 0) << ',' << (contact.ambiguous ? 1 : 0) << ',';
        line << std::setprecision(4);
        if (reading.has_mm) {
            line << reading.mm.x << ',' << reading.mm.y << ',';
        } else {
            line << ",,";
        }
        if (has_robot_mm) {
            line << robot_mm.x << ',' << robot_mm.y << ',';
        } else {
            line << ",,";
        }
        if (contact.valid) {
            line << std::setprecision(3) << contact.pixel.x << ',' << contact.pixel.y << ','
                 << contact.cell_col << ',' << contact.cell_row << ',' << contact.divergence << ','
                 << contact.coherence << ',';
        } else {
            line << ",,,,,,";
        }
        line << contact.tracked_markers << '\n';
        contact_csv_ << line.str();
    }

    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double, std::milli>(now - last_contact_emit_).count() < kContactDisplayIntervalMs) {
        return;
    }
    last_contact_emit_ = now;
    emit contactEstimateReady(contact.valid, contact.ambiguous, reading.has_mm, reading.mm.x, reading.mm.y,
                              contact.divergence, contact.tracked_markers);
}

// The contact log shares the .raw's base name, the way the F/T log does.
void CameraSessionWorker::startContactLog(const std::string &raw_path_string) {
    stopContactLog();
    const std::filesystem::path raw_path(raw_path_string);
    const std::filesystem::path csv_path = raw_path.parent_path() / (raw_path.stem().string() + "_contact.csv");
    contact_csv_.open(csv_path);
    if (!contact_csv_.is_open()) {
        std::cout << kLogPrefix << " [contact] could not open " << csv_path.string() << '\n';
        emit recordingLogLine(
            QStringLiteral("Contact log: could not open %1").arg(QString::fromStdString(csv_path.string())));
        return;
    }
    contact_csv_ << "unix_time_s,window_end_us,valid,ambiguous,x_pad_mm,y_pad_mm,x_robot_mm,y_robot_mm,"
                    "pixel_x,pixel_y,cell_col,cell_row,divergence_px_per_cell,coherence,tracked_markers\n";
    contact_logging_ = true;

    // The estimate is only as good as the baseline it is differenced against, and
    // a baseline captured while the indenter was already down defines the loaded
    // shape as "undeformed". Nothing downstream can detect that, so say it here.
    const bool tracking_ready = circle_tracking_source_ && circle_tracking_active_;
    emit recordingLogLine(
        QStringLiteral("Contact log: %1%2")
            .arg(QString::fromStdString(csv_path.filename().string()),
                 tracking_ready ? QString()
                                : QStringLiteral(" (circle tracking was closed -- its baseline is being "
                                                 "collected NOW; keep the pad unloaded for ~2 s)")));
}

void CameraSessionWorker::stopContactLog() {
    if (!contact_csv_.is_open()) {
        contact_logging_ = false;
        return;
    }
    contact_csv_.flush();
    contact_csv_.close();
    contact_logging_ = false;
}

void CameraSessionWorker::teardownCamera(const QString &reason) {
    std::cout << kLogPrefix << " Tearing down camera session: " << reason.toStdString() << '\n';
    poll_timer_->stop();

    if (camera_source_) {
        camera_source_->stop();
        camera_source_.reset();
    }
    stopContactLog();
    circle_tracking_source_.reset();
    spatter_tracking_source_.reset();
    sequence_recording_.stop_watching();
    camera_view_active_      = false;
    circle_tracking_active_  = false;
    spatter_tracking_active_ = false;
    // Sensor state is gone with the handle; re-applied on the next connect.
    applied_calibration_ = e_bts::AppliedCalibration{};

    if (camera_) {
        camera_->stop();
        camera_.reset();
    }

    emit disconnected(reason);
}

} // namespace e_bts::gui
