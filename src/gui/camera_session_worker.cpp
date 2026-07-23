#include "camera_session_worker.h"

#include <exception>
#include <iostream>

#include "../camera_utils.h"
#include "../circle_tracker_config.h"

#include <QTimer>

#include <metavision/sdk/driver/camera_exception.h>
#include <metavision/sdk/ui/utils/event_loop.h>
#include <opencv2/imgproc.hpp>

namespace e_bts::gui {

namespace {

constexpr int kPollIntervalMs = 5;

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

    poll_timer_ = new QTimer(this);
    poll_timer_->setInterval(kPollIntervalMs);
    connect(poll_timer_, &QTimer::timeout, this, &CameraSessionWorker::pollTick);
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

        camera_source_.emplace(width_, height_, accumulation_time_us_);
        camera_source_->connect_to_camera(*camera_);
        camera_source_->start([this](const cv::Mat &frame) {
            if (camera_view_active_) {
                emit cameraFrameReady(mat_to_qimage(frame));
            }
        });

        circle_tracking_source_.emplace(width_, height_, accumulation_time_us_, kDefaultMinimumCircleDensity);
        circle_tracking_source_->connect_to_camera(*camera_);

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
        camera_.reset();
        emit connectionFailed(QString::fromStdString(error.what()));
    } catch (const std::exception &error) {
        std::cerr << kLogPrefix << " Error: " << error.what() << '\n';
        e_bts::print_available_sources(std::cerr);
        camera_source_.reset();
        circle_tracking_source_.reset();
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

    if (circle_tracking_active_ && circle_tracking_source_) {
        cv::Mat output;
        if (circle_tracking_source_->process_next_window(output)) {
            emit trackingFrameReady(mat_to_qimage(output));
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

void CameraSessionWorker::teardownCamera(const QString &reason) {
    std::cout << kLogPrefix << " Tearing down camera session: " << reason.toStdString() << '\n';
    poll_timer_->stop();

    if (camera_source_) {
        camera_source_->stop();
        camera_source_.reset();
    }
    circle_tracking_source_.reset();
    sequence_recording_.stop_watching();
    camera_view_active_     = false;
    circle_tracking_active_ = false;

    if (camera_) {
        camera_->stop();
        camera_.reset();
    }

    emit disconnected(reason);
}

} // namespace e_bts::gui
