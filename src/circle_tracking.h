#ifndef E_BTS_CIRCLE_TRACKING_H
#define E_BTS_CIRCLE_TRACKING_H

#include <atomic>
#include <cstdint>
#include <memory>

#include "circle_detector.h"
#include "event_window_buffer.h"
#include "temporal_circle_tracker.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <opencv2/core.hpp>

namespace e_bts {

// Owns the circle-detection/tracking branch of a shared camera session: an
// EventWindowBuffer fed by the camera's CD events, a CircleDetector, and a
// TemporalCircleTracker. The Metavision::Camera itself is owned by the
// caller (CameraSessionWorker in the GUI, or the standalone tracker CLI's
// main) and shared with CameraSource, since the EVK1 only supports one open
// handle at a time.
class CircleTrackingSource {
public:
    CircleTrackingSource(int width, int height, Metavision::timestamp collection_time_us,
                         double minimum_circle_density);

    void connect_to_camera(Metavision::Camera &camera);

    // Call periodically (e.g. every 5ms) from the owning poll loop. If a new
    // event window has been processed, renders it and returns true.
    bool process_next_window(cv::Mat &output);

    // Thread-safe: may be called from a different thread than
    // process_next_window() (e.g. a Qt GUI thread requesting a rebuild).
    void request_baseline_restart();

    void set_collection_time_us(Metavision::timestamp collection_time_us);
    void set_minimum_circle_density(double minimum_circle_density);
    double minimum_circle_density() const;
    Metavision::timestamp collection_time_us() const;
    double expected_radius_px() const;

private:
    int width_;
    int height_;
    CircleDetector detector_;
    TemporalCircleTracker tracker_;
    std::shared_ptr<EventWindowBuffer> event_windows_;
    std::atomic_bool baseline_restart_pending_{false};
};

} // namespace e_bts

#endif // E_BTS_CIRCLE_TRACKING_H
