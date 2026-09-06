#ifndef E_BTS_SPATTER_TRACKING_H
#define E_BTS_SPATTER_TRACKING_H

#include <atomic>
#include <memory>

#include "event_window_buffer.h"
#include "spatter_tracker.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <opencv2/core.hpp>

namespace e_bts {

// Owns the spatter-tracking branch of a shared camera session: its own
// EventWindowBuffer fed by the camera's CD events, and a SpatterTracker over it.
// Deliberately parallel to CircleTrackingSource -- same construction, same
// connect_to_camera/process_next_window contract -- so CameraSessionWorker can
// drive both from one poll loop without special-casing either.
//
// The buffer is this source's own rather than shared with CircleTrackingSource:
// each camera CD callback delivers to every registered consumer, and
// EventWindowBuffer::pop_latest() is destructive (it drains the queue and counts
// what it skipped), so two consumers on one buffer would each see an arbitrary
// half of the windows.
class SpatterTrackingSource {
public:
    SpatterTrackingSource(int width, int height, Metavision::timestamp collection_time_us,
                          const SpatterTrackerParams &params);

    void connect_to_camera(Metavision::Camera &camera);

    // Call periodically (e.g. every 5ms) from the owning poll loop. If a new
    // event window has been processed, renders it and returns true.
    bool process_next_window(cv::Mat &output);

    void set_collection_time_us(Metavision::timestamp collection_time_us);
    Metavision::timestamp collection_time_us() const;

    void set_params(const SpatterTrackerParams &params);
    SpatterTrackerParams params() const;

    // Thread-safe: may be called from a different thread than
    // process_next_window(), like CircleTrackingSource::request_baseline_restart().
    void request_reset();

private:
    int width_;
    int height_;
    SpatterTracker tracker_;
    std::shared_ptr<EventWindowBuffer> event_windows_;
    std::atomic_bool reset_pending_{false};
};

} // namespace e_bts

#endif // E_BTS_SPATTER_TRACKING_H
