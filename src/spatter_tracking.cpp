#include "spatter_tracking.h"

#include "spatter_render.h"

#include <cstddef>
#include <cstdint>
#include <vector>


namespace e_bts {

SpatterTrackingSource::SpatterTrackingSource(int width, int height, Metavision::timestamp collection_time_us,
                                             const SpatterTrackerParams &params) :
    width_(width), height_(height), tracker_(width, height, params),
    event_windows_(std::make_shared<EventWindowBuffer>(width, height, collection_time_us)) {}

void SpatterTrackingSource::connect_to_camera(Metavision::Camera &camera) {
    auto event_windows = event_windows_;
    camera.cd().add_callback([event_windows](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        event_windows->add_events(begin, end);
    });
}

bool SpatterTrackingSource::process_next_window(cv::Mat &output) {
    EventWindow event_window;
    if (!event_windows_->pop_latest(event_window)) {
        return false;
    }

    if (reset_pending_.exchange(false)) {
        tracker_.reset();
    }

    const cv::Mat occupied_pixels                 = make_occupied_pixel_frame(event_window, width_, height_);
    const std::vector<SpatterCluster> &clusters   = tracker_.update(event_window);

    output = render_spatter_frame(event_window, occupied_pixels, clusters, tracker_.params(),
                                  tracker_.diagnostics(), tracker_.track_count(),
                                  event_windows_->dropped_window_count(), tracker_.roi());
    return true;
}

void SpatterTrackingSource::set_collection_time_us(Metavision::timestamp collection_time_us) {
    event_windows_->set_collection_time_us(collection_time_us);
}

Metavision::timestamp SpatterTrackingSource::collection_time_us() const {
    return event_windows_->collection_time_us();
}

void SpatterTrackingSource::set_params(const SpatterTrackerParams &params) {
    tracker_.set_params(params);
}

SpatterTrackerParams SpatterTrackingSource::params() const {
    return tracker_.params();
}

void SpatterTrackingSource::request_reset() {
    reset_pending_ = true;
}

} // namespace e_bts
