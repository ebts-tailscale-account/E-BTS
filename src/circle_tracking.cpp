#include "circle_tracking.h"

#include <iostream>

#include "circle_tracker_config.h"
#include "tracking_render_utils.h"

namespace e_bts {

namespace {

void log_baseline_update(const TrackingUpdate &tracking_update, const TemporalCircleTracker &tracker) {
    if (!tracking_update.baseline_captured) {
        return;
    }
    std::cout << "Baseline collected " << tracker.baseline_observed_circle_count()
              << " stable observed circles across " << tracker.baseline_collection_window_count()
              << " processed windows.\n";
    if (tracker.circle_map_available()) {
        std::cout << "Circle map built: " << tracker.circle_map_column_count() << " columns x "
                  << tracker.circle_map_row_count() << " rows = " << tracker.circle_map_site_count()
                  << " expected sites; tracking " << tracker.baseline_track_count()
                  << " baseline-visible markers; mean radius " << tracker.circle_map_radius()
                  << " px; horizontal/vertical pitch " << tracker.circle_map_horizontal_pitch() << "/"
                  << tracker.circle_map_vertical_pitch() << " px; local search radius "
                  << tracker.circle_map_search_radius() << " px.\n";
    } else {
        std::cout << "Could not infer a consistent circle lattice; continuing with contour-based "
                     "tracking for this baseline.\n";
    }
}

} // namespace

CircleTrackingSource::CircleTrackingSource(int width, int height, Metavision::timestamp collection_time_us,
                                           double minimum_circle_density) :
    width_(width), height_(height), detector_(width, height, minimum_circle_density),
    tracker_(detector_.expected_radius_px(), collection_time_us * kTemporalFilterWindowCount),
    event_windows_(std::make_shared<EventWindowBuffer>(width, height, collection_time_us)) {}

void CircleTrackingSource::connect_to_camera(Metavision::Camera &camera) {
    auto event_windows = event_windows_;
    camera.cd().add_callback([event_windows](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        event_windows->add_events(begin, end);
    });
}

bool CircleTrackingSource::process_next_window(cv::Mat &output) {
    EventWindow event_window;
    if (!event_windows_->pop_latest(event_window)) {
        return false;
    }

    const cv::Mat occupied_pixels = make_occupied_pixel_frame(event_window, width_, height_);
    const bool baseline_restart_pending = baseline_restart_pending_.exchange(false);

    CircleDetectionResult detection_result;
    if (tracker_.circle_map_available() && !baseline_restart_pending) {
        detection_result = detector_.detect_from_map(occupied_pixels, tracker_.circle_map_centers(),
                                                      tracker_.circle_map_radius(),
                                                      tracker_.circle_map_search_radius());
    } else {
        detection_result = detector_.detect(occupied_pixels);
    }
    const TrackingUpdate tracking_update =
        tracker_.update(detection_result.detections, event_window.end_us, baseline_restart_pending);
    log_baseline_update(tracking_update, tracker_);

    output = render_tracking_frame(event_window, occupied_pixels, detection_result, tracking_update.circles,
                                   tracker_, event_windows_->dropped_window_count());
    return true;
}

void CircleTrackingSource::request_baseline_restart() {
    baseline_restart_pending_ = true;
}

void CircleTrackingSource::set_collection_time_us(Metavision::timestamp collection_time_us) {
    event_windows_->set_collection_time_us(collection_time_us);
}

void CircleTrackingSource::set_minimum_circle_density(double minimum_circle_density) {
    detector_.set_minimum_circle_density(minimum_circle_density);
}

double CircleTrackingSource::minimum_circle_density() const {
    return detector_.minimum_circle_density();
}

Metavision::timestamp CircleTrackingSource::collection_time_us() const {
    return event_windows_->collection_time_us();
}

double CircleTrackingSource::expected_radius_px() const {
    return detector_.expected_radius_px();
}

} // namespace e_bts
