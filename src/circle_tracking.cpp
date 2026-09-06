#include "circle_tracking.h"

#include <algorithm>
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
    process_window(event_window, &output);
    return true;
}

// See the header: the live and offline paths differ only in which window they
// take and whether they draw it. Everything that decides WHERE THE CONTACT IS
// lives here, once.
ContactReading CircleTrackingSource::process_window(const EventWindow &event_window, cv::Mat *output) {
    const cv::Mat occupied_pixels = make_occupied_pixel_frame(event_window, width_, height_);
    const bool baseline_restart_pending = baseline_restart_pending_.exchange(false);

    CircleDetectionResult detection_result;
    if (tracker_.circle_map_available() && !baseline_restart_pending) {
        // SEARCH centres, not baseline sites: see
        // TemporalCircleTracker::circle_map_search_centers() for why the difference
        // matters and what it cost.
        const std::vector<cv::Point2f> &search_centers = legacy_search_centers_
                                                             ? tracker_.circle_map_centers()
                                                             : tracker_.circle_map_search_centers();
        detection_result = detector_.detect_from_map(occupied_pixels, search_centers,
                                                      tracker_.circle_map_radius(),
                                                      tracker_.circle_map_search_radius());
    } else {
        detection_result = detector_.detect(occupied_pixels);
    }
    const TrackingUpdate tracking_update =
        tracker_.update(detection_result.detections, event_window.end_us, baseline_restart_pending);
    log_baseline_update(tracking_update, tracker_);

    // The contact estimate needs a lattice to differentiate over, so it is only
    // attempted once the circle map exists. Without one there is a set of tracked
    // blobs but no neighbour relation, and a divergence has no meaning.
    ContactReading reading;
    reading.window_end_us = event_window.end_us;
    if (tracker_.circle_map_available() && !tracker_.baseline_collecting()) {
        reading.estimate = localise_contact(tracking_update.circles, tracker_.circle_map_row_count(),
                                            tracker_.circle_map_column_count(), minimum_divergence_);
        // peak_found, not valid: a peak that survived pruning but fell below the
        // divergence threshold still has a position, and converting it costs
        // nothing. Without that, re-thresholding a recorded run offline would be
        // limited to the peaks the run had already accepted.
        if (reading.estimate.peak_found && pixel_to_mm_) {
            cv::Point2d mm;
            if (pixel_to_mm_(reading.estimate.pixel, mm)) {
                reading.has_mm = true;
                reading.mm     = mm;
            }
        }
    }
    if (field_observer_ && tracker_.circle_map_available() && !tracker_.baseline_collecting()) {
        const DisplacementField field = build_displacement_field(
            tracking_update.circles, tracker_.circle_map_row_count(), tracker_.circle_map_column_count());
        field_observer_(field, field_divergence(field), reading.estimate, event_window.end_us);
    }
    {
        const std::lock_guard<std::mutex> lock(contact_mutex_);
        last_contact_ = reading;
    }

    if (output != nullptr) {
        *output = render_tracking_frame(event_window, occupied_pixels, detection_result,
                                        tracking_update.circles, tracker_,
                                        event_windows_->dropped_window_count(), reading.estimate,
                                        reading.has_mm, reading.mm);
    }
    return reading;
}

void CircleTrackingSource::connect_to_camera_offline(Metavision::Camera &camera,
                                                     std::function<void(const ContactReading &)> sink,
                                                     Metavision::timestamp start_us) {
    camera.cd().add_callback(
        [this, sink = std::move(sink), start_us](const Metavision::EventCD *begin,
                                                 const Metavision::EventCD *end) {
            if (start_us > 0) {
                // Events within a callback are in timestamp order, so the cut is
                // a partition point rather than a filter.
                begin = std::lower_bound(begin, end, start_us,
                                         [](const Metavision::EventCD &event, Metavision::timestamp t) {
                                             return event.t < t;
                                         });
                if (begin == end) {
                    return;
                }
            }
            event_windows_->add_events(begin, end);
            // Drain to empty before returning, so the reader thread is throttled
            // by the estimator rather than racing it.
            EventWindow event_window;
            while (event_windows_->pop_oldest(event_window)) {
                sink(process_window(event_window, nullptr));
            }
        });
}

void CircleTrackingSource::flush_offline(const std::function<void(const ContactReading &)> &sink) {
    event_windows_->flush();
    EventWindow event_window;
    while (event_windows_->pop_oldest(event_window)) {
        sink(process_window(event_window, nullptr));
    }
}

void CircleTrackingSource::set_minimum_divergence(double minimum_divergence) {
    minimum_divergence_ = minimum_divergence;
}

void CircleTrackingSource::set_field_observer(FieldObserver observer) {
    field_observer_ = std::move(observer);
}

void CircleTrackingSource::set_legacy_search_centers(bool legacy_search_centers) {
    legacy_search_centers_ = legacy_search_centers;
}

void CircleTrackingSource::set_pixel_to_mm(std::function<bool(const cv::Point2d &, cv::Point2d &)> pixel_to_mm) {
    pixel_to_mm_ = std::move(pixel_to_mm);
}

TrackingBufferStats CircleTrackingSource::buffer_stats() const {
    return TrackingBufferStats{event_windows_->total_event_count(), event_windows_->queued_window_count(),
                               event_windows_->dropped_window_count()};
}

ContactReading CircleTrackingSource::last_contact_reading() const {
    const std::lock_guard<std::mutex> lock(contact_mutex_);
    return last_contact_;
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
