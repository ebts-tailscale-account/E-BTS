#ifndef E_BTS_CIRCLE_TRACKING_H
#define E_BTS_CIRCLE_TRACKING_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>

#include "circle_detector.h"
#include "contact_localiser.h"
#include "event_window_buffer.h"
#include "temporal_circle_tracker.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <opencv2/core.hpp>

namespace e_bts {

// One window's answer to "where is the indenter": the estimator's own output plus,
// when a pixel->mm calibration has been installed, the same point in millimetres.
struct ContactReading {
    ContactEstimate estimate;
    bool has_mm = false;          // false = no calibration installed, or the peak
                                  // fell outside the calibrated field
    cv::Point2d mm{0.0, 0.0};     // pad-centred millimetres, robot base axes
    Metavision::timestamp window_end_us = 0;
};

// Why the tracking pane is showing nothing, in three numbers.
struct TrackingBufferStats {
    std::uint64_t events_received = 0;   // events this branch's callback has seen
    std::size_t queued_windows    = 0;   // completed windows waiting to be processed
    std::uint64_t dropped_windows = 0;
};

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

    // ---- offline replay (E_BTS_contact_replay) ---------------------------
    //
    // The live path above is deliberately lossy in two ways that are right for a
    // GUI and wrong for a measurement:
    //
    //   pop_latest()   superseded windows are DROPPED so the pane stays current.
    //                  A replay must see every window, in order.
    //   rendering      every window is drawn. Offline there is nobody to show it
    //                  to, and the draw costs more than the estimate does.
    //
    // So this registers a camera callback that feeds the buffer and then drains
    // it SYNCHRONOUSLY, on the reader's own thread. That is what supplies the
    // backpressure: while the estimator works, the file reader is blocked inside
    // add_events() and cannot run ahead, so the bounded queue never overflows and
    // no window is dropped. Polling from another thread cannot achieve this --
    // from_file() reads far faster than the estimator runs.
    //
    // Everything between the events and the estimate is the SAME code the live
    // path runs (process_window below); only the drop and render policies differ.
    // This class got its gain defect from a divergence between what the tracker
    // was told and what it searched, so a second copy of the estimator loop --
    // one for the robot, one for the analysis -- is exactly the thing not to add.
    // start_us discards events before that camera timestamp. Default 0 -- the
    // whole file. It exists for diagnostics (replaying one stretch of a long
    // recording) and NOT to skip a supposed run-in: an earlier version of the
    // replay skipped the first ~90 s believing them to be un-logged setup, when
    // in fact the .raw and the GUI's log merely count from different epochs and
    // cover exactly the same recording. Skipping threw away real pokes.
    void connect_to_camera_offline(Metavision::Camera &camera,
                                   std::function<void(const ContactReading &)> sink,
                                   Metavision::timestamp start_us = 0);

    // Closes the window still accumulating and drains the remainder. Call once
    // the camera has stopped, or the file's last partial window is lost.
    void flush_offline(const std::function<void(const ContactReading &)> &sink);

    // Divergence threshold for localise_contact(). Live code leaves this at
    // kContactMinimumDivergence; the replay sets it to sweep the threshold.
    void set_minimum_divergence(double minimum_divergence);

    // Observe the intermediate state of one window's estimate: the displacement
    // field the divergence is taken over, that divergence, and the resulting
    // estimate. Unset by default and never set by the GUI, so the live path pays
    // nothing; E_BTS_contact_replay uses it to dump a single window for figures.
    // The field and divergence are recomputed with the same functions
    // localise_contact() uses rather than being smuggled out of it, so what is
    // observed cannot drift from what was decided.
    using FieldObserver = std::function<void(const DisplacementField &, const std::vector<double> &,
                                             const ContactEstimate &, Metavision::timestamp)>;
    void set_field_observer(FieldObserver observer);

    // Search the map around each marker's BASELINE REST SITE, the way the code
    // did before circle_map_search_centers() existed. This is the defect, kept
    // switchable on purpose: it is the only way to run the old and new estimators
    // over the same recording and attribute a difference to the change rather
    // than to the pokes. E_BTS_marker_travel_test keeps the same switch.
    // Nothing live should ever set this.
    void set_legacy_search_centers(bool legacy_search_centers);

    // Thread-safe: may be called from a different thread than
    // process_next_window() (e.g. a Qt GUI thread requesting a rebuild).
    void request_baseline_restart();

    // Installs the pixel -> millimetre map. Left unset (the CLI tracker) the
    // overlay reports pixels only; this class deliberately knows nothing about
    // the calibration's file format, so the Qt-based loader stays out of the
    // non-Qt E_BTS_event_circle_tracker target.
    void set_pixel_to_mm(std::function<bool(const cv::Point2d &, cv::Point2d &)> pixel_to_mm);

    // Thread-safe: the GUI thread reads what the poll thread last wrote.
    ContactReading last_contact_reading() const;

    TrackingBufferStats buffer_stats() const;

    void set_collection_time_us(Metavision::timestamp collection_time_us);
    void set_minimum_circle_density(double minimum_circle_density);
    double minimum_circle_density() const;
    Metavision::timestamp collection_time_us() const;
    double expected_radius_px() const;

private:
    // The estimator for ONE window: occupancy frame -> detection -> tracking ->
    // contact. Shared by the live and offline paths; `output` is nullptr when no
    // frame is wanted, which is the only thing the replay does differently.
    ContactReading process_window(const EventWindow &event_window, cv::Mat *output);

    int width_;
    int height_;
    CircleDetector detector_;
    TemporalCircleTracker tracker_;
    std::shared_ptr<EventWindowBuffer> event_windows_;
    std::atomic_bool baseline_restart_pending_{false};
    double minimum_divergence_ = kContactMinimumDivergence;
    bool legacy_search_centers_ = false;
    FieldObserver field_observer_;

    std::function<bool(const cv::Point2d &, cv::Point2d &)> pixel_to_mm_;
    mutable std::mutex contact_mutex_;
    ContactReading last_contact_;
};

} // namespace e_bts

#endif // E_BTS_CIRCLE_TRACKING_H
