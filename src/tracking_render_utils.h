#ifndef E_BTS_TRACKING_RENDER_UTILS_H
#define E_BTS_TRACKING_RENDER_UTILS_H

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "circle_detector.h"
#include "circle_tracker_config.h"
#include "contact_localiser.h"
#include "event_window_buffer.h"
#include "temporal_circle_tracker.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/ui/utils/mt_window.h>
#include <metavision/sdk/ui/utils/ui_events.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace e_bts {

inline cv::Point rounded_point(const cv::Point2f &point) {
    return cv::Point(cvRound(point.x), cvRound(point.y));
}

inline cv::Scalar rejection_color(CircleRejectionReason reason) {
    switch (reason) {
    case CircleRejectionReason::SmallRadius:
        return cv::Scalar(255, 128, 0);
    case CircleRejectionReason::LargeRadius:
        return cv::Scalar(0, 128, 255);
    case CircleRejectionReason::LowContourFill:
        return cv::Scalar(255, 0, 255);
    case CircleRejectionReason::LowDensity:
        return cv::Scalar(255, 255, 0);
    }
    return cv::Scalar(255, 255, 255);
}

inline std::string format_millimetres(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%+.2f", value);
    return buffer;
}

// Marks the estimated contact: a crosshair at the peak plus its coordinates. The
// crosshair sits at the peak's position in BASELINE (undeformed) pixels, which is
// the frame the divergence field lives in -- material directly under the indenter
// barely moves sideways, so on the live view the two coincide to well under a
// marker.
inline void draw_contact_overlay(cv::Mat &output, const ContactEstimate &contact, bool has_mm,
                                 const cv::Point2d &mm) {
    if (!contact.valid) {
        return;
    }
    const cv::Point centre(cvRound(contact.pixel.x), cvRound(contact.pixel.y));
    const cv::Scalar colour = contact.ambiguous ? cv::Scalar(0, 200, 255) : cv::Scalar(255, 255, 255);
    const int arm = 12;
    cv::line(output, cv::Point(centre.x - arm, centre.y), cv::Point(centre.x + arm, centre.y), colour, 1,
             cv::LINE_AA);
    cv::line(output, cv::Point(centre.x, centre.y - arm), cv::Point(centre.x, centre.y + arm), colour, 1,
             cv::LINE_AA);
    cv::circle(output, centre, 5, colour, 1, cv::LINE_AA);

    const std::string label =
        has_mm ? "x " + format_millimetres(mm.x) + "  y " + format_millimetres(mm.y) + " mm"
               : "px " + std::to_string(cvRound(contact.pixel.x)) + "," +
                     std::to_string(cvRound(contact.pixel.y)) + " (no mm calibration)";
    cv::putText(output, label, cv::Point(std::min(centre.x + 14, output.cols - 200), centre.y - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv::LINE_AA);
    if (contact.ambiguous) {
        // Advertised as a SINGLE-contact readout, so two surviving peaks make the
        // number meaningless rather than merely uncertain. Say so on the frame.
        cv::putText(output, "2 contacts -- estimate not single-point",
                    cv::Point(std::min(centre.x + 14, output.cols - 260), centre.y + 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
    }
}

inline cv::Mat render_tracking_frame(const EventWindow &event_window, const cv::Mat &occupied_pixels,
                                     const CircleDetectionResult &detection_result,
                                     const std::vector<TrackedCircle> &tracked_circles,
                                     const TemporalCircleTracker &tracker,
                                     std::uint64_t dropped_window_count,
                                     const ContactEstimate &contact = ContactEstimate{},
                                     bool contact_has_mm = false,
                                     const cv::Point2d &contact_mm = cv::Point2d(0.0, 0.0)) {
    cv::Mat output = cv::Mat::zeros(occupied_pixels.size(), CV_8UC3);
    output.setTo(cv::Scalar(32, 32, 32), detection_result.candidate_mask);

    // The cleaned mask is gray. Rejected contours are orange for excessive
    // radius, magenta for contour fill, and cyan for occupied-pixel density.
    for (const auto &rejected_circle : detection_result.rejected_circles) {
        cv::circle(output, rounded_point(rejected_circle.center),
                   std::max(1, cvRound(rejected_circle.radius)),
                   rejection_color(rejected_circle.reason), 1, cv::LINE_AA);
    }

    const cv::Scalar circle_color(0, 255, 0);
    const cv::Scalar vector_color(0, 0, 255);
    for (const auto &tracked_circle : tracked_circles) {
        const cv::Point center = rounded_point(tracked_circle.detection.center);
        cv::circle(output, center, cvRound(tracked_circle.detection.radius), circle_color, cv::FILLED, cv::LINE_AA);

        if (!tracked_circle.has_baseline) {
            continue;
        }
        const cv::Point2f displacement = tracked_circle.detection.center - tracked_circle.baseline_center;
        if (displacement.dot(displacement) >=
            kMinimumDisplacementVectorLengthPx * kMinimumDisplacementVectorLengthPx) {
            // The arrow starts at the current marker center and has the true
            // baseline-relative displacement length in native camera pixels.
            cv::arrowedLine(output, center, rounded_point(tracked_circle.detection.center + displacement),
                            vector_color, 3, cv::LINE_AA, 0, 0.30);
        }
    }

    const Metavision::timestamp collection_time_us = event_window.end_us - event_window.start_us;
    std::string status = std::to_string(collection_time_us) + " us | circles: " +
                         std::to_string(tracked_circles.size());
    if (tracker.baseline_collecting()) {
        status += " | base collect w:" +
                  std::to_string(tracker.baseline_collection_window_count()) + " stable:" +
                  std::to_string(tracker.stable_baseline_candidate_count());
    } else {
        status += " | base:" + std::to_string(tracker.baseline_track_count());
        if (tracker.circle_map_available()) {
            status += " map:" + std::to_string(tracker.circle_map_column_count()) + "x" +
                      std::to_string(tracker.circle_map_row_count());
        }
        status += " (B: reset)";
    }
    if (dropped_window_count != 0) {
        status += " | drop:" + std::to_string(dropped_window_count);
    }
    cv::putText(output, status, cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1,
                cv::LINE_AA);

    const CircleDetectionDiagnostics &diagnostics = detection_result.diagnostics;
    std::string rejection_status;
    if (diagnostics.used_circle_map) {
        rejection_status = "map sites:" + std::to_string(diagnostics.map_site_count) +
                           " | active:" + std::to_string(detection_result.detections.size()) +
                           " density reject:" + std::to_string(diagnostics.low_density_count);
    } else {
        rejection_status =
            "contours:" + std::to_string(diagnostics.contour_count) +
            " | reject radius(s/l):" + std::to_string(diagnostics.small_radius_count) + "/" +
            std::to_string(diagnostics.large_radius_count) +
            " fill:" + std::to_string(diagnostics.low_contour_fill_count) +
            " density:" + std::to_string(diagnostics.low_density_count) +
            " tiny:" + std::to_string(diagnostics.short_contour_count) +
            " dedup:" + std::to_string(diagnostics.duplicate_detection_count);
    }
    cv::putText(output, rejection_status, cv::Point(8, 36), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(0, 255, 255), 1, cv::LINE_AA);

    std::string contact_status;
    if (contact.valid) {
        contact_status = "contact: ";
        contact_status += contact_has_mm
                              ? "x " + format_millimetres(contact_mm.x) + " y " +
                                    format_millimetres(contact_mm.y) + " mm"
                              : "px " + std::to_string(cvRound(contact.pixel.x)) + "," +
                                    std::to_string(cvRound(contact.pixel.y));
        char strength[48];
        std::snprintf(strength, sizeof(strength), " | div %.1f coh %.2f", contact.divergence,
                      contact.coherence);
        contact_status += strength;
    } else if (contact.lattice_rows > 0) {
        contact_status = "contact: none (no divergence peak above threshold)";
    }
    if (!contact_status.empty()) {
        cv::putText(output, contact_status, cv::Point(8, 54), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                    cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
    draw_contact_overlay(output, contact, contact_has_mm, contact_mm);
    return output;
}

inline void set_tracking_window_keyboard_callback(Metavision::MTWindow &window, std::atomic_bool &baseline_requested) {
    window.set_keyboard_callback(
        [&window, &baseline_requested](Metavision::UIKeyEvent key, int, Metavision::UIAction action, int) {
            if (action != Metavision::UIAction::RELEASE) {
                return;
            }

            if (key == Metavision::UIKeyEvent::KEY_ESCAPE || key == Metavision::UIKeyEvent::KEY_Q) {
                window.set_close_flag();
            } else if (key == Metavision::UIKeyEvent::KEY_B) {
                baseline_requested = true;
            }
        });
}

} // namespace e_bts

#endif // E_BTS_TRACKING_RENDER_UTILS_H
