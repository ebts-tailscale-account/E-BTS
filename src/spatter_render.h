#ifndef E_BTS_SPATTER_RENDER_H
#define E_BTS_SPATTER_RENDER_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "event_window_buffer.h"
#include "spatter_tracker.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// The Spatter Tracking pane's frame, factored out of spatter_tracking.cpp so the
// offline tool (spatter_track_csv --video) draws the SAME picture the live pane
// does. Two renderers would drift apart, and then a video would stop being
// evidence about what the tracker actually did.

namespace e_bts {

// Stable per-id colour. The hue steps by the golden-angle fraction of the wheel,
// which keeps consecutively-issued ids far apart in hue -- important because new
// tracks are created in bursts and adjacent ids are usually adjacent on screen.
// Full saturation/value so every colour reads against the near-black event frame.
inline cv::Scalar track_color(std::uint32_t id) {
    const float hue = std::fmod(static_cast<float>(id) * 0.618033988749895F, 1.0F) * 180.0F; // OpenCV hue is 0-179
    cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, 255, 255));
    cv::Mat bgr;
    cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
    const cv::Vec3b &pixel = bgr.at<cv::Vec3b>(0, 0);
    return cv::Scalar(pixel[0], pixel[1], pixel[2]);
}

inline cv::Point rounded_point(const cv::Point2f &point) {
    return cv::Point(cvRound(point.x), cvRound(point.y));
}

inline cv::Mat render_spatter_frame(const EventWindow &event_window, const cv::Mat &occupied_pixels,
                             const std::vector<SpatterCluster> &clusters,
                             const SpatterTrackerParams &params, const SpatterDiagnostics &diagnostics,
                             std::size_t track_count, std::uint64_t dropped_window_count,
                             const cv::Rect &roi) {
    // The whole sensor is still drawn, not just the ROI: a cropped view would
    // make it impossible to tell a badly-placed ROI from a genuinely quiet
    // scene. Events inside the ROI keep the usual grey, events outside are
    // painted darker so the crop reads at a glance without hiding them.
    cv::Mat output = cv::Mat::zeros(occupied_pixels.size(), CV_8UC3);
    output.setTo(cv::Scalar(18, 18, 18), occupied_pixels);
    cv::Mat inside_roi = output(roi);
    inside_roi.setTo(cv::Scalar(48, 48, 48), occupied_pixels(roi));
    cv::rectangle(output, roi, cv::Scalar(90, 90, 0), 1, cv::LINE_4);

    for (const SpatterCluster &cluster : clusters) {
        const cv::Scalar color = track_color(cluster.id);

        // Trail first, so boxes and labels stay on top of it.
        for (std::size_t index = 1; index < cluster.trail.size(); ++index) {
            cv::line(output, rounded_point(cluster.trail[index - 1]), rounded_point(cluster.trail[index]), color,
                     1, cv::LINE_AA);
        }

        cv::rectangle(output, cluster.box, color, 1, cv::LINE_AA);
        cv::circle(output, rounded_point(cluster.center), 2, color, cv::FILLED, cv::LINE_AA);

        // Label above the box where there is room, inside it otherwise, so ids
        // on objects touching the top edge do not get clipped away.
        const int label_y = cluster.box.y >= 14 ? cluster.box.y - 4 : cluster.box.y + 12;
        cv::putText(output, std::to_string(cluster.id), cv::Point(cluster.box.x, label_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv::LINE_AA);
    }

    const Metavision::timestamp collection_time_us = event_window.end_us - event_window.start_us;
    std::string status = std::to_string(collection_time_us) + " us | tracked: " + std::to_string(clusters.size()) +
                         "/" + std::to_string(track_count) + " alive";
    if (diagnostics.coasting_tracks != 0) {
        status += " | coast:" + std::to_string(diagnostics.coasting_tracks);
    }
    if (dropped_window_count != 0) {
        status += " | drop:" + std::to_string(dropped_window_count);
    }
    cv::putText(output, status, cv::Point(8, 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1,
                cv::LINE_AA);

    // Second line is the "why is nothing showing up" line: it separates a dead
    // event stream from cells that never activate from clusters killed by the
    // size filter, which are three different knobs to reach for.
    const std::string detail = "px:" + std::to_string(diagnostics.pixels_in_roi) + "/" +
                               std::to_string(diagnostics.occupied_pixels) +
                               " cells:" + std::to_string(diagnostics.active_cells) +
                               " clust:" + std::to_string(diagnostics.raw_clusters) +
                               " reject(s/l):" + std::to_string(diagnostics.too_small) + "/" +
                               std::to_string(diagnostics.too_large) +
                               " new:" + std::to_string(diagnostics.new_tracks) +
                               " ret:" + std::to_string(diagnostics.retired_tracks);
    cv::putText(output, detail, cv::Point(8, 36), cv::FONT_HERSHEY_SIMPLEX, 0.38, cv::Scalar(0, 255, 255), 1,
                cv::LINE_AA);

    const std::string tuning = "cell " + std::to_string(params.cell_width) + "x" +
                               std::to_string(params.cell_height) + " | act " +
                               std::to_string(params.activation_threshold) + " | size " +
                               std::to_string(params.min_size) + "-" + std::to_string(params.max_size) +
                               " | gate " + std::to_string(params.max_distance) + " | keep " +
                               std::to_string(params.untracked_threshold) + " | roi " +
                               std::to_string(roi.x) + "," + std::to_string(roi.y) + " " +
                               std::to_string(roi.width) + "x" + std::to_string(roi.height);
    cv::putText(output, tuning, cv::Point(8, occupied_pixels.rows - 8), cv::FONT_HERSHEY_SIMPLEX, 0.38,
                cv::Scalar(160, 160, 160), 1, cv::LINE_AA);
    return output;
}

} // namespace e_bts

#endif // E_BTS_SPATTER_RENDER_H
