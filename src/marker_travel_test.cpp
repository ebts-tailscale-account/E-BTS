// E_BTS_marker_travel_test -- can a marker be followed past the search radius?
//
// WHY THIS EXISTS
// ---------------
// Run ladder_20260906_150159 (1100 pokes) measured the live contact estimate
// travelling only 0.62-0.64x as far as the indenter really did. The cause was that
// CircleDetector::detect_from_map searched a fixed +-9.3 px window around each
// marker's BASELINE site, while markers beside a 5 mm indent move 15-25 px -- so the
// markers carrying the contact signal were undetectable and the divergence peak was
// computed from a truncated annulus.
//
// That diagnosis was arithmetic plus a regression on field data. This makes the
// mechanism re-runnable: a synthetic marker lattice, a radial deformation that
// displaces markers by a known amount, and the real CircleDetector and
// TemporalCircleTracker in between. It reports the LARGEST DISPLACEMENT the tracker
// can still follow.
//
//     ./E_BTS_marker_travel_test               # current code
//     ./E_BTS_marker_travel_test --legacy      # search anchored to baseline (the bug)
//
// ⚠ SCOPE. This tests marker TRACKING, not contact localisation, and deliberately
// stops there. An earlier version of this file tried to score localise_contact()
// against a known contact position and produced nonsense, because a synthetic
// lattice of hard discs is not a marker image: displaced discs collide at this
// pitch, the density gate then rejects them, and the divergence peak lands on the
// resulting holes. Rather than tune a simulation until it flatters the code, the
// test was cut back to the one claim a synthetic lattice CAN support -- how far a
// marker can move before the detector loses it -- which is exactly the quantity the
// bug was about.
//
// End-to-end localisation accuracy is not obtainable here. It needs a re-run of
// franka/campaign_ladder.py against the robot.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "circle_detector.h"
#include "circle_tracker_config.h"
#include "contact_localiser.h"
#include "temporal_circle_tracker.h"

#include <opencv2/imgproc.hpp>

namespace {

constexpr int kWidth = 640;
constexpr int kHeight = 480;
constexpr float kPitch = 31.0F;        // measured column pitch, px
constexpr float kRadius = 9.5F;        // measured marker radius, px
constexpr int kCols = 17;
constexpr int kRows = 13;
constexpr float kOriginX = 70.0F;
constexpr float kOriginY = 45.0F;

// Markers are pushed AWAY from the contact. Amplitude and width are set from the
// rig's own measurements: peak marker displacement 19-25 px at depth, deformation
// field half-width ~8 mm at 12.7 px/mm ~= 100 px.
constexpr double kPeakDisplacementPx = 22.0;
// The rig's deformation field has a half-width of ~8 mm (HANDOFF 1165) which at
// 12.7 px/mm is ~100 px, i.e. ~3 marker pitches. Getting this wrong makes the
// divergence lobe narrower than a real one, and contact_localiser's coherence gate
// -- calibrated on real runs at 0.60 -- then correctly rejects it as a one-cell
// spike. A synthetic test has to be realistic in the dimension the gate measures.
constexpr double kFieldSigmaPx = 100.0;

cv::Point2f rest_position(int col, int row) {
    return cv::Point2f(kOriginX + static_cast<float>(col) * kPitch,
                       kOriginY + static_cast<float>(row) * kPitch);
}

// A source field: displacement points radially outward, peaking at ~1 sigma.
cv::Point2f displacement_at(const cv::Point2f &rest, const cv::Point2f &contact) {
    const double dx = rest.x - contact.x;
    const double dy = rest.y - contact.y;
    const double r = std::sqrt(dx * dx + dy * dy);
    if (r < 1e-6) {
        return cv::Point2f(0.0F, 0.0F);
    }
    const double magnitude =
        kPeakDisplacementPx * (r / kFieldSigmaPx) * std::exp(0.5 - 0.5 * (r * r) / (kFieldSigmaPx * kFieldSigmaPx));
    return cv::Point2f(static_cast<float>(magnitude * dx / r), static_cast<float>(magnitude * dy / r));
}

cv::Mat render(const cv::Point2f &contact, bool loaded) {
    cv::Mat occupied = cv::Mat::zeros(kHeight, kWidth, CV_8UC1);
    for (int row = 0; row < kRows; ++row) {
        for (int col = 0; col < kCols; ++col) {
            const cv::Point2f rest = rest_position(col, row);
            cv::Point2f centre = rest;
            if (loaded) {
                centre += displacement_at(rest, contact);
            }
            cv::circle(occupied, cv::Point(cvRound(centre.x), cvRound(centre.y)),
                       cvRound(kRadius), cv::Scalar(255), cv::FILLED, cv::LINE_8);
        }
    }
    return occupied;
}

} // namespace

int main(int argc, char *argv[]) {
    const bool legacy = argc > 1 && std::strcmp(argv[1], "--legacy") == 0;
    printf("search centres: %s\n\n", legacy ? "BASELINE sites (the pre-fix behaviour)"
                                            : "last observed position (current code)");

    e_bts::CircleDetector detector(kWidth, kHeight, e_bts::kDefaultMinimumCircleDensity);
    e_bts::TemporalCircleTracker tracker(detector.expected_radius_px(),
                                         40000 * e_bts::kTemporalFilterWindowCount);

    // --- baseline: unloaded pad, long enough for the map to be built ---------
    Metavision::timestamp t = 0;
    const cv::Mat rest_frame = render(cv::Point2f(0, 0), false);
    for (int i = 0; i < 40; ++i) {
        t += 40000;
        tracker.update(detector.detect(rest_frame).detections, t, i == 0);
    }
    if (!tracker.circle_map_available()) {
        printf("FAIL: no circle map was built from the synthetic lattice\n");
        return 1;
    }
    printf("circle map: %d cols x %d rows, %zu tracked markers, search radius %d px\n\n",
           tracker.circle_map_column_count(), tracker.circle_map_row_count(),
           tracker.baseline_track_count(), tracker.circle_map_search_radius());

    // --- sweep a contact across the field ------------------------------------
    printf("  contact at        | max displacement the tracker still follows | detections\n");
    std::vector<double> true_x, est_x, true_y, est_y, errors;
    int lost = 0;
    for (int step = -4; step <= 4; ++step) {
        const cv::Point2f contact(320.0F + static_cast<float>(step) * 25.0F,
                                  240.0F + static_cast<float>(step) * 18.0F);
        const cv::Mat loaded = render(contact, true);

        e_bts::ContactEstimate estimate;
        for (int i = 0; i < 6; ++i) {   // let the tracker settle onto the deformed field
            t += 40000;
            const auto &centres = legacy ? tracker.circle_map_centers() : tracker.circle_map_search_centers();
            const auto detections = detector.detect_from_map(loaded, centres, tracker.circle_map_radius(),
                                                             tracker.circle_map_search_radius());
            const auto update = tracker.update(detections.detections, t, false);
            estimate = e_bts::localise_contact(update.circles, tracker.circle_map_row_count(),
                                               tracker.circle_map_column_count());
            if (getenv("GAIN_DEBUG")) {
                double max_disp = 0.0;
                for (const auto &c : update.circles) {
                    if (!c.has_baseline) continue;
                    const cv::Point2f d = c.detection.center - c.baseline_center;
                    max_disp = std::max(max_disp, std::sqrt((double)d.dot(d)));
                }
                const auto field = e_bts::build_displacement_field(update.circles,
                        tracker.circle_map_row_count(), tracker.circle_map_column_count());
                const auto div = e_bts::field_divergence(field);
                double max_div = -1e9;
                for (double v : div) if (std::isfinite(v)) max_div = std::max(max_div, v);
                printf("      [iter %d] detections %zu  max|disp| %.2f px  max div %.2f  coh %.2f  valid %d\n",
                       i, detections.detections.size(), max_disp, max_div, estimate.coherence, (int)estimate.valid);
            }
        }
        double max_followed = 0.0;
        {
            const auto &centres = legacy ? tracker.circle_map_centers() : tracker.circle_map_search_centers();
            const auto det = detector.detect_from_map(loaded, centres, tracker.circle_map_radius(),
                                                      tracker.circle_map_search_radius());
            const auto upd = tracker.update(det.detections, t, false);
            for (const auto &c : upd.circles) {
                if (!c.has_baseline) continue;
                const cv::Point2f d = c.detection.center - c.baseline_center;
                max_followed = std::max(max_followed, std::sqrt((double)d.dot(d)));
            }
            printf("  %7.1f %7.1f   |                %6.2f px                   | %4zu\n",
                   contact.x, contact.y, max_followed, det.detections.size());
            errors.push_back(max_followed);
        }

        // release the load, so the next position starts from a settled rest state
        for (int i = 0; i < 6; ++i) {
            t += 40000;
            const auto &centres = legacy ? tracker.circle_map_centers() : tracker.circle_map_search_centers();
            tracker.update(detector.detect_from_map(rest_frame, centres, tracker.circle_map_radius(),
                                                    tracker.circle_map_search_radius()).detections,
                           t, false);
        }
    }


    double best = 0.0;
    for (double e : errors) best = std::max(best, e);
    printf("\n  search radius                    : %d px\n", tracker.circle_map_search_radius());
    printf("  largest displacement followed    : %.2f px\n", best);
    printf("  a marker beside a 5 mm indent moves 15-25 px (HANDOFF 23/24), so anything\n"
           "  at or near the search radius means the contact signal is being truncated.\n");
    printf("\n  %s\n", best > 1.5 * tracker.circle_map_search_radius()
           ? "PASS -- markers are followed well past the search radius."
           : "FAIL -- displacement is capped at the search radius.");
    return 0;
}
