#ifndef E_BTS_CONTACT_LOCALISER_H
#define E_BTS_CONTACT_LOCALISER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "temporal_circle_tracker.h"

#include <opencv2/core.hpp>

// Live contact localisation: where on the elastomer is the indenter?
//
// THIS IS A PORT OF ml/contact_detect.py, DELIBERATELY LINE FOR LINE.
// -------------------------------------------------------------------
// The offline analysis (ml/contact_detect.py, consumed by plot_two_cyl.py and by
// ml/fit_pixel_mm_warp.py, which is where the pixel->mm calibration itself comes
// from) localises a contact as a positive peak of the DIVERGENCE of the marker
// displacement field: markers are pushed AWAY from a contact, so a contact is a
// source. Every accuracy number this rig has ever quoted about localisation was
// measured with that estimator.
//
// If the live readout used a different estimator -- a displacement-weighted
// centroid, say -- it would be a different measurement wearing the same units,
// and the existing validation would not transfer to it. So the maths here is the
// same maths, in the same order, including the three corrections contact_detect.py
// documents as having been learned from wrong answers:
//
//   1. HOLES ARE NOT ZEROS. Untracked cells are missing data, not zero displacement.
//      Differencing across one manufactures a gradient from nothing (it once
//      invented a 7.54 divergence peak in a region with no tracked markers at all),
//      so the derivative there is NaN, not a number.
//   2. A PEAK FINDER ASKED FOR N PEAKS RETURNS N PEAKS. Candidates must earn their
//      place: a real contact is a broad lobe (coherence), and two real contacts are
//      separated by a saddle. A single hot cell is noise.
//   3. FEWER CONTACTS THAN EXPECTED IS A RESULT. `valid == false` is an answer;
//      it must never be reported as (0, 0).
//
// The one thing this adds over the offline version is the second-peak check. The
// GUI advertises SINGLE-point contact, so if a second candidate survives pruning
// the estimate is flagged `ambiguous` rather than silently reporting the stronger
// of two contacts as though it were the only one.

namespace e_bts {

// Divergence below this is not a contact, in px per lattice cell.
//
// ⚠ THIS ONE IS A STARTING VALUE, NOT A MEASURED ONE -- unlike the coherence and
// resolvability thresholds below, which ml/contact_detect.py calibrated against
// real runs and documents the numbers for. It is set where an unloaded pad's
// residual field (uncorrelated sub-pixel tracker jitter, differenced between
// neighbours) should not reach, well under the tens that a 2 mm indent produces.
// Check it rather than trusting it: open Circle Tracking with nothing touching the
// pad, rebuild the baseline, and watch the "div" figure in the status line -- if it
// wanders anywhere near this, raise it. Phantom contacts on an unloaded pad and a
// live readout that ignores a light touch are the two ways it can be wrong.
inline constexpr double kContactMinimumDivergence = 1.5;
inline constexpr int kContactEdgeCells            = 1;    // masked border, as ml/contact_detect.py EDGE
inline constexpr double kContactMinimumResolve    = 0.15; // MIN_RESOLVE
inline constexpr double kContactMinimumCoherence  = 0.60; // MIN_COHERENCE
inline constexpr double kContactMinimumSeparation = 3.0;  // find_peaks min_sep, in cells
inline constexpr int kContactLocalFitRadius       = 2;    // cells each way for the cell->pixel fit
inline constexpr std::size_t kContactMinimumLocalFitCells = 6;

struct ContactEstimate {
    bool valid     = false;   // a contact was found and survived pruning AND cleared min_divergence
    bool ambiguous = false;   // a SECOND candidate also survived -- not a single contact
    double cell_row = 0.0;    // sub-cell peak position in the live lattice
    double cell_col = 0.0;
    cv::Point2d pixel{0.0, 0.0};  // peak in sensor pixels, BASELINE (tare) coordinates
    double divergence   = 0.0;    // peak strength, px per cell
    double coherence    = 0.0;
    int tracked_markers = 0;      // cells with a live displacement this window
    int lattice_rows    = 0;
    int lattice_columns = 0;

    // The two fields below describe the peak REGARDLESS of the divergence
    // threshold, so that a threshold can be chosen after the fact instead of
    // being baked into the run.
    //
    // Before, a window whose peak fell below kContactMinimumDivergence returned
    // an all-zero estimate, so `divergence` read 0.0 and the peak that was
    // actually there was unrecoverable. That made the calibration procedure in
    // the comment above kContactMinimumDivergence -- "watch the div figure and
    // see whether it wanders near this" -- impossible to carry out: below the
    // threshold the figure was pinned at zero by construction. It also meant
    // choosing a different threshold required re-running the robot, since
    // nothing in the log said what the rejected peaks had been.
    //
    // Now the peak is reported whenever it survives PRUNING (the coherence,
    // resolvability and separation gates, which are calibrated and stay fixed);
    // only `valid` depends on min_divergence. One offline replay therefore
    // supports any threshold >= 0 -- see E_BTS_contact_replay.
    bool peak_found = false;      // a peak survived pruning, whatever its strength
    double second_divergence = 0.0;  // runner-up peak, 0 if there was only one.
                                     // `ambiguous` is (second >= min_divergence).
};

// A row-major (rows x cols) displacement field with an occupancy mask, built from
// whatever the tracker saw this window.
struct DisplacementField {
    int rows = 0;
    int cols = 0;
    std::vector<double> dx;
    std::vector<double> dy;
    std::vector<unsigned char> ok;
    std::vector<cv::Point2f> baseline;   // per cell, the marker's undeformed centre

    bool in_range(int r, int c) const {
        return r >= 0 && r < rows && c >= 0 && c < cols;
    }
    std::size_t index(int r, int c) const {
        return static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(c);
    }
};

inline DisplacementField build_displacement_field(const std::vector<TrackedCircle> &circles, int rows,
                                                  int cols) {
    DisplacementField field;
    if (rows < 3 || cols < 3) {
        return field;
    }
    field.rows = rows;
    field.cols = cols;
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    field.dx.assign(count, 0.0);
    field.dy.assign(count, 0.0);
    field.ok.assign(count, 0);
    field.baseline.assign(count, cv::Point2f(0.0F, 0.0F));

    for (const auto &circle : circles) {
        if (!circle.has_baseline || circle.column < 0 || circle.row < 0 ||
            !field.in_range(circle.row, circle.column)) {
            continue;
        }
        const std::size_t index      = field.index(circle.row, circle.column);
        const cv::Point2f displacement = circle.detection.center - circle.baseline_center;
        field.dx[index]       = displacement.x;
        field.dy[index]       = displacement.y;
        field.ok[index]       = 1;
        field.baseline[index] = circle.baseline_center;
    }
    return field;
}

// Central-difference divergence; NaN wherever the stencil touches missing data.
// Mirrors ml/contact_detect.py::divergence, including the border mask.
inline std::vector<double> field_divergence(const DisplacementField &field, int edge = kContactEdgeCells) {
    const std::size_t count = field.dx.size();
    std::vector<double> div(count, 0.0);
    if (field.rows < 3 || field.cols < 3) {
        return div;
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<unsigned char> bad(count, 0);

    for (int r = 0; r < field.rows; ++r) {
        for (int c = 1; c < field.cols - 1; ++c) {
            div[field.index(r, c)] += (field.dx[field.index(r, c + 1)] - field.dx[field.index(r, c - 1)]) / 2.0;
            if (!field.ok[field.index(r, c + 1)] || !field.ok[field.index(r, c - 1)]) {
                bad[field.index(r, c)] = 1;
            }
        }
    }
    for (int r = 1; r < field.rows - 1; ++r) {
        for (int c = 0; c < field.cols; ++c) {
            div[field.index(r, c)] += (field.dy[field.index(r + 1, c)] - field.dy[field.index(r - 1, c)]) / 2.0;
            if (!field.ok[field.index(r + 1, c)] || !field.ok[field.index(r - 1, c)]) {
                bad[field.index(r, c)] = 1;
            }
        }
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (bad[index] || !field.ok[index]) {
            div[index] = nan;
        }
    }
    for (int r = 0; r < field.rows; ++r) {
        for (int c = 0; c < field.cols; ++c) {
            if (r < edge || r >= field.rows - edge || c < edge || c >= field.cols - edge) {
                div[field.index(r, c)] = nan;
            }
        }
    }
    return div;
}

// 3x3 mean over the peak value: how LOBE-like a candidate is. ml/contact_detect.py
// measured 0.71-0.83 for genuine contacts and 0.42 for a one-cell spike.
inline double peak_coherence(const std::vector<double> &div, const DisplacementField &field, double row,
                             double col) {
    const int r = static_cast<int>(std::lround(row));
    const int c = static_cast<int>(std::lround(col));
    if (!field.in_range(r, c)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double centre = div[field.index(r, c)];
    if (!(centre > 0.0)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum       = 0.0;
    std::size_t seen = 0;
    for (int rr = std::max(0, r - 1); rr <= std::min(field.rows - 1, r + 1); ++rr) {
        for (int cc = std::max(0, c - 1); cc <= std::min(field.cols - 1, c + 1); ++cc) {
            const double value = div[field.index(rr, cc)];
            if (std::isfinite(value)) {
                sum += value;
                ++seen;
            }
        }
    }
    if (seen == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return (sum / static_cast<double>(seen)) / centre;
}

// Minimum divergence strictly between two peaks -- the dip that separates them.
inline double saddle_between(const std::vector<double> &div, const DisplacementField &field, double r0,
                             double c0, double r1, double c1) {
    double minimum   = std::numeric_limits<double>::infinity();
    bool sampled_any = false;
    for (int step = 0; step <= 119; ++step) {
        const double t = static_cast<double>(step) / 119.0;
        if (t <= 0.05 || t >= 0.95) {
            continue;
        }
        const int r = static_cast<int>(std::lround(r0 + t * (r1 - r0)));
        const int c = static_cast<int>(std::lround(c0 + t * (c1 - c0)));
        if (!field.in_range(r, c)) {
            continue;
        }
        const double value = div[field.index(r, c)];
        if (std::isfinite(value)) {
            minimum     = std::min(minimum, value);
            sampled_any = true;
        }
    }
    return sampled_any ? minimum : std::numeric_limits<double>::quiet_NaN();
}

struct DivergencePeak {
    double value = 0.0;
    double row   = 0.0;
    double col   = 0.0;
};

// Up to `wanted` well-separated positive maxima, divergence-weighted 3x3 refined.
inline std::vector<DivergencePeak> find_divergence_peaks(const std::vector<double> &div,
                                                         const DisplacementField &field, int wanted) {
    std::vector<DivergencePeak> ordered;
    ordered.reserve(div.size());
    for (int r = 0; r < field.rows; ++r) {
        for (int c = 0; c < field.cols; ++c) {
            const double value = div[field.index(r, c)];
            if (std::isfinite(value) && value > 0.0) {
                ordered.push_back(DivergencePeak{value, static_cast<double>(r), static_cast<double>(c)});
            }
        }
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const DivergencePeak &left, const DivergencePeak &right) { return left.value > right.value; });

    std::vector<DivergencePeak> peaks;
    for (const auto &candidate : ordered) {
        const bool far_enough = std::all_of(peaks.begin(), peaks.end(), [&](const DivergencePeak &peak) {
            const double dr = candidate.row - peak.row;
            const double dc = candidate.col - peak.col;
            return dr * dr + dc * dc > kContactMinimumSeparation * kContactMinimumSeparation;
        });
        if (far_enough) {
            peaks.push_back(candidate);
        }
        if (static_cast<int>(peaks.size()) == wanted) {
            break;
        }
    }

    for (auto &peak : peaks) {  // divergence-weighted 3x3 centroid -> sub-cell
        const int r = static_cast<int>(peak.row);
        const int c = static_cast<int>(peak.col);
        double weight_sum = 0.0;
        double row_sum    = 0.0;
        double col_sum    = 0.0;
        for (int rr = std::max(0, r - 1); rr <= std::min(field.rows - 1, r + 1); ++rr) {
            for (int cc = std::max(0, c - 1); cc <= std::min(field.cols - 1, c + 1); ++cc) {
                const double value = div[field.index(rr, cc)];
                const double w     = std::isfinite(value) ? std::max(0.0, value) : 0.0;
                weight_sum += w;
                row_sum += rr * w;
                col_sum += cc * w;
            }
        }
        if (weight_sum > 0.0) {
            peak.row = row_sum / weight_sum;
            peak.col = col_sum / weight_sum;
        }
    }
    return peaks;
}

// Keep only candidates a real contact would produce. May return fewer than given.
inline std::vector<DivergencePeak> prune_unsupported(const std::vector<DivergencePeak> &peaks,
                                                     const std::vector<double> &div,
                                                     const DisplacementField &field) {
    std::vector<DivergencePeak> kept;
    for (const auto &candidate : peaks) {
        const double coherence = peak_coherence(div, field, candidate.row, candidate.col);
        if (std::isfinite(coherence) && coherence < kContactMinimumCoherence) {
            continue;  // a one-cell spike, not a lobe
        }
        bool supported = true;
        for (const auto &existing : kept) {
            const double saddle = saddle_between(div, field, candidate.row, candidate.col, existing.row,
                                                 existing.col);
            const double weaker = std::min(candidate.value, existing.value);
            if (!std::isfinite(saddle) || weaker <= 0.0 || (1.0 - saddle / weaker) < kContactMinimumResolve) {
                supported = false;  // a shoulder of the stronger peak, not its own contact
                break;
            }
        }
        if (supported) {
            kept.push_back(candidate);
        }
    }
    return kept;
}

// Sub-cell lattice position -> sensor pixel, via a local affine fit of the
// BASELINE marker centres around the peak.
//
// Bilinear interpolation of the four surrounding baseline centres would be the
// obvious choice and is wrong here: the frozen marker set has holes (the dark
// middle row, and any marker the baseline never saw), and one missing corner would
// leave the peak unmappable. A plane fitted over the 5x5 neighbourhood degrades
// instead of failing, and over two cells the lattice is affine to well under a
// pixel anyway.
//
// BASELINE, not current, centres: ml/blue_circle_grid indexes displacement by the
// cell a marker occupied in the TARE frame, so the divergence field lives in
// undeformed material coordinates -- and the pixel->mm calibration's node
// positions are tare positions too. Mapping through deformed centres would mix
// the two frames.
inline bool lattice_cell_to_pixel(const DisplacementField &field, double row, double col,
                                  cv::Point2d &pixel) {
    const int r0 = static_cast<int>(std::lround(row));
    const int c0 = static_cast<int>(std::lround(col));

    // Normal equations for px = a0 + a1*col + a2*row, accumulated in doubles.
    double ata[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    double atx[3]    = {0, 0, 0};
    double aty[3]    = {0, 0, 0};
    std::size_t used = 0;
    for (int rr = r0 - kContactLocalFitRadius; rr <= r0 + kContactLocalFitRadius; ++rr) {
        for (int cc = c0 - kContactLocalFitRadius; cc <= c0 + kContactLocalFitRadius; ++cc) {
            if (!field.in_range(rr, cc) || !field.ok[field.index(rr, cc)]) {
                continue;
            }
            const cv::Point2f &base = field.baseline[field.index(rr, cc)];
            const double basis[3]   = {1.0, static_cast<double>(cc), static_cast<double>(rr)};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    ata[i][j] += basis[i] * basis[j];
                }
                atx[i] += basis[i] * base.x;
                aty[i] += basis[i] * base.y;
            }
            ++used;
        }
    }
    if (used < kContactMinimumLocalFitCells) {
        return false;
    }

    cv::Matx33d normal(ata[0][0], ata[0][1], ata[0][2], ata[1][0], ata[1][1], ata[1][2], ata[2][0],
                       ata[2][1], ata[2][2]);
    cv::Vec3d solution_x;
    cv::Vec3d solution_y;
    if (!cv::solve(normal, cv::Vec3d(atx[0], atx[1], atx[2]), solution_x, cv::DECOMP_SVD) ||
        !cv::solve(normal, cv::Vec3d(aty[0], aty[1], aty[2]), solution_y, cv::DECOMP_SVD)) {
        return false;
    }
    pixel.x = solution_x[0] + solution_x[1] * col + solution_x[2] * row;
    pixel.y = solution_y[0] + solution_y[1] * col + solution_y[2] * row;
    return true;
}

// The whole estimator, one window in, one answer out.
// min_divergence defaults to the live constant, so the GUI's behaviour is
// unchanged; E_BTS_contact_replay passes its own so a threshold can be swept
// offline. Only `valid`/`ambiguous` depend on it -- see ContactEstimate.
inline ContactEstimate localise_contact(const std::vector<TrackedCircle> &circles, int lattice_rows,
                                        int lattice_columns,
                                        double min_divergence = kContactMinimumDivergence) {
    ContactEstimate estimate;
    estimate.lattice_rows    = lattice_rows;
    estimate.lattice_columns = lattice_columns;

    const DisplacementField field = build_displacement_field(circles, lattice_rows, lattice_columns);
    if (field.rows == 0) {
        return estimate;   // no circle map: no cells to differentiate over
    }
    estimate.tracked_markers =
        static_cast<int>(std::count(field.ok.begin(), field.ok.end(), static_cast<unsigned char>(1)));

    const std::vector<double> div = field_divergence(field);
    // Two, not one: the second is only ever used to decide whether the first is
    // ALONE. Reporting the stronger of two contacts as "the" contact is the
    // failure mode this catches.
    const std::vector<DivergencePeak> kept = prune_unsupported(find_divergence_peaks(div, field, 2), div, field);
    if (kept.empty()) {
        return estimate;
    }

    const DivergencePeak &peak = kept.front();
    cv::Point2d pixel;
    // The cell->pixel fit is what makes a peak reportable at all: without it
    // there is a cell index but no sensor position, so there is nothing to
    // record at any threshold.
    if (!lattice_cell_to_pixel(field, peak.row, peak.col, pixel)) {
        return estimate;
    }

    // Recorded whatever the threshold; see ContactEstimate::peak_found.
    estimate.peak_found        = true;
    estimate.cell_row          = peak.row;
    estimate.cell_col          = peak.col;
    estimate.pixel             = pixel;
    estimate.divergence        = peak.value;
    estimate.second_divergence = kept.size() > 1 ? kept[1].value : 0.0;
    estimate.coherence         = peak_coherence(div, field, peak.row, peak.col);

    estimate.valid     = peak.value >= min_divergence;
    estimate.ambiguous = estimate.valid && estimate.second_divergence >= min_divergence;
    return estimate;
}

} // namespace e_bts

#endif // E_BTS_CONTACT_LOCALISER_H
