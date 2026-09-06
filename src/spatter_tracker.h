#ifndef E_BTS_SPATTER_TRACKER_H
#define E_BTS_SPATTER_TRACKER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

#include "event_window_buffer.h"
#include "spatter_tracker_config.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <opencv2/core.hpp>

// Spatter tracking: follow small, non-colliding moving objects through the event
// stream and give each a stable id for as long as it stays visible.
//
// WHY THIS IS HAND-WRITTEN RATHER THAN A CALL INTO THE SDK
// -------------------------------------------------------
// Prophesee ships exactly this as Metavision::SpatterTrackerAlgorithm, but that
// class lives in the SDK's *analytics* module, which is closed-source and only
// distributed as prebuilt Debian packages from apt.prophesee.ai. This project
// builds against OpenEB 3.1.2, whose four modules -- base, core, driver, ui --
// contain no analytics and no clustering of any kind. There is nothing to call,
// so the documented algorithm is reimplemented here on top of OpenEB's base
// event types plus OpenCV for drawing only.
//
// THE ALGORITHM, PER WINDOW
// -------------------------
//   0. Discard every event outside the ROI rectangle. Not in the SDK's version;
//      added because this rig only ever wants the marker band, and cropping at
//      the front makes the ROI bound cost as well as output.
//   1. Overlay the sensor with a grid of cell_width x cell_height cells.
//   2. A cell is ACTIVE if at least activation_threshold of its pixels fired.
//   3. Group 8-connected active cells into clusters.
//   4. Drop clusters whose bounding box falls outside [min_size, max_size].
//   5. Associate surviving clusters with the previous window's tracks, nearest
//      pair first, accepting only pairs closer than max_distance.
//   6. Unmatched clusters become new tracks; tracks that go unmatched for more
//      than untracked_threshold consecutive windows are retired.
//
// Step 2 counts DISTINCT PIXELS, not raw events -- a pixel that fires ten times
// in one window still contributes one. That is deliberate and matches the
// sample's `--apply-filter true` default; it is also free here, because
// EventWindowBuffer already collapses each window to a set of occupied pixel
// indices before this class ever sees it.
//
// WHERE THIS BREAKS. The association step is nearest-neighbour with no motion
// model, so it holds only while objects stay farther apart than they move per
// window. Two objects passing within max_distance of each other can swap ids,
// and there is no way to detect that after the fact -- which is precisely why
// the SDK documents its equivalent as suiting "only non-colliding objects".
// Raising max_distance to survive fast motion makes swaps more likely, and
// lowering it to prevent swaps breaks tracks on fast motion. If both cannot
// hold at once, shorten the accumulation time instead: it reduces per-window
// travel without touching the separation the gate has to respect.

namespace e_bts {

// Live tuning knobs, all adjustable while the camera runs. See
// spatter_tracker_config.h for defaults and the ranges these clamp to.
struct SpatterTrackerParams {
    int cell_width           = kDefaultSpatterCellWidth;
    int cell_height          = kDefaultSpatterCellHeight;
    int activation_threshold = kDefaultSpatterActivationThreshold;
    int min_size             = kDefaultSpatterMinSizePx;
    int max_size             = kDefaultSpatterMaxSizePx;
    int max_distance         = kDefaultSpatterMaxDistancePx;
    int untracked_threshold  = kDefaultSpatterUntrackedThreshold;

    // Software crop, in full-sensor coordinates. Events landing outside it are
    // discarded before anything else runs, so the ROI bounds cost as well as
    // output: a band a tenth of the sensor's height costs about a tenth of the
    // accumulation. Clipped to the sensor by clamp_params(), never rejected.
    int roi_x      = kDefaultSpatterRoiX;
    int roi_y      = kDefaultSpatterRoiY;
    int roi_width  = kDefaultSpatterRoiWidth;
    int roi_height = kDefaultSpatterRoiHeight;
};

// One tracked object as of the window just processed. The equivalent of the
// SDK's Metavision::EventSpatterCluster.
struct SpatterCluster {
    std::uint32_t id = 0;
    cv::Point2f center;                    // event-weighted centroid, px
    cv::Rect box;                          // tight bounding box over fired pixels
    std::size_t pixel_count = 0;           // distinct pixels that fired
    Metavision::timestamp t  = 0;          // end of the window it was seen in
    std::uint32_t window_count = 0;        // windows this track has been matched in
    std::vector<cv::Point2f> trail;        // recent centroids, oldest first
};

// Counters for the on-frame status line: they explain *why* a frame looks empty,
// which is otherwise a guessing game between "no events", "cells never reached
// the activation threshold", and "clusters all failed the size filter".
struct SpatterDiagnostics {
    std::size_t occupied_pixels    = 0;
    std::size_t pixels_in_roi      = 0;  // of those, the ones the ROI kept
    std::size_t active_cells       = 0;
    std::size_t raw_clusters       = 0;
    std::size_t too_small          = 0;
    std::size_t too_large          = 0;
    std::size_t new_tracks         = 0;
    std::size_t retired_tracks     = 0;
    std::size_t coasting_tracks    = 0;  // alive but unmatched this window
};

class SpatterTracker {
public:
    SpatterTracker(int width, int height, const SpatterTrackerParams &params) :
        width_(width), height_(height), params_(clamp_params(params, width, height)) {
        rebuild_grid();
    }

    // Consumes one event window and returns the tracks visible in it. The
    // returned reference stays valid until the next update() call.
    const std::vector<SpatterCluster> &update(const EventWindow &event_window) {
        diagnostics_ = SpatterDiagnostics{};
        diagnostics_.occupied_pixels = event_window.occupied_pixel_indices.size();

        accumulate_cells(event_window);
        find_clusters();
        associate(event_window.end_us);
        return live_clusters_;
    }

    void set_params(const SpatterTrackerParams &params) {
        const SpatterTrackerParams clamped = clamp_params(params, width_, height_);
        const bool grid_changed =
            clamped.cell_width != params_.cell_width || clamped.cell_height != params_.cell_height;
        params_ = clamped;
        if (grid_changed) {
            rebuild_grid();
            // Cell geometry changed under the tracks' feet: their centroids are
            // still valid pixel coordinates, so they are kept rather than reset.
            // Only the grid scratch buffers are invalidated.
        }
    }

    const SpatterTrackerParams &params() const {
        return params_;
    }

    // The clipped ROI actually in force, for drawing it on the frame. Always
    // inside the sensor, so it can be used as a cv::Mat sub-rect unchecked.
    cv::Rect roi() const {
        return cv::Rect(params_.roi_x, params_.roi_y, params_.roi_width, params_.roi_height);
    }

    const SpatterDiagnostics &diagnostics() const {
        return diagnostics_;
    }

    std::size_t track_count() const {
        return tracks_.size();
    }

    // Forgets every track and restarts id numbering. Used when the pane is
    // reopened so a fresh session does not begin at id 4211.
    void reset() {
        tracks_.clear();
        live_clusters_.clear();
        next_track_id_ = 1;
        diagnostics_   = SpatterDiagnostics{};
    }

private:
    // Per-cell accumulator. Reset lazily via the stamp trick (see stamp_), so a
    // window only pays for the cells it actually touches -- at cell_width 1 the
    // grid has one entry per pixel and clearing it wholesale every window would
    // cost more than the clustering.
    struct Cell {
        std::uint32_t stamp  = 0;
        std::int32_t count   = 0;
        std::int64_t sum_x   = 0;
        std::int64_t sum_y   = 0;
        std::int32_t min_x   = 0;
        std::int32_t max_x   = 0;
        std::int32_t min_y   = 0;
        std::int32_t max_y   = 0;
        bool active          = false;
        bool visited         = false;
    };

    struct Track {
        std::uint32_t id = 0;
        cv::Point2f center;
        cv::Rect box;
        std::size_t pixel_count    = 0;
        std::uint32_t window_count = 0;
        int missed                 = 0;
        std::deque<cv::Point2f> trail;
    };

    struct Detection {
        cv::Point2f center;
        cv::Rect box;
        std::size_t pixel_count = 0;
    };

    static SpatterTrackerParams clamp_params(const SpatterTrackerParams &params, int width, int height) {
        SpatterTrackerParams clamped = params;
        clamped.cell_width  = std::clamp(clamped.cell_width, kMinimumSpatterCellSize, kMaximumSpatterCellSize);
        clamped.cell_height = std::clamp(clamped.cell_height, kMinimumSpatterCellSize, kMaximumSpatterCellSize);
        clamped.activation_threshold = std::clamp(clamped.activation_threshold,
                                                  kMinimumSpatterActivationThreshold,
                                                  kMaximumSpatterActivationThreshold);
        clamped.min_size = std::clamp(clamped.min_size, kMinimumSpatterSizePx, kMaximumSpatterSizePx);
        clamped.max_size = std::clamp(clamped.max_size, kMinimumSpatterSizePx, kMaximumSpatterSizePx);
        // A min above max would reject everything with no visible cause; treat
        // the pair as an unordered range instead.
        if (clamped.min_size > clamped.max_size) {
            std::swap(clamped.min_size, clamped.max_size);
        }
        clamped.max_distance = std::clamp(clamped.max_distance, kMinimumSpatterMaxDistancePx,
                                          kMaximumSpatterMaxDistancePx);
        clamped.untracked_threshold = std::clamp(clamped.untracked_threshold,
                                                 kMinimumSpatterUntrackedThreshold,
                                                 kMaximumSpatterUntrackedThreshold);

        // Clip the ROI into the sensor. The origin is pulled inside first, then
        // the extent is trimmed to whatever remains to the right of / below it,
        // so an over-wide rectangle (the shipped default is one: x 6 + w 640 =
        // 646 on a 640 px sensor) simply stops at the edge instead of reading
        // out of bounds or being rejected outright.
        clamped.roi_x      = std::clamp(clamped.roi_x, 0, std::max(0, width - kMinimumSpatterRoiSizePx));
        clamped.roi_y      = std::clamp(clamped.roi_y, 0, std::max(0, height - kMinimumSpatterRoiSizePx));
        clamped.roi_width  = std::clamp(clamped.roi_width, kMinimumSpatterRoiSizePx, width - clamped.roi_x);
        clamped.roi_height = std::clamp(clamped.roi_height, kMinimumSpatterRoiSizePx, height - clamped.roi_y);
        return clamped;
    }

    void rebuild_grid() {
        columns_ = (width_ + params_.cell_width - 1) / params_.cell_width;
        rows_    = (height_ + params_.cell_height - 1) / params_.cell_height;
        cells_.assign(static_cast<std::size_t>(columns_) * static_cast<std::size_t>(rows_), Cell{});
        stamp_ = 0;
        touched_cells_.clear();
    }

    void accumulate_cells(const EventWindow &event_window) {
        // Stamps are compared for equality only, so wrapping is harmless as long
        // as no cell carries a stamp from exactly 2^32 windows ago -- which the
        // full clear below rules out.
        if (++stamp_ == 0) {
            std::fill(cells_.begin(), cells_.end(), Cell{});
            stamp_ = 1;
        }
        touched_cells_.clear();

        const int cell_width  = params_.cell_width;
        const int cell_height = params_.cell_height;
        // Half-open bounds, precomputed so the inner loop tests four ints rather
        // than reconstructing the rectangle per event.
        const std::int32_t roi_min_x = params_.roi_x;
        const std::int32_t roi_min_y = params_.roi_y;
        const std::int32_t roi_end_x = params_.roi_x + params_.roi_width;
        const std::int32_t roi_end_y = params_.roi_y + params_.roi_height;
        for (const std::uint32_t pixel_index : event_window.occupied_pixel_indices) {
            const std::int32_t x = static_cast<std::int32_t>(pixel_index % static_cast<std::uint32_t>(width_));
            const std::int32_t y = static_cast<std::int32_t>(pixel_index / static_cast<std::uint32_t>(width_));
            // Rejected here rather than after clustering: a pixel outside the
            // ROI must not contribute to a cell's activation count either, or a
            // cell straddling the boundary would activate on events the user
            // asked to ignore.
            if (x < roi_min_x || x >= roi_end_x || y < roi_min_y || y >= roi_end_y) {
                continue;
            }
            ++diagnostics_.pixels_in_roi;

            const std::size_t cell_index =
                static_cast<std::size_t>(y / cell_height) * static_cast<std::size_t>(columns_) +
                static_cast<std::size_t>(x / cell_width);

            Cell &cell = cells_[cell_index];
            if (cell.stamp != stamp_) {
                cell = Cell{};
                cell.stamp = stamp_;
                cell.min_x = cell.max_x = x;
                cell.min_y = cell.max_y = y;
                touched_cells_.push_back(cell_index);
            }
            ++cell.count;
            cell.sum_x += x;
            cell.sum_y += y;
            cell.min_x = std::min(cell.min_x, x);
            cell.max_x = std::max(cell.max_x, x);
            cell.min_y = std::min(cell.min_y, y);
            cell.max_y = std::max(cell.max_y, y);
        }

        for (const std::size_t cell_index : touched_cells_) {
            Cell &cell  = cells_[cell_index];
            cell.active = cell.count >= params_.activation_threshold;
            if (cell.active) {
                ++diagnostics_.active_cells;
            }
        }
    }

    // 8-connected flood fill over active cells, seeded from the touched list so
    // the sweep is proportional to activity rather than to grid size.
    void find_clusters() {
        detections_.clear();

        for (const std::size_t seed_index : touched_cells_) {
            Cell &seed = cells_[seed_index];
            if (!seed.active || seed.visited) {
                continue;
            }

            seed.visited = true;
            flood_stack_.clear();
            flood_stack_.push_back(seed_index);

            std::int64_t sum_x        = 0;
            std::int64_t sum_y        = 0;
            std::size_t pixel_count   = 0;
            std::int32_t min_x        = width_;
            std::int32_t max_x        = -1;
            std::int32_t min_y        = height_;
            std::int32_t max_y        = -1;

            while (!flood_stack_.empty()) {
                const std::size_t cell_index = flood_stack_.back();
                flood_stack_.pop_back();
                const Cell &cell = cells_[cell_index];

                sum_x += cell.sum_x;
                sum_y += cell.sum_y;
                pixel_count += static_cast<std::size_t>(cell.count);
                min_x = std::min(min_x, cell.min_x);
                max_x = std::max(max_x, cell.max_x);
                min_y = std::min(min_y, cell.min_y);
                max_y = std::max(max_y, cell.max_y);

                const int column = static_cast<int>(cell_index % static_cast<std::size_t>(columns_));
                const int row    = static_cast<int>(cell_index / static_cast<std::size_t>(columns_));
                for (int row_offset = -1; row_offset <= 1; ++row_offset) {
                    const int neighbor_row = row + row_offset;
                    if (neighbor_row < 0 || neighbor_row >= rows_) {
                        continue;
                    }
                    for (int column_offset = -1; column_offset <= 1; ++column_offset) {
                        if (row_offset == 0 && column_offset == 0) {
                            continue;
                        }
                        const int neighbor_column = column + column_offset;
                        if (neighbor_column < 0 || neighbor_column >= columns_) {
                            continue;
                        }
                        const std::size_t neighbor_index =
                            static_cast<std::size_t>(neighbor_row) * static_cast<std::size_t>(columns_) +
                            static_cast<std::size_t>(neighbor_column);
                        Cell &neighbor = cells_[neighbor_index];
                        if (neighbor.stamp != stamp_ || !neighbor.active || neighbor.visited) {
                            continue;
                        }
                        neighbor.visited = true;
                        flood_stack_.push_back(neighbor_index);
                    }
                }
            }

            ++diagnostics_.raw_clusters;

            const int box_width  = max_x - min_x + 1;
            const int box_height = max_y - min_y + 1;
            // Both dimensions must fit the range, matching the SDK sample's
            // "objects from 10x10 to 300x300 px" reading of --min-size/--max-size.
            // A long thin streak is therefore rejected on its short side; widen
            // min_size' lower bound rather than expecting elongated blobs through.
            if (box_width < params_.min_size || box_height < params_.min_size) {
                ++diagnostics_.too_small;
                continue;
            }
            if (box_width > params_.max_size || box_height > params_.max_size) {
                ++diagnostics_.too_large;
                continue;
            }

            Detection detection;
            detection.center = cv::Point2f(static_cast<float>(static_cast<double>(sum_x) / pixel_count),
                                           static_cast<float>(static_cast<double>(sum_y) / pixel_count));
            detection.box         = cv::Rect(min_x, min_y, box_width, box_height);
            detection.pixel_count = pixel_count;
            detections_.push_back(detection);
        }
    }

    // Greedy global nearest-neighbour: every (track, detection) pair within the
    // gate is scored, then the closest surviving pair is committed repeatedly.
    // Cheaper than the Hungarian algorithm and, unlike per-track greedy, its
    // result does not depend on the order tracks happen to sit in the vector.
    void associate(Metavision::timestamp window_end_us) {
        const float gate         = static_cast<float>(params_.max_distance);
        const float gate_squared = gate * gate;

        candidate_pairs_.clear();
        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            for (std::size_t detection_index = 0; detection_index < detections_.size(); ++detection_index) {
                const cv::Point2f delta = detections_[detection_index].center - tracks_[track_index].center;
                const float distance_squared = delta.dot(delta);
                if (distance_squared <= gate_squared) {
                    candidate_pairs_.push_back({distance_squared, track_index, detection_index});
                }
            }
        }
        std::sort(candidate_pairs_.begin(), candidate_pairs_.end(),
                  [](const CandidatePair &left, const CandidatePair &right) {
                      return left.distance_squared < right.distance_squared;
                  });

        track_matched_.assign(tracks_.size(), false);
        detection_matched_.assign(detections_.size(), false);
        for (const CandidatePair &pair : candidate_pairs_) {
            if (track_matched_[pair.track_index] || detection_matched_[pair.detection_index]) {
                continue;
            }
            track_matched_[pair.track_index]         = true;
            detection_matched_[pair.detection_index] = true;

            Track &track           = tracks_[pair.track_index];
            const Detection &found = detections_[pair.detection_index];
            track.center           = found.center;
            track.box              = found.box;
            track.pixel_count      = found.pixel_count;
            track.missed           = 0;
            ++track.window_count;
            push_trail(track);
        }

        // Unmatched detections become new tracks. Done before the retirement
        // sweep so a brand-new track is never a candidate for its own removal.
        for (std::size_t detection_index = 0; detection_index < detections_.size(); ++detection_index) {
            if (detection_matched_[detection_index]) {
                continue;
            }
            const Detection &found = detections_[detection_index];
            Track track;
            track.id           = next_track_id_++;
            track.center       = found.center;
            track.box          = found.box;
            track.pixel_count  = found.pixel_count;
            track.window_count = 1;
            track.missed       = 0;
            push_trail(track);
            tracks_.push_back(std::move(track));
            track_matched_.push_back(true);
            ++diagnostics_.new_tracks;
        }

        // Age the tracks that went unseen, and retire the ones past the limit.
        // A coasting track keeps its last known position: it is deliberately not
        // extrapolated, because a wrong prediction would drag the association
        // gate away from where the object actually reappears.
        const std::size_t before = tracks_.size();
        std::size_t write_index  = 0;
        for (std::size_t read_index = 0; read_index < tracks_.size(); ++read_index) {
            if (!track_matched_[read_index]) {
                if (++tracks_[read_index].missed > params_.untracked_threshold) {
                    continue;
                }
                ++diagnostics_.coasting_tracks;
            }
            if (write_index != read_index) {
                tracks_[write_index] = std::move(tracks_[read_index]);
                track_matched_[write_index] = track_matched_[read_index];
            }
            ++write_index;
        }
        tracks_.resize(write_index);
        track_matched_.resize(write_index);
        diagnostics_.retired_tracks = before - write_index;

        // Only tracks seen in *this* window are published, so a coasting track
        // does not paint a stale box over the live view.
        live_clusters_.clear();
        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            if (!track_matched_[track_index]) {
                continue;
            }
            const Track &track = tracks_[track_index];
            SpatterCluster cluster;
            cluster.id           = track.id;
            cluster.center       = track.center;
            cluster.box          = track.box;
            cluster.pixel_count  = track.pixel_count;
            cluster.t            = window_end_us;
            cluster.window_count = track.window_count;
            cluster.trail.assign(track.trail.begin(), track.trail.end());
            live_clusters_.push_back(std::move(cluster));
        }
    }

    static void push_trail(Track &track) {
        track.trail.push_back(track.center);
        while (track.trail.size() > kSpatterTrailLength) {
            track.trail.pop_front();
        }
    }

    struct CandidatePair {
        float distance_squared;
        std::size_t track_index;
        std::size_t detection_index;
    };

    int width_;
    int height_;
    SpatterTrackerParams params_;

    int columns_ = 0;
    int rows_    = 0;
    std::vector<Cell> cells_;
    std::uint32_t stamp_ = 0;
    std::vector<std::size_t> touched_cells_;
    std::vector<std::size_t> flood_stack_;

    std::vector<Detection> detections_;
    std::vector<Track> tracks_;
    std::vector<CandidatePair> candidate_pairs_;
    std::vector<bool> track_matched_;
    std::vector<bool> detection_matched_;

    std::vector<SpatterCluster> live_clusters_;
    std::uint32_t next_track_id_ = 1;
    SpatterDiagnostics diagnostics_;
};

} // namespace e_bts

#endif // E_BTS_SPATTER_TRACKER_H
