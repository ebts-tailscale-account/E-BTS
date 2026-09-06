#ifndef E_BTS_TEMPORAL_CIRCLE_TRACKER_H
#define E_BTS_TEMPORAL_CIRCLE_TRACKER_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include "circle_detector.h"
#include "circle_lattice.h"
#include "circle_tracker_config.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <opencv2/core.hpp>

namespace e_bts {

struct TrackedCircle {
    CircleDetection detection;
    bool has_baseline = false;
    cv::Point2f baseline_center;
    // Lattice index within the inferred circle map's bounding box, or -1/-1
    // when the baseline fell back to contour tracking and there is no map.
    // Carried through because tracks_ is COMPACTED after the map is built
    // (unobserved sites are erased), so a track's position in the vector no
    // longer implies its cell -- and the contact localiser differentiates the
    // displacement field over cells, not over vector order.
    int column = -1;
    int row    = -1;
};

struct MarkerTrack {
    int column = -1;
    int row    = -1;
    cv::Point2f baseline_center;
    cv::Point2f expected_center;
    cv::Point2f last_observed_center;
    Metavision::timestamp last_observed_us = 0;

    struct TimedDetection {
        Metavision::timestamp timestamp_us = 0;
        CircleDetection detection;
    };
    std::deque<TimedDetection> detection_history;
};

struct MatchCandidate {
    std::size_t track_index     = 0;
    std::size_t detection_index = 0;
    float distance_squared      = 0.0F;
};

struct TrackingUpdate {
    std::vector<TrackedCircle> circles;
    bool baseline_captured = false;
};

struct BaselineCandidate {
    cv::Point2d center_sum;
    double radius_sum       = 0.0;
    double density_sum      = 0.0;
    std::size_t observations = 0;
};

class TemporalCircleTracker {
public:
    TemporalCircleTracker(double expected_radius_px, Metavision::timestamp filter_duration_us) :
        maximum_step_squared_(static_cast<float>(expected_radius_px * kMaximumTrackStepInRadii *
                                                 expected_radius_px * kMaximumTrackStepInRadii)),
        baseline_match_distance_squared_(
            static_cast<float>(expected_radius_px * kBaselineClusterDistanceInExpectedRadii *
                               expected_radius_px * kBaselineClusterDistanceInExpectedRadii)),
        filter_duration_us_(filter_duration_us) {}

    // Builds the baseline from repeated observations across many event windows,
    // then associates each detection with at most one baseline marker. This keeps
    // a noisy or incomplete 10 ms window from permanently defining the marker set.
    TrackingUpdate update(const std::vector<CircleDetection> &detections,
                          Metavision::timestamp timestamp_us, bool restart_baseline) {
        if (restart_baseline) {
            begin_baseline_collection();
        }

        if (baseline_collecting_) {
            collect_baseline_window(detections, timestamp_us);
            if (baseline_collection_ready(timestamp_us)) {
                const bool baseline_captured = capture_baseline(timestamp_us);
                if (baseline_captured) {
                    return TrackingUpdate{make_tracked_circles(timestamp_us), true};
                }
            }

            std::vector<TrackedCircle> visible_detections;
            visible_detections.reserve(detections.size());
            for (const auto &detection : detections) {
                visible_detections.push_back(TrackedCircle{detection, false, {}, -1, -1});
            }
            return TrackingUpdate{std::move(visible_detections), false};
        }

        std::vector<MatchCandidate> match_candidates;
        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
                const cv::Point2f difference =
                    detections[detection_index].center - tracks_[track_index].last_observed_center;
                const float distance_squared = difference.dot(difference);
                if (distance_squared <= maximum_step_squared_) {
                    match_candidates.push_back(MatchCandidate{track_index, detection_index, distance_squared});
                }
            }
        }
        std::sort(match_candidates.begin(), match_candidates.end(), [](const auto &left, const auto &right) {
            return left.distance_squared < right.distance_squared;
        });

        std::vector<bool> track_matched(tracks_.size(), false);
        std::vector<bool> detection_matched(detections.size(), false);
        for (const auto &candidate : match_candidates) {
            if (track_matched[candidate.track_index] || detection_matched[candidate.detection_index]) {
                continue;
            }

            MarkerTrack &track = tracks_[candidate.track_index];
            add_detection(track, detections[candidate.detection_index], timestamp_us);
            track_matched[candidate.track_index]         = true;
            detection_matched[candidate.detection_index] = true;
        }

        TrackingUpdate update_result{make_tracked_circles(timestamp_us), false};
        refresh_search_centers(timestamp_us);
        return update_result;
    }

    bool baseline_collecting() const {
        return baseline_collecting_;
    }

    std::size_t baseline_collection_window_count() const {
        return baseline_collection_window_count_;
    }

    std::size_t stable_baseline_candidate_count() const {
        const std::size_t required_observations = minimum_baseline_observations();
        return static_cast<std::size_t>(std::count_if(
            baseline_candidates_.begin(), baseline_candidates_.end(),
            [required_observations](const auto &candidate) {
                return candidate.observations >= required_observations;
            }));
    }

    std::size_t baseline_track_count() const {
        return tracks_.size();
    }

    std::size_t circle_map_site_count() const {
        return circle_map_geometry_.centers.size();
    }

    std::size_t baseline_observed_circle_count() const {
        return baseline_observed_circle_count_;
    }

    bool circle_map_available() const {
        return circle_map_available_;
    }

    int circle_map_column_count() const {
        return circle_map_geometry_.column_count;
    }

    int circle_map_row_count() const {
        return circle_map_geometry_.row_count;
    }

    float circle_map_radius() const {
        return circle_map_geometry_.mean_radius_px;
    }

    double circle_map_horizontal_pitch() const {
        return cv::norm(circle_map_geometry_.horizontal_step);
    }

    double circle_map_vertical_pitch() const {
        return cv::norm(circle_map_geometry_.vertical_step);
    }

    int circle_map_search_radius() const {
        const double minimum_pitch = std::min(circle_map_horizontal_pitch(), circle_map_vertical_pitch());
        return std::max(2, cvRound(std::min(static_cast<double>(circle_map_radius()),
                                            minimum_pitch * kMapSearchRadiusInPitch)));
    }

    // The BASELINE sites: where each tracked marker rests unloaded. Fixed for the
    // life of a baseline.
    const std::vector<cv::Point2f> &circle_map_centers() const {
        return enabled_circle_map_centers_;
    }

    // Where to LOOK for each marker in the next window -- which is not the same
    // thing, and conflating the two cost this rig a 1100-poke campaign.
    //
    // detect_from_map() only searches +-search_radius (9.3 px here) around each
    // centre it is given. Anchored to the baseline site, that is a hard ceiling on
    // detectable displacement -- but a marker beside a 5 mm indent moves 15-25 px,
    // so precisely the markers carrying the contact signal became undetectable.
    // The measured cost, run ladder_20260906_150159: the estimated contact position
    // moved only 0.62-0.64x as far as the indenter really did, because the
    // divergence peak was computed from a truncated annulus that the field edge
    // then clipped inward. Tracked markers fell 129 -> 114 from 1 mm to 5 mm.
    //
    // So the search follows the MARKER. Per-window motion is small (the arm creeps
    // at 0.03 m/s and windows are 40 ms), while cumulative motion is not, and only
    // the cumulative part was ever the problem.
    //
    // ⚠ A marker that has NOT been seen within the filter window falls back to its
    // baseline site, not to a stale position. Without that, a marker lost at depth
    // keeps being searched for where it was last seen while it has actually sprung
    // back to rest -- and stays lost for the remainder of the run.
    const std::vector<cv::Point2f> &circle_map_search_centers() const {
        return search_centers_;
    }

private:
    void refresh_search_centers(Metavision::timestamp timestamp_us) {
        if (!circle_map_available_) {
            return;
        }
        search_centers_.resize(tracks_.size());
        for (std::size_t index = 0; index < tracks_.size(); ++index) {
            const MarkerTrack &track = tracks_[index];
            const bool seen_recently = !track.detection_history.empty() &&
                                       was_seen_within_filter_window(track, timestamp_us);
            search_centers_[index] = seen_recently ? track.last_observed_center : track.baseline_center;
        }
    }

    void begin_baseline_collection() {
        tracks_.clear();
        baseline_candidates_.clear();
        circle_map_geometry_              = {};
        enabled_circle_map_centers_.clear();
        search_centers_.clear();
        circle_map_available_             = false;
        baseline_observed_circle_count_   = 0;
        baseline_collection_started_      = false;
        baseline_collection_window_count_ = 0;
        baseline_collecting_               = true;
    }

    void collect_baseline_window(const std::vector<CircleDetection> &detections,
                                 Metavision::timestamp timestamp_us) {
        if (!baseline_collection_started_) {
            baseline_collection_start_us_ = timestamp_us;
            baseline_collection_started_  = true;
        }
        ++baseline_collection_window_count_;

        std::vector<MatchCandidate> match_candidates;
        for (std::size_t candidate_index = 0; candidate_index < baseline_candidates_.size(); ++candidate_index) {
            const cv::Point2f candidate_center = average_baseline_detection(
                baseline_candidates_[candidate_index]).center;
            for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
                const cv::Point2f difference = detections[detection_index].center - candidate_center;
                const float distance_squared = difference.dot(difference);
                if (distance_squared <= baseline_match_distance_squared_) {
                    match_candidates.push_back(
                        MatchCandidate{candidate_index, detection_index, distance_squared});
                }
            }
        }
        std::sort(match_candidates.begin(), match_candidates.end(), [](const auto &left, const auto &right) {
            return left.distance_squared < right.distance_squared;
        });

        std::vector<bool> candidate_matched(baseline_candidates_.size(), false);
        std::vector<bool> detection_matched(detections.size(), false);
        for (const auto &match : match_candidates) {
            if (candidate_matched[match.track_index] || detection_matched[match.detection_index]) {
                continue;
            }
            add_baseline_observation(baseline_candidates_[match.track_index],
                                     detections[match.detection_index]);
            candidate_matched[match.track_index]       = true;
            detection_matched[match.detection_index]   = true;
        }

        for (std::size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
            if (detection_matched[detection_index]) {
                continue;
            }
            BaselineCandidate candidate;
            add_baseline_observation(candidate, detections[detection_index]);
            baseline_candidates_.push_back(candidate);
        }
    }

    bool baseline_collection_ready(Metavision::timestamp timestamp_us) const {
        return baseline_collection_started_ &&
               timestamp_us >= baseline_collection_start_us_ &&
               timestamp_us - baseline_collection_start_us_ >= kBaselineCollectionDurationUs &&
               baseline_collection_window_count_ >= kMinimumBaselineCollectionWindows;
    }

    std::size_t minimum_baseline_observations() const {
        const auto fractional_observations = static_cast<std::size_t>(
            std::ceil(static_cast<double>(baseline_collection_window_count_) *
                      kMinimumBaselineObservationFraction));
        return std::max(kMinimumBaselineObservations, fractional_observations);
    }

    bool capture_baseline(Metavision::timestamp timestamp_us) {
        const std::size_t required_observations = minimum_baseline_observations();
        std::vector<const BaselineCandidate *> stable_candidates;
        stable_candidates.reserve(baseline_candidates_.size());
        for (const auto &candidate : baseline_candidates_) {
            if (candidate.observations >= required_observations) {
                stable_candidates.push_back(&candidate);
            }
        }
        std::sort(stable_candidates.begin(), stable_candidates.end(), [](const auto *left, const auto *right) {
            return left->observations > right->observations;
        });

        std::vector<CircleDetection> stable_detections;
        stable_detections.reserve(stable_candidates.size());
        for (const BaselineCandidate *candidate : stable_candidates) {
            const CircleDetection detection = average_baseline_detection(*candidate);
            const bool duplicates_detection =
                std::any_of(stable_detections.begin(), stable_detections.end(), [&](const auto &existing) {
                    const cv::Point2f difference = detection.center - existing.center;
                    return difference.dot(difference) <= baseline_match_distance_squared_;
                });
            if (duplicates_detection) {
                continue;
            }
            stable_detections.push_back(detection);
        }

        if (stable_detections.empty()) {
            return false;
        }
        baseline_observed_circle_count_ = stable_detections.size();

        CircleMapGeometry inferred_geometry;
        circle_map_available_ = infer_circle_map(stable_detections, inferred_geometry);
        if (circle_map_available_) {
            circle_map_geometry_ = std::move(inferred_geometry);
            initialize_map_tracks(stable_detections, timestamp_us);
        } else {
            initialize_observed_tracks(stable_detections, timestamp_us);
        }

        baseline_candidates_.clear();
        baseline_collecting_ = false;
        return true;
    }

    void initialize_observed_tracks(const std::vector<CircleDetection> &detections,
                                    Metavision::timestamp timestamp_us) {
        tracks_.clear();
        tracks_.reserve(detections.size());
        for (const auto &detection : detections) {
            MarkerTrack track;
            track.baseline_center      = detection.center;
            track.expected_center      = detection.center;
            track.last_observed_center = detection.center;
            add_detection(track, detection, timestamp_us);
            tracks_.push_back(std::move(track));
        }
    }

    // Every inferred site becomes a track/search target, but only sites actually
    // observed during baseline receive history. Their observed positions remain
    // displacement zero-points; fitted positions guide later local searches.
    void initialize_map_tracks(const std::vector<CircleDetection> &observed_detections,
                               Metavision::timestamp timestamp_us) {
        tracks_.clear();
        tracks_.reserve(circle_map_geometry_.centers.size());
        // circle_map_geometry_.centers is row-major over the map's bounding box
        // (see infer_circle_map), so the index IS the cell -- but only until the
        // erase below compacts the vector. Stamp it now.
        const int map_columns = std::max(1, circle_map_geometry_.column_count);
        for (std::size_t site_index = 0; site_index < circle_map_geometry_.centers.size(); ++site_index) {
            const cv::Point2f &center = circle_map_geometry_.centers[site_index];
            MarkerTrack track;
            track.column               = static_cast<int>(site_index) % map_columns;
            track.row                  = static_cast<int>(site_index) / map_columns;
            track.baseline_center      = center;
            track.expected_center      = center;
            track.last_observed_center = center;
            tracks_.push_back(std::move(track));
        }

        const float initial_match_distance = std::max(
            circle_map_geometry_.mean_radius_px * kInitialMapMatchDistanceInRadii,
            std::sqrt(baseline_match_distance_squared_));
        const float initial_match_distance_squared = initial_match_distance * initial_match_distance;
        std::vector<MatchCandidate> matches;
        for (std::size_t track_index = 0; track_index < tracks_.size(); ++track_index) {
            for (std::size_t detection_index = 0; detection_index < observed_detections.size();
                 ++detection_index) {
                const cv::Point2f difference =
                    observed_detections[detection_index].center - tracks_[track_index].baseline_center;
                const float distance_squared = difference.dot(difference);
                if (distance_squared <= initial_match_distance_squared) {
                    matches.push_back(MatchCandidate{track_index, detection_index, distance_squared});
                }
            }
        }
        std::sort(matches.begin(), matches.end(), [](const auto &left, const auto &right) {
            return left.distance_squared < right.distance_squared;
        });

        std::vector<bool> track_matched(tracks_.size(), false);
        std::vector<bool> detection_matched(observed_detections.size(), false);
        for (const auto &match : matches) {
            if (track_matched[match.track_index] || detection_matched[match.detection_index]) {
                continue;
            }
            tracks_[match.track_index].baseline_center = observed_detections[match.detection_index].center;
            add_detection(tracks_[match.track_index], observed_detections[match.detection_index], timestamp_us);
            track_matched[match.track_index]         = true;
            detection_matched[match.detection_index] = true;
        }

        // Freeze the visible marker set at baseline time. Inferred sites with no
        // baseline observation (notably the dark middle row) remain in the grid
        // geometry but are never searched, tracked, or rendered afterward.
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [](const auto &track) {
                          return track.detection_history.empty();
                      }),
                      tracks_.end());
        enabled_circle_map_centers_.clear();
        enabled_circle_map_centers_.reserve(tracks_.size());
        for (const auto &track : tracks_) {
            enabled_circle_map_centers_.push_back(track.expected_center);
        }
        // First window after a baseline has nothing observed yet, so the search
        // starts at the resting sites and follows the markers from there.
        search_centers_ = enabled_circle_map_centers_;
    }

    static void add_baseline_observation(BaselineCandidate &candidate,
                                         const CircleDetection &detection) {
        candidate.center_sum.x += detection.center.x;
        candidate.center_sum.y += detection.center.y;
        candidate.radius_sum += detection.radius;
        candidate.density_sum += detection.density;
        ++candidate.observations;
    }

    static CircleDetection average_baseline_detection(const BaselineCandidate &candidate) {
        const double observation_count = static_cast<double>(candidate.observations);
        return CircleDetection{
            cv::Point2f(static_cast<float>(candidate.center_sum.x / observation_count),
                        static_cast<float>(candidate.center_sum.y / observation_count)),
            static_cast<float>(candidate.radius_sum / observation_count),
            candidate.density_sum / observation_count};
    }

    std::vector<TrackedCircle> make_tracked_circles(Metavision::timestamp timestamp_us) {
        std::vector<TrackedCircle> tracked_circles;
        tracked_circles.reserve(tracks_.size());
        for (MarkerTrack &track : tracks_) {
            if (track.detection_history.empty()) {
                continue;
            }
            trim_history(track, timestamp_us);
            if (!was_seen_within_filter_window(track, timestamp_us)) {
                continue;
            }
            tracked_circles.push_back(
                TrackedCircle{current_detection(track), true, track.baseline_center, track.column, track.row});
        }
        return tracked_circles;
    }

    void add_detection(MarkerTrack &track, const CircleDetection &detection,
                       Metavision::timestamp timestamp_us) const {
        track.last_observed_center = detection.center;
        track.last_observed_us     = timestamp_us;
        track.detection_history.push_back(MarkerTrack::TimedDetection{timestamp_us, detection});
        trim_history(track, timestamp_us);
    }

    void trim_history(MarkerTrack &track, Metavision::timestamp timestamp_us) const {
        if (timestamp_us < filter_duration_us_) {
            return;
        }
        const Metavision::timestamp oldest_allowed_us = timestamp_us - filter_duration_us_;
        while (track.detection_history.size() > 1 &&
               track.detection_history.front().timestamp_us <= oldest_allowed_us) {
            track.detection_history.pop_front();
        }
    }

    bool was_seen_within_filter_window(const MarkerTrack &track, Metavision::timestamp timestamp_us) const {
        return timestamp_us >= track.last_observed_us &&
               timestamp_us - track.last_observed_us <= filter_duration_us_;
    }

    // Position must be the latest accepted observation so displacement returns
    // immediately to zero with the marker. Radius and density remain averaged to
    // avoid visual size flicker without introducing lag into the motion vector.
    CircleDetection current_detection(const MarkerTrack &track) const {
        double radius_sum  = 0.0;
        double density_sum = 0.0;
        for (const auto &sample : track.detection_history) {
            radius_sum += sample.detection.radius;
            density_sum += sample.detection.density;
        }

        const double sample_count = static_cast<double>(track.detection_history.size());
        return CircleDetection{track.detection_history.back().detection.center,
                               static_cast<float>(radius_sum / sample_count), density_sum / sample_count};
    }

    float maximum_step_squared_ = 0.0F;
    float baseline_match_distance_squared_ = 0.0F;
    Metavision::timestamp filter_duration_us_ = 0;
    std::vector<MarkerTrack> tracks_;
    std::vector<BaselineCandidate> baseline_candidates_;
    CircleMapGeometry circle_map_geometry_;
    std::vector<cv::Point2f> enabled_circle_map_centers_;
    std::vector<cv::Point2f> search_centers_;
    bool circle_map_available_ = false;
    std::size_t baseline_observed_circle_count_ = 0;
    bool baseline_collecting_          = true;
    bool baseline_collection_started_ = false;
    Metavision::timestamp baseline_collection_start_us_ = 0;
    std::size_t baseline_collection_window_count_ = 0;
};

} // namespace e_bts

#endif // E_BTS_TEMPORAL_CIRCLE_TRACKER_H
