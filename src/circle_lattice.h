#ifndef E_BTS_CIRCLE_LATTICE_H
#define E_BTS_CIRCLE_LATTICE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "circle_detector.h"
#include "circle_tracker_config.h"

#include <opencv2/core.hpp>

namespace e_bts {

struct LatticeObservation {
    int column = 0;
    int row    = 0;
    CircleDetection detection;
    float residual_squared = 0.0F;
};

inline double median_value(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto middle = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 != 0) {
        return *middle;
    }
    const double upper = *middle;
    const double lower = *std::max_element(values.begin(), middle);
    return (lower + upper) * 0.5;
}

// Estimates one lattice basis vector from nearest neighbors that are aligned
// on the other image axis. A median first rejects gaps caused by missed markers;
// the final vector is the mean of neighbors close to that fundamental pitch.
inline bool estimate_lattice_step(const std::vector<CircleDetection> &detections, float mean_radius,
                                  bool horizontal, cv::Point2f &step) {
    const float minimum_separation = mean_radius * kLatticeMinimumNeighborDistanceInRadii;
    const float alignment_tolerance = std::max(3.0F, mean_radius * kLatticeAlignmentToleranceInRadii);
    std::vector<cv::Point2f> nearest_offsets;
    nearest_offsets.reserve(detections.size());

    for (const auto &origin : detections) {
        float nearest_primary_distance = std::numeric_limits<float>::max();
        cv::Point2f nearest_offset;
        bool neighbor_found = false;
        for (const auto &candidate : detections) {
            const cv::Point2f offset = candidate.center - origin.center;
            const float primary = horizontal ? offset.x : offset.y;
            const float cross   = horizontal ? std::abs(offset.y) : std::abs(offset.x);
            if (primary <= minimum_separation || cross > alignment_tolerance ||
                primary >= nearest_primary_distance) {
                continue;
            }
            nearest_primary_distance = primary;
            nearest_offset           = offset;
            neighbor_found           = true;
        }
        if (neighbor_found) {
            nearest_offsets.push_back(nearest_offset);
        }
    }

    if (nearest_offsets.size() < kMinimumLatticeNeighborPairCount) {
        return false;
    }
    std::vector<double> primary_distances;
    primary_distances.reserve(nearest_offsets.size());
    for (const auto &offset : nearest_offsets) {
        primary_distances.push_back(horizontal ? offset.x : offset.y);
    }
    const double median_spacing = median_value(std::move(primary_distances));
    if (median_spacing <= 0.0) {
        return false;
    }

    cv::Point2d offset_sum;
    std::size_t accepted_offset_count = 0;
    for (const auto &offset : nearest_offsets) {
        const double primary = horizontal ? offset.x : offset.y;
        if (std::abs(primary - median_spacing) > median_spacing * kLatticePitchDeviationFraction) {
            continue;
        }
        offset_sum.x += offset.x;
        offset_sum.y += offset.y;
        ++accepted_offset_count;
    }
    if (accepted_offset_count < kMinimumLatticeNeighborPairCount) {
        return false;
    }

    step = cv::Point2f(static_cast<float>(offset_sum.x / accepted_offset_count),
                       static_cast<float>(offset_sum.y / accepted_offset_count));
    return true;
}

// Assigns stable baseline detections to integer grid coordinates, rejects
// off-grid observations, and fits an affine lattice. Filling the integer bounds
// reconstructs internal missing sites such as the intentionally dark middle row.
inline bool infer_circle_map(const std::vector<CircleDetection> &detections, CircleMapGeometry &geometry) {
    if (detections.size() < kMinimumLatticeDetectionCount) {
        return false;
    }

    double radius_sum = 0.0;
    for (const auto &detection : detections) {
        radius_sum += detection.radius;
    }
    const float initial_mean_radius =
        static_cast<float>(radius_sum / static_cast<double>(detections.size()));

    cv::Point2f horizontal_step;
    cv::Point2f vertical_step;
    if (!estimate_lattice_step(detections, initial_mean_radius, true, horizontal_step) ||
        !estimate_lattice_step(detections, initial_mean_radius, false, vertical_step)) {
        return false;
    }

    const float determinant = horizontal_step.x * vertical_step.y -
                              horizontal_step.y * vertical_step.x;
    if (std::abs(determinant) < 1.0F) {
        return false;
    }

    const float minimum_pitch = std::min(static_cast<float>(cv::norm(horizontal_step)),
                                         static_cast<float>(cv::norm(vertical_step)));
    const float maximum_residual =
        std::max(initial_mean_radius, kLatticeFitToleranceInPitch * minimum_pitch);
    const float maximum_residual_squared = maximum_residual * maximum_residual;

    cv::Point2f anchor;
    std::size_t best_inlier_count = 0;
    double best_residual_sum = std::numeric_limits<double>::max();
    for (const auto &anchor_candidate : detections) {
        std::size_t inlier_count = 0;
        double residual_sum = 0.0;
        for (const auto &detection : detections) {
            const cv::Point2f offset = detection.center - anchor_candidate.center;
            const int column = cvRound((offset.x * vertical_step.y - offset.y * vertical_step.x) / determinant);
            const int row = cvRound((horizontal_step.x * offset.y - horizontal_step.y * offset.x) / determinant);
            const cv::Point2f reconstructed = anchor_candidate.center +
                                              static_cast<float>(column) * horizontal_step +
                                              static_cast<float>(row) * vertical_step;
            const cv::Point2f residual = detection.center - reconstructed;
            const float residual_squared = residual.dot(residual);
            if (residual_squared <= maximum_residual_squared) {
                ++inlier_count;
                residual_sum += residual_squared;
            }
        }
        if (inlier_count > best_inlier_count ||
            (inlier_count == best_inlier_count && residual_sum < best_residual_sum)) {
            anchor            = anchor_candidate.center;
            best_inlier_count = inlier_count;
            best_residual_sum = residual_sum;
        }
    }
    if (best_inlier_count < kMinimumLatticeDetectionCount) {
        return false;
    }

    std::vector<LatticeObservation> observations;
    observations.reserve(detections.size());
    for (const auto &detection : detections) {
        const cv::Point2f offset = detection.center - anchor;
        const float column_coordinate = (offset.x * vertical_step.y - offset.y * vertical_step.x) / determinant;
        const float row_coordinate    = (horizontal_step.x * offset.y - horizontal_step.y * offset.x) / determinant;
        const int column = cvRound(column_coordinate);
        const int row    = cvRound(row_coordinate);
        const cv::Point2f reconstructed = anchor +
                                          static_cast<float>(column) * horizontal_step +
                                          static_cast<float>(row) * vertical_step;
        const cv::Point2f residual = detection.center - reconstructed;
        const float residual_squared = residual.dot(residual);
        if (residual_squared > maximum_residual_squared) {
            continue;
        }

        const auto duplicate = std::find_if(observations.begin(), observations.end(), [&](const auto &existing) {
            return existing.column == column && existing.row == row;
        });
        if (duplicate == observations.end()) {
            observations.push_back(LatticeObservation{column, row, detection, residual_squared});
        } else if (residual_squared < duplicate->residual_squared) {
            *duplicate = LatticeObservation{column, row, detection, residual_squared};
        }
    }
    if (observations.size() < kMinimumLatticeDetectionCount) {
        return false;
    }

    cv::Mat design(static_cast<int>(observations.size()), 3, CV_64F);
    cv::Mat image_x(static_cast<int>(observations.size()), 1, CV_64F);
    cv::Mat image_y(static_cast<int>(observations.size()), 1, CV_64F);
    for (int index = 0; index < static_cast<int>(observations.size()); ++index) {
        const auto &observation = observations[static_cast<std::size_t>(index)];
        design.at<double>(index, 0)  = 1.0;
        design.at<double>(index, 1)  = observation.column;
        design.at<double>(index, 2)  = observation.row;
        image_x.at<double>(index, 0) = observation.detection.center.x;
        image_y.at<double>(index, 0) = observation.detection.center.y;
    }

    cv::Mat coefficients_x;
    cv::Mat coefficients_y;
    if (!cv::solve(design, image_x, coefficients_x, cv::DECOMP_SVD) ||
        !cv::solve(design, image_y, coefficients_y, cv::DECOMP_SVD)) {
        return false;
    }

    const cv::Point2f origin(static_cast<float>(coefficients_x.at<double>(0, 0)),
                             static_cast<float>(coefficients_y.at<double>(0, 0)));
    geometry.horizontal_step = cv::Point2f(static_cast<float>(coefficients_x.at<double>(1, 0)),
                                           static_cast<float>(coefficients_y.at<double>(1, 0)));
    geometry.vertical_step   = cv::Point2f(static_cast<float>(coefficients_x.at<double>(2, 0)),
                                           static_cast<float>(coefficients_y.at<double>(2, 0)));

    int minimum_column = observations.front().column;
    int maximum_column = observations.front().column;
    int minimum_row    = observations.front().row;
    int maximum_row    = observations.front().row;
    radius_sum         = 0.0;
    for (const auto &observation : observations) {
        minimum_column = std::min(minimum_column, observation.column);
        maximum_column = std::max(maximum_column, observation.column);
        minimum_row    = std::min(minimum_row, observation.row);
        maximum_row    = std::max(maximum_row, observation.row);
        radius_sum += observation.detection.radius;
    }
    geometry.column_count = maximum_column - minimum_column + 1;
    geometry.row_count    = maximum_row - minimum_row + 1;
    if (geometry.column_count < 2 || geometry.row_count < 2 ||
        geometry.column_count > kMaximumLatticeDimension ||
        geometry.row_count > kMaximumLatticeDimension) {
        return false;
    }

    geometry.mean_radius_px = static_cast<float>(radius_sum / observations.size());
    geometry.centers.clear();
    geometry.centers.reserve(static_cast<std::size_t>(geometry.column_count * geometry.row_count));
    for (int row = minimum_row; row <= maximum_row; ++row) {
        for (int column = minimum_column; column <= maximum_column; ++column) {
            geometry.centers.push_back(origin +
                                       static_cast<float>(column) * geometry.horizontal_step +
                                       static_cast<float>(row) * geometry.vertical_step);
        }
    }
    return true;
}

} // namespace e_bts

#endif // E_BTS_CIRCLE_LATTICE_H
