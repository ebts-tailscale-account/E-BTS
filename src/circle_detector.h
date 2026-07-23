#ifndef E_BTS_CIRCLE_DETECTOR_H
#define E_BTS_CIRCLE_DETECTOR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "circle_tracker_config.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace e_bts {

struct CircleDetection {
    cv::Point2f center;
    float radius   = 0.0F;
    double density = 0.0;
};

struct CircleDetectionDiagnostics {
    std::size_t contour_count             = 0;
    std::size_t short_contour_count       = 0;
    std::size_t small_radius_count        = 0;
    std::size_t large_radius_count        = 0;
    std::size_t low_contour_fill_count    = 0;
    std::size_t low_density_count         = 0;
    std::size_t duplicate_detection_count = 0;
    std::size_t map_site_count             = 0;
    bool used_circle_map                   = false;
};

enum class CircleRejectionReason {
    SmallRadius,
    LargeRadius,
    LowContourFill,
    LowDensity
};

struct RejectedCircle {
    cv::Point2f center;
    float radius = 0.0F;
    CircleRejectionReason reason = CircleRejectionReason::LowDensity;
};

struct CircleDetectionResult {
    std::vector<CircleDetection> detections;
    std::vector<RejectedCircle> rejected_circles;
    CircleDetectionDiagnostics diagnostics;
    cv::Mat candidate_mask;
};

struct CircleMapGeometry {
    std::vector<cv::Point2f> centers;
    cv::Point2f horizontal_step;
    cv::Point2f vertical_step;
    float mean_radius_px   = 0.0F;
    int column_count       = 0;
    int row_count          = 0;
};

inline double occupied_fraction_in_circle(const cv::Mat &occupied_pixels, const cv::Point2f &center, float radius) {
    const int minimum_x = std::max(0, static_cast<int>(std::floor(center.x - radius)));
    const int maximum_x = std::min(occupied_pixels.cols - 1, static_cast<int>(std::ceil(center.x + radius)));
    const int minimum_y = std::max(0, static_cast<int>(std::floor(center.y - radius)));
    const int maximum_y = std::min(occupied_pixels.rows - 1, static_cast<int>(std::ceil(center.y + radius)));
    const float radius_squared = radius * radius;

    std::uint32_t pixels_in_circle = 0;
    std::uint32_t occupied_count   = 0;
    for (int y = minimum_y; y <= maximum_y; ++y) {
        const auto *row = occupied_pixels.ptr<std::uint8_t>(y);
        for (int x = minimum_x; x <= maximum_x; ++x) {
            const float delta_x = static_cast<float>(x) - center.x;
            const float delta_y = static_cast<float>(y) - center.y;
            if (delta_x * delta_x + delta_y * delta_y > radius_squared) {
                continue;
            }

            ++pixels_in_circle;
            occupied_count += row[x] != 0;
        }
    }

    if (pixels_in_circle == 0) {
        return 0.0;
    }
    return static_cast<double>(occupied_count) / static_cast<double>(pixels_in_circle);
}

class CircleDetector {
public:
    CircleDetector(int width, int height, double minimum_circle_density) :
        minimum_circle_density_(minimum_circle_density) {
        const double horizontal_diameter_px = kMarkerDiameterMm * static_cast<double>(width) / kSensorWidthMm;
        const double vertical_diameter_px   = kMarkerDiameterMm * static_cast<double>(height) / kSensorHeightMm;
        expected_radius_px_ = (horizontal_diameter_px + vertical_diameter_px) / 4.0;
        minimum_radius_px_  = expected_radius_px_ * kMinimumRadiusFactor;
        maximum_radius_px_  = expected_radius_px_ * kMaximumRadiusFactor;
        morphology_kernel_  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    }

    CircleDetectionResult detect(const cv::Mat &occupied_pixels) const {
        // Remove isolated activity and thin noise bridges before closing small
        // holes in a marker. Density is still measured from the unmodified map.
        CircleDetectionResult result;
        result.candidate_mask = make_candidate_mask(occupied_pixels);

        std::vector<std::vector<cv::Point>> contours;
        cv::Mat contour_mask = result.candidate_mask.clone();
        cv::findContours(contour_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        result.diagnostics.contour_count = contours.size();

        std::vector<CircleDetection> detections;
        detections.reserve(contours.size());
        for (const auto &contour : contours) {
            if (contour.size() < 3) {
                ++result.diagnostics.short_contour_count;
                continue;
            }

            cv::Point2f center;
            float radius = 0.0F;
            cv::minEnclosingCircle(contour, center, radius);
            if (radius < minimum_radius_px_) {
                ++result.diagnostics.small_radius_count;
                continue;
            }
            if (radius > maximum_radius_px_) {
                ++result.diagnostics.large_radius_count;
                record_rejected_circle(result, center, radius, CircleRejectionReason::LargeRadius);
                continue;
            }

            const double enclosing_area = CV_PI * static_cast<double>(radius) * static_cast<double>(radius);
            const double contour_fill   = std::abs(cv::contourArea(contour)) / enclosing_area;
            if (contour_fill < 0.50) {
                ++result.diagnostics.low_contour_fill_count;
                record_rejected_circle(result, center, radius, CircleRejectionReason::LowContourFill);
                continue;
            }

            const double density = occupied_fraction_in_circle(occupied_pixels, center, radius);
            if (density >= minimum_circle_density_) {
                detections.push_back(CircleDetection{center, radius, density});
            } else {
                ++result.diagnostics.low_density_count;
                record_rejected_circle(result, center, radius, CircleRejectionReason::LowDensity);
            }
        }

        // Separate event islands from one dome can each pass the contour tests.
        // Keep only the most radius-consistent detection within one expected
        // radius so they cannot become separate tracks at neighboring positions.
        std::sort(detections.begin(), detections.end(), [this](const auto &left, const auto &right) {
            const double left_radius_error  = std::abs(left.radius - expected_radius_px_);
            const double right_radius_error = std::abs(right.radius - expected_radius_px_);
            if (std::abs(left_radius_error - right_radius_error) > 0.1) {
                return left_radius_error < right_radius_error;
            }
            return left.density > right.density;
        });

        std::vector<CircleDetection> unique_detections;
        unique_detections.reserve(detections.size());
        const float duplicate_distance_px =
            static_cast<float>(expected_radius_px_ * kDuplicateCenterDistanceInExpectedRadii);
        const float duplicate_distance_squared = duplicate_distance_px * duplicate_distance_px;
        for (const auto &detection : detections) {
            const bool overlaps_existing =
                std::any_of(unique_detections.begin(), unique_detections.end(), [&](const auto &existing) {
                    const cv::Point2f difference = detection.center - existing.center;
                    return difference.dot(difference) <= duplicate_distance_squared;
                });
            if (!overlaps_existing) {
                unique_detections.push_back(detection);
            } else {
                ++result.diagnostics.duplicate_detection_count;
            }
        }

        std::sort(unique_detections.begin(), unique_detections.end(), [](const auto &left, const auto &right) {
            if (std::abs(left.center.y - right.center.y) > 1.0F) {
                return left.center.y < right.center.y;
            }
            return left.center.x < right.center.x;
        });
        result.detections = std::move(unique_detections);
        return result;
    }

    // Once a baseline lattice exists, contour shape is no longer used as a
    // gate. A small search around each expected marker position finds the
    // locally densest center, then the original circular density test decides
    // whether that marker is active in this event window.
    CircleDetectionResult detect_from_map(const cv::Mat &occupied_pixels,
                                          const std::vector<cv::Point2f> &expected_centers,
                                          float radius, int search_radius) const {
        CircleDetectionResult result;
        result.candidate_mask                  = make_candidate_mask(occupied_pixels);
        result.diagnostics.used_circle_map     = true;
        result.diagnostics.map_site_count      = expected_centers.size();

        cv::Mat integral_image;
        cv::integral(occupied_pixels, integral_image, CV_32S);
        const int score_radius = std::max(1, cvRound(radius * kMapCenterScoreRadiusFactor));

        result.detections.reserve(expected_centers.size());
        for (const cv::Point2f &map_center : expected_centers) {
            const cv::Point expected_center = rounded_image_point(map_center, occupied_pixels.size());
            cv::Point best_center           = expected_center;
            int best_score                  = -1;
            int best_distance_squared       = std::numeric_limits<int>::max();

            for (int offset_y = -search_radius; offset_y <= search_radius; ++offset_y) {
                for (int offset_x = -search_radius; offset_x <= search_radius; ++offset_x) {
                    const cv::Point candidate(expected_center.x + offset_x, expected_center.y + offset_y);
                    if (candidate.x < 0 || candidate.x >= occupied_pixels.cols ||
                        candidate.y < 0 || candidate.y >= occupied_pixels.rows) {
                        continue;
                    }

                    const int score = occupied_sum_in_square(integral_image, candidate, score_radius);
                    const int distance_squared = offset_x * offset_x + offset_y * offset_y;
                    if (score > best_score ||
                        (score == best_score && distance_squared < best_distance_squared)) {
                        best_score            = score;
                        best_distance_squared = distance_squared;
                        best_center           = candidate;
                    }
                }
            }

            const cv::Point2f center(static_cast<float>(best_center.x), static_cast<float>(best_center.y));
            const double density = occupied_fraction_in_circle(occupied_pixels, center, radius);
            if (density >= minimum_circle_density_) {
                result.detections.push_back(CircleDetection{center, radius, density});
            } else {
                ++result.diagnostics.low_density_count;
            }
        }
        return result;
    }

    double expected_radius_px() const {
        return expected_radius_px_;
    }

    double minimum_radius_px() const {
        return minimum_radius_px_;
    }

    double maximum_radius_px() const {
        return maximum_radius_px_;
    }

    double minimum_circle_density() const {
        return minimum_circle_density_;
    }

    // Runtime tuning hook for the settings overlay. No locking: detect()/
    // detect_from_map() and this setter both run on the main thread only.
    void set_minimum_circle_density(double minimum_circle_density) {
        minimum_circle_density_ =
            std::clamp(minimum_circle_density, kMinimumDetectionDensity, kMaximumDetectionDensity);
    }

private:
    cv::Mat make_candidate_mask(const cv::Mat &occupied_pixels) const {
        cv::Mat candidate_mask;
        cv::morphologyEx(occupied_pixels, candidate_mask, cv::MORPH_OPEN,
                         morphology_kernel_, cv::Point(-1, -1), 1);
        cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_CLOSE,
                         morphology_kernel_, cv::Point(-1, -1), 1);
        return candidate_mask;
    }

    static cv::Point rounded_image_point(const cv::Point2f &point, const cv::Size &image_size) {
        return cv::Point(std::clamp(cvRound(point.x), 0, image_size.width - 1),
                         std::clamp(cvRound(point.y), 0, image_size.height - 1));
    }

    static int occupied_sum_in_square(const cv::Mat &integral_image, const cv::Point &center,
                                      int radius) {
        const int minimum_x = std::max(0, center.x - radius);
        const int maximum_x = std::min(integral_image.cols - 2, center.x + radius);
        const int minimum_y = std::max(0, center.y - radius);
        const int maximum_y = std::min(integral_image.rows - 2, center.y + radius);
        const auto *top     = integral_image.ptr<int>(minimum_y);
        const auto *bottom  = integral_image.ptr<int>(maximum_y + 1);
        return bottom[maximum_x + 1] - bottom[minimum_x] -
               top[maximum_x + 1] + top[minimum_x];
    }

    static void record_rejected_circle(CircleDetectionResult &result, const cv::Point2f &center,
                                       float radius, CircleRejectionReason reason) {
        if (result.rejected_circles.size() < kMaximumRenderedRejectedCircles) {
            result.rejected_circles.push_back(RejectedCircle{center, radius, reason});
        }
    }

    double minimum_circle_density_ = 0.0;
    double expected_radius_px_ = 0.0;
    double minimum_radius_px_  = 0.0;
    double maximum_radius_px_  = 0.0;
    cv::Mat morphology_kernel_;
};

} // namespace e_bts

#endif // E_BTS_CIRCLE_DETECTOR_H
