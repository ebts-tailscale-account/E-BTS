#ifndef E_BTS_CIRCLE_TRACKER_CONFIG_H
#define E_BTS_CIRCLE_TRACKER_CONFIG_H

#include <cstddef>
#include <cstdint>

#include <metavision/sdk/base/utils/timestamp.h>

// Tuning and physical constants for the marker circle tracker
// (event_circle_tracker.cpp and its supporting headers). Grouped by concern so
// a value's role is clear without re-deriving it from call sites.

namespace e_bts {

// ---- Interactive-prompt defaults ----
// In combined mode (E_BTS_combined), this one collection time seeds both the
// tracker's event-window buffer and the live view's display accumulation; see
// the "Shared live-tunable ranges" section below for the range it can be
// adjusted within afterward via the on-screen settings overlay.
constexpr Metavision::timestamp kDefaultEventCollectionTimeUs = 10'000;
constexpr double kDefaultMinimumCircleDensity                 = 0.50;
constexpr std::uint16_t kLiveDisplayFps                       = 60;

// ---- Physical marker/sensor geometry ----
// These are construction values, not a camera calibration. They give an initial
// marker diameter of 14.2 px horizontally and 12.8 px vertically at 640x480.
constexpr double kSensorWidthMm       = 36.0;
constexpr double kSensorHeightMm      = 30.0;
constexpr double kMarkerDiameterMm    = 0.8;
constexpr double kMinimumRadiusFactor = 0.55;
constexpr double kMaximumRadiusFactor = 2.10;

// ---- Event window buffering ----
constexpr std::size_t kMaximumQueuedWindows = 8;

// ---- Detection / contour-based circle finding ----
constexpr double kDuplicateCenterDistanceInExpectedRadii = 1.0;
constexpr std::size_t kMaximumRenderedRejectedCircles    = 256;

// ---- Temporal tracking ----
constexpr double kMaximumTrackStepInRadii = 2.75;
constexpr int kTemporalFilterWindowCount  = 10;

// ---- Baseline collection ----
constexpr Metavision::timestamp kBaselineCollectionDurationUs = 500'000;
constexpr std::size_t kMinimumBaselineCollectionWindows       = 20;
constexpr std::size_t kMinimumBaselineObservations            = 4;
constexpr double kMinimumBaselineObservationFraction          = 0.25;
constexpr double kBaselineClusterDistanceInExpectedRadii      = 1.5;

// ---- Lattice inference (baseline marker grid fitting) ----
constexpr std::size_t kMinimumLatticeDetectionCount    = 8;
constexpr std::size_t kMinimumLatticeNeighborPairCount = 4;
constexpr float kLatticeMinimumNeighborDistanceInRadii = 1.5F;
constexpr float kLatticeAlignmentToleranceInRadii      = 0.8F;
constexpr double kLatticePitchDeviationFraction        = 0.25;
constexpr float kLatticeFitToleranceInPitch            = 0.30F;
constexpr int kMaximumLatticeDimension                 = 64;

// ---- Circle-map-guided detection (post-baseline tracking) ----
constexpr float kMapSearchRadiusInPitch         = 0.30F;
constexpr float kMapCenterScoreRadiusFactor     = 0.60F;
constexpr float kInitialMapMatchDistanceInRadii = 1.25F;

// ---- Rendering ----
constexpr float kMinimumDisplacementVectorLengthPx = 1.0F;

// ---- Shared live-tunable ranges (E_BTS_GUI's Accumulation Time / Detection
// Percent dialogs) ----
// Bounds for the runtime setters on EventWindowBuffer/CDFrameGenerator
// (accumulation time) and CircleDetector (minimum circle density), used to
// clamp whatever value the GUI (or the standalone tracker CLI's startup
// prompt) requests. Independent of the much larger overflow guard applied
// to the startup prompt in circle_tracker_cli_main.cpp; these are just a
// sane interactive range.
constexpr Metavision::timestamp kMinimumCollectionTimeUs = 1'000;
constexpr Metavision::timestamp kMaximumCollectionTimeUs = 100'000;
constexpr double kMinimumDetectionDensity                = 0.01;
constexpr double kMaximumDetectionDensity                = 1.00;

} // namespace e_bts

#endif // E_BTS_CIRCLE_TRACKER_CONFIG_H
