#ifndef E_BTS_SPATTER_TRACKER_CONFIG_H
#define E_BTS_SPATTER_TRACKER_CONFIG_H

#include <cstddef>

// Tuning defaults and live-adjustable ranges for the spatter (blob) tracker in
// spatter_tracker.h. Separate from circle_tracker_config.h on purpose: the
// circle tracker's constants are tied to the marker lattice's physical geometry,
// whereas nothing here assumes what is being tracked -- the spatter tracker is
// the general-purpose "moving blobs" front-end and every value below is meant to
// be dialled in from the GUI against whatever the camera is pointed at.

namespace e_bts {

// ---- Defaults ----
// These mirror the Metavision Analytics spatter-tracking sample's documented
// defaults (cells of 7x7 px, 5 events to activate a cell, objects between 10x10
// and 300x300 px, 50 px association radius, 5 missed windows before a track is
// dropped). They are a starting point for tuning, not a calibration.
constexpr int kDefaultSpatterCellWidth          = 7;
constexpr int kDefaultSpatterCellHeight         = 7;
constexpr int kDefaultSpatterActivationThreshold = 5;
constexpr int kDefaultSpatterMinSizePx          = 10;
constexpr int kDefaultSpatterMaxSizePx          = 300;
constexpr int kDefaultSpatterMaxDistancePx      = 50;
constexpr int kDefaultSpatterUntrackedThreshold = 5;

// ---- Region of interest ----
// Events outside this rectangle are ignored before the grid is accumulated, so
// nothing outside it can activate a cell, join a cluster or start a track.
// Unlike the hardware ROI in calibration/camera.roi this is a purely software
// crop applied to one consumer: the live view and Circle Tracking still see the
// whole sensor. Coordinates are therefore full-sensor coordinates, the same
// frame the hardware ROI's events arrive in.
//
// The default is the marker band measured for this rig. Note it is intentionally
// wider than the sensor: x 6 + width 640 runs to 646 on a 640 px sensor, so the
// tracker clips it to the 634 px that exist. Clipping, rather than rejecting,
// keeps "the full width from x=6 rightwards" expressible without knowing the
// sensor size.
constexpr int kDefaultSpatterRoiX      = 6;
constexpr int kDefaultSpatterRoiY      = 181;
constexpr int kDefaultSpatterRoiWidth  = 640;
constexpr int kDefaultSpatterRoiHeight = 64;

// ---- Live-tunable ranges (E_BTS_GUI's "Spatter Params" dialog) ----
// The setters clamp to these, so a nonsensical value typed into the dialog
// degrades the tracking rather than dividing by zero or allocating a grid with
// one cell per pixel.
constexpr int kMinimumSpatterCellSize           = 1;
constexpr int kMaximumSpatterCellSize           = 128;
constexpr int kMinimumSpatterActivationThreshold = 1;
constexpr int kMaximumSpatterActivationThreshold = 4096;
constexpr int kMinimumSpatterSizePx             = 1;
constexpr int kMaximumSpatterSizePx             = 4096;
constexpr int kMinimumSpatterMaxDistancePx      = 1;
constexpr int kMaximumSpatterMaxDistancePx      = 4096;
constexpr int kMinimumSpatterUntrackedThreshold = 0;
constexpr int kMaximumSpatterUntrackedThreshold = 256;
// The ROI is clipped to the sensor rather than clamped to a fixed range, since
// only the tracker knows the sensor size. This is just the floor that stops a
// zero- or negative-sized rectangle from disabling tracking with no visible
// cause; an origin past the sensor edge is pulled back to fit it.
constexpr int kMinimumSpatterRoiSizePx = 1;

// ---- Rendering ----
// Trail length drawn behind each track, in past centroids. Purely cosmetic; it
// makes direction of travel readable in a still screenshot.
constexpr std::size_t kSpatterTrailLength = 16;

} // namespace e_bts

#endif // E_BTS_SPATTER_TRACKER_CONFIG_H
