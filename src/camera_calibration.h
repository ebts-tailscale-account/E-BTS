#ifndef E_BTS_CAMERA_CALIBRATION_H
#define E_BTS_CAMERA_CALIBRATION_H

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <metavision/hal/device/device.h>
#include <metavision/hal/facilities/i_ll_biases.h>
#include <metavision/hal/facilities/i_roi.h>
#include <metavision/sdk/driver/camera.h>

// Applies the event-camera calibration produced by calibration/bias_tuner to a
// freshly-opened camera: the low-level biases AND the hardware ROI.
//
// WHY THIS EXISTS: the EVK1 does NOT retain biases or ROI between sessions --
// every open comes up at sensor defaults. Tuning in bias_tuner therefore has no
// effect on a later E_BTS_GUI recording unless the values are re-applied here.
//
// Two files, both written by bias_tuner's "Save biases…":
//   <name>.bias  Metavision format, "<value> % <bias_name>" per line. Also
//                loadable by Studio/HAL; the SDK writes one of these next to
//                every .raw recording, so biases are self-documenting.
//   <name>.roi   E-BTS-specific companion, "x y width height" (sensor pixels)
//                on one line, or "disabled". The .bias format has no ROI field
//                and the .raw header records only the full sensor geometry, so
//                without this file an ROI would be applied with no record of it
//                anywhere -- an undocumented spatial crop in the dataset.
//
// Default location is calibration/camera.bias relative to the working directory
// (the GUI is launched from the repo root), overridable with $E_BTS_CAMERA_BIAS.

namespace e_bts {

struct RoiRect {
    int x      = 0;
    int y      = 0;
    int width  = 0;
    int height = 0;
};

struct CameraCalibration {
    std::map<std::string, int> biases;
    std::optional<RoiRect> roi; // nullopt => leave the sensor at full frame
    std::filesystem::path bias_path;
    std::filesystem::path roi_path;
    bool bias_file_found = false;
    bool roi_file_found  = false;
};

struct AppliedCalibration {
    std::map<std::string, int> biases;  // read-back values, i.e. what actually stuck
    std::vector<std::string> clamped;   // "name (asked X, got Y)"
    std::vector<std::string> unknown;   // names the device does not expose
    std::optional<RoiRect> roi;         // the ROI actually programmed
    bool roi_applied = false;
};

// Resolution order: $E_BTS_CAMERA_BIAS, then calibration/camera.bias relative to
// the working directory, then relative to the executable's parent (so launching
// build/E_BTS_GUI from anywhere still finds calibration/camera.bias). Falls back
// to the cwd-relative path so a "not found" message names the expected location.
inline std::filesystem::path default_camera_bias_path() {
    if (const char *override_path = std::getenv("E_BTS_CAMERA_BIAS")) {
        if (*override_path != '\0') {
            return std::filesystem::path(override_path);
        }
    }

    const std::filesystem::path relative = std::filesystem::path("calibration") / "camera.bias";
    std::error_code ignored;
    if (std::filesystem::exists(relative, ignored)) {
        return relative;
    }

    // /proc/self/exe is Linux-only, which is all this rig targets.
    std::error_code link_error;
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", link_error);
    if (!link_error) {
        const std::filesystem::path beside_exe = exe.parent_path().parent_path() / relative;
        if (std::filesystem::exists(beside_exe, ignored)) {
            return beside_exe;
        }
    }

    return relative;
}

inline std::filesystem::path roi_companion_path(const std::filesystem::path &bias_path) {
    std::filesystem::path path = bias_path;
    path.replace_extension(".roi");
    return path;
}

// "<value> % <bias_name>", tolerating the variable padding different writers
// emit ("221  % bias_diff_off"). Blank lines and '#' comments are skipped.
inline bool parse_bias_line(const std::string &line, std::string &name_out, int &value_out) {
    const std::size_t separator = line.find('%');
    if (separator == std::string::npos) {
        return false;
    }
    std::istringstream value_stream(line.substr(0, separator));
    if (!(value_stream >> value_out)) {
        return false;
    }
    std::istringstream name_stream(line.substr(separator + 1));
    return static_cast<bool>(name_stream >> name_out);
}

inline CameraCalibration load_camera_calibration(const std::filesystem::path &bias_path) {
    CameraCalibration calibration;
    calibration.bias_path = bias_path;
    calibration.roi_path  = roi_companion_path(bias_path);

    std::ifstream bias_file(bias_path);
    if (bias_file) {
        calibration.bias_file_found = true;
        std::string line;
        while (std::getline(bias_file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::string name;
            int value = 0;
            if (parse_bias_line(line, name, value)) {
                calibration.biases[name] = value;
            }
        }
    }

    std::ifstream roi_file(calibration.roi_path);
    if (roi_file) {
        calibration.roi_file_found = true;
        std::string line;
        while (std::getline(roi_file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream stream(line);
            std::string first;
            stream >> first;
            if (first == "disabled" || first == "off" || first == "full") {
                break; // explicit full-frame
            }
            std::istringstream rect_stream(line);
            RoiRect rect;
            if (rect_stream >> rect.x >> rect.y >> rect.width >> rect.height) {
                calibration.roi = rect;
            }
            break; // one geometry line is all we support
        }
    }

    return calibration;
}

// One line per '<x> <y> <width> <height>', matching what load_camera_calibration
// reads back. Written next to each .raw so a recording's crop is recoverable.
inline std::string roi_sidecar_contents(const std::optional<RoiRect> &roi, int width, int height) {
    std::ostringstream out;
    out << "# E-BTS hardware ROI (I_ROI) - x y width height, in sensor pixels.\n";
    out << "# Sensor geometry " << width << "x" << height
        << "; event coordinates in the .raw stay absolute (NOT re-based to the ROI origin).\n";
    if (roi) {
        out << roi->x << ' ' << roi->y << ' ' << roi->width << ' ' << roi->height << '\n';
    } else {
        out << "disabled\n";
    }
    return out.str();
}

inline bool write_roi_sidecar(const std::filesystem::path &path, const std::optional<RoiRect> &roi, int width,
                              int height) {
    std::ofstream out(path);
    if (!out) {
        return false;
    }
    out << roi_sidecar_contents(roi, width, height);
    return static_cast<bool>(out);
}

// Programs biases then ROI on an OPEN but NOT-yet-started camera. Never throws:
// a missing facility or a rejected value is reported through the log callback
// and the returned struct, because losing the calibration must not cost you the
// camera session.
inline AppliedCalibration apply_camera_calibration(Metavision::Camera &camera,
                                                   const CameraCalibration &calibration,
                                                   const std::function<void(const std::string &)> &log = nullptr) {
    AppliedCalibration applied;
    const auto report = [&log](const std::string &message) {
        if (log) {
            log(message);
        }
    };

    const int width  = camera.geometry().width();
    const int height = camera.geometry().height();

    // ---- biases ----
    if (!calibration.biases.empty()) {
        Metavision::I_LL_Biases *biases = nullptr;
        try {
            biases = camera.get_device().get_facility<Metavision::I_LL_Biases>();
        } catch (const std::exception &error) {
            report(std::string("no I_LL_Biases facility (") + error.what() + "); biases NOT applied");
        }
        if (biases != nullptr) {
            const std::map<std::string, int> available = biases->get_all_biases();
            for (const auto &[name, value] : calibration.biases) {
                if (available.find(name) == available.end()) {
                    applied.unknown.push_back(name);
                    continue;
                }
                try {
                    biases->set(name, value);
                    const int read_back  = biases->get(name);
                    applied.biases[name] = read_back;
                    if (read_back != value) {
                        applied.clamped.push_back(name + " (asked " + std::to_string(value) + ", got " +
                                                  std::to_string(read_back) + ")");
                    }
                } catch (const std::exception &error) {
                    applied.clamped.push_back(name + " (rejected: " + error.what() + ")");
                }
            }
            std::ostringstream summary;
            summary << "biases applied: " << applied.biases.size() << '/' << calibration.biases.size() << " from "
                    << calibration.bias_path.string();
            report(summary.str());
            for (const auto &[name, value] : applied.biases) {
                report("  " + name + " = " + std::to_string(value));
            }
            if (!applied.clamped.empty()) {
                std::ostringstream clamped;
                clamped << "  clamped/rejected:";
                for (const std::string &entry : applied.clamped) {
                    clamped << ' ' << entry;
                }
                report(clamped.str());
            }
            if (!applied.unknown.empty()) {
                std::ostringstream unknown;
                unknown << "  not present on this device:";
                for (const std::string &entry : applied.unknown) {
                    unknown << ' ' << entry;
                }
                report(unknown.str());
            }
        }
    }

    // ---- hardware ROI ----
    if (!calibration.roi) {
        report("ROI: full frame (no geometry in " + calibration.roi_path.string() + ")");
        return applied;
    }

    RoiRect rect = *calibration.roi;
    const RoiRect requested = rect;
    rect.x      = std::max(0, std::min(rect.x, width - 1));
    rect.y      = std::max(0, std::min(rect.y, height - 1));
    rect.width  = std::max(1, std::min(rect.width, width - rect.x));
    rect.height = std::max(1, std::min(rect.height, height - rect.y));
    if (rect.x != requested.x || rect.y != requested.y || rect.width != requested.width ||
        rect.height != requested.height) {
        std::ostringstream clamped;
        clamped << "ROI clamped to the sensor: asked " << requested.x << ' ' << requested.y << ' '
                << requested.width << ' ' << requested.height << ", using " << rect.x << ' ' << rect.y << ' '
                << rect.width << ' ' << rect.height;
        report(clamped.str());
    }

    Metavision::I_ROI *roi_facility = nullptr;
    try {
        roi_facility = camera.get_device().get_facility<Metavision::I_ROI>();
    } catch (const std::exception &error) {
        report(std::string("no I_ROI facility (") + error.what() + "); ROI NOT applied");
        return applied;
    }
    if (roi_facility == nullptr) {
        report("no I_ROI facility; ROI NOT applied");
        return applied;
    }

    // A row+column binary mask IS a rectangle on this sensor ("a pixel is
    // enabled only if both its row and column are enabled"). This overload is
    // pure-virtual in HAL 3.1.2, so it resolves through the vtable and needs no
    // direct link against the HAL library, unlike set_ROI(DeviceRoi).
    std::vector<bool> cols(static_cast<std::size_t>(width), false);
    std::vector<bool> rows(static_cast<std::size_t>(height), false);
    for (int col = rect.x; col < rect.x + rect.width; ++col) {
        cols[static_cast<std::size_t>(col)] = true;
    }
    for (int row = rect.y; row < rect.y + rect.height; ++row) {
        rows[static_cast<std::size_t>(row)] = true;
    }

    try {
        // enable is passed in the same call as the geometry: I_ROI::enable()
        // warns that an ROI must already be set before enabling.
        if (!roi_facility->set_ROIs(cols, rows, true)) {
            report("ROI rejected by the sensor; continuing at full frame");
            return applied;
        }
    } catch (const std::exception &error) {
        report(std::string("ROI failed (") + error.what() + "); continuing at full frame");
        return applied;
    }

    applied.roi         = rect;
    applied.roi_applied = true;
    std::ostringstream summary;
    summary << "ROI applied: x=" << rect.x << " y=" << rect.y << " w=" << rect.width << " h=" << rect.height
            << " (" << (rect.width * rect.height) << " px, "
            << (100.0 * rect.width * rect.height / (width * height)) << "% of frame) from "
            << calibration.roi_path.string();
    report(summary.str());
    report("  events outside the ROI are never read out; .raw coordinates stay absolute");
    return applied;
}

} // namespace e_bts

#endif // E_BTS_CAMERA_CALIBRATION_H
