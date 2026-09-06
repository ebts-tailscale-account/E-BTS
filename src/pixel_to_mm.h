#ifndef E_BTS_PIXEL_TO_MM_H
#define E_BTS_PIXEL_TO_MM_H

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <QByteArray>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <opencv2/core.hpp>

// Sensor pixel -> millimetres on the marker plane, live.
//
// THE C++ SIDE OF ml/undistort.py, AND IT MUST STAY THAT.
// -------------------------------------------------------
// ml/undistort.py is documented as "the ONE place that turns sensor pixels into
// millimetres", and everything measured offline goes through it. This file is a
// second implementation of the same evaluation, which is exactly the situation
// that file warns about -- so it reads the SAME calibration/pixel_to_mm.json and
// evaluates the SAME model, and ml/check_pixel_to_mm_port.py exists to prove on
// real numbers that the two agree. If they ever disagree, the offline one is
// right by definition and this one is the bug.
//
// Only the `lattice_affine` model is implemented, because that is what the
// current calibration (fitted 2026-09-03 from franka/calib_raster.py) is. A file
// carrying any other model is REFUSED rather than approximated: an affine
// evaluated where a lattice_affine was meant is a plausible wrong number, and
// those do not get noticed.
//
// WHAT THE OUTPUT MEANS
// ---------------------
// pixel_to_mm() returns millimetres in the ROBOT BASE frame -- the frame the
// calibration raster measured, and therefore the frame the Franka reports its own
// pose in, which is the whole point: the comparison is a subtraction, not a
// transform. pixel_to_pad_mm() then subtracts the taught elastomer centre from
// calibration/elastomer_origin.json, so (0, 0) is the middle of the pad with the
// axes still parallel to the robot's.

namespace e_bts {

inline constexpr int kLatticeInvertIterations = 4;      // ml/undistort.py _INVERT_ITERS
inline constexpr double kLatticeEdgeMarginCells = 0.75; // _EDGE_MARGIN_CELLS
inline constexpr double kLatticeInvertTolerancePx = 0.5; // _INVERT_TOL_PX

struct PixelToMm {
    bool loaded = false;
    QString source_path;
    QString error;                 // why it did not load, for the console + the GUI

    int rows = 0;                  // measured dome lattice
    int cols = 0;
    std::vector<cv::Point2d> node_px;   // row-major rows*cols; NaN where unmeasured
    double origin_x = 0.0, origin_y = 0.0, pitch_x = 0.0, pitch_y = 0.0;  // seed lattice
    cv::Point2d norm_centre{0.0, 0.0};
    double norm_scale = 1.0;
    double coeffs[3][2] = {{0, 0}, {0, 0}, {0, 0}};     // [1, col_n, row_n] -> (x_mm, y_mm)

    bool has_origin = false;       // calibration/elastomer_origin.json
    cv::Point2d origin_mm{0.0, 0.0};
    QString origin_source_path;
    QString origin_error;

    const cv::Point2d &node(int r, int c) const {
        return node_px[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                       static_cast<std::size_t>(c)];
    }

    // Continuous (row, col) index -> pixel, bilinear over the MEASURED nodes.
    cv::Point2d lattice_forward(double row, double col) const {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const int r0 = static_cast<int>(std::min(std::max(std::floor(row), 0.0),
                                                 static_cast<double>(rows - 2)));
        const int c0 = static_cast<int>(std::min(std::max(std::floor(col), 0.0),
                                                 static_cast<double>(cols - 2)));
        const double fr = row - r0;
        const double fc = col - c0;
        const cv::Point2d q00 = node(r0, c0);
        const cv::Point2d q01 = node(r0, c0 + 1);
        const cv::Point2d q10 = node(r0 + 1, c0);
        const cv::Point2d q11 = node(r0 + 1, c0 + 1);
        if (!std::isfinite(q00.x) || !std::isfinite(q01.x) || !std::isfinite(q10.x) ||
            !std::isfinite(q11.x)) {
            return cv::Point2d(nan, nan);
        }
        const double w00 = (1.0 - fr) * (1.0 - fc);
        const double w01 = (1.0 - fr) * fc;
        const double w10 = fr * (1.0 - fc);
        const double w11 = fr * fc;
        return cv::Point2d(w00 * q00.x + w01 * q01.x + w10 * q10.x + w11 * q11.x,
                           w00 * q00.y + w01 * q01.y + w10 * q10.y + w11 * q11.y);
    }

    // Pixel -> continuous (row, col). Fixed point from the ideal lattice, corrected
    // by the local Jacobian of the measured grid.
    //
    // RETURNS NaN OUTSIDE THE CALIBRATED FIELD, and that is the point: past the
    // outermost dome there is no measurement. ml/undistort.py records that an
    // earlier version clipped to the lattice instead and SATURATED -- an
    // out-of-field pixel silently came back as the edge dome, at 464 px/mm rather
    // than ~16, and nothing said so.
    bool lattice_inverse(const cv::Point2d &uv, double &row, double &col) const {
        row = (uv.y - origin_y) / pitch_y;
        col = (uv.x - origin_x) / pitch_x;
        for (int iteration = 0; iteration < kLatticeInvertIterations; ++iteration) {
            const cv::Point2d projected = lattice_forward(row, col);
            if (!std::isfinite(projected.x)) {
                // The iterate landed on a quad with an unmeasured corner, so there
                // is nothing here to correct against. ml/undistort.py reaches the
                // same place as a NaN that then fails its reprojection check; this
                // says so one step earlier rather than iterating on NaN.
                row = col = std::numeric_limits<double>::quiet_NaN();
                return false;
            }
            const cv::Point2d error(uv.x - projected.x, uv.y - projected.y);
            // d(pixel)/d(index) from a half-cell step on the measured grid
            const cv::Point2d jr = lattice_forward(row + 0.5, col) - lattice_forward(row - 0.5, col);
            const cv::Point2d jc = lattice_forward(row, col + 0.5) - lattice_forward(row, col - 0.5);
            const double determinant = jc.x * jr.y - jr.x * jc.y;
            if (std::isfinite(determinant) && std::abs(determinant) >= 1e-9) {
                const double d_col = (error.x * jr.y - jr.x * error.y) / determinant;
                const double d_row = (jc.x * error.y - error.x * jc.y) / determinant;
                row += d_row;
                col += d_col;
            }
            row = std::min(std::max(row, -kLatticeEdgeMarginCells), rows - 1 + kLatticeEdgeMarginCells);
            col = std::min(std::max(col, -kLatticeEdgeMarginCells), cols - 1 + kLatticeEdgeMarginCells);
        }
        // Convergence is checked by reprojection: a point that cannot be
        // reprojected to within the tolerance was never really solved.
        const cv::Point2d check = lattice_forward(row, col);
        if (!std::isfinite(check.x) || cv::norm(check - uv) > kLatticeInvertTolerancePx) {
            row = col = std::numeric_limits<double>::quiet_NaN();
            return false;
        }
        return true;
    }

    // Sensor pixel -> millimetres in the ROBOT BASE frame.
    bool pixel_to_mm(const cv::Point2d &uv, cv::Point2d &mm) const {
        if (!loaded) {
            return false;
        }
        double row = 0.0;
        double col = 0.0;
        if (!lattice_inverse(uv, row, col)) {
            return false;
        }
        // design((col, row), norm, "affine") @ coeffs -- the term order is part of
        // the file format (ml/undistort.py::design).
        const double u = (col - norm_centre.x) / norm_scale;
        const double v = (row - norm_centre.y) / norm_scale;
        mm.x = coeffs[0][0] + u * coeffs[1][0] + v * coeffs[2][0];
        mm.y = coeffs[0][1] + u * coeffs[1][1] + v * coeffs[2][1];
        return true;
    }

    // ...and the same thing with the elastomer centre as (0, 0).
    bool pixel_to_pad_mm(const cv::Point2d &uv, cv::Point2d &mm) const {
        cv::Point2d base;
        if (!pixel_to_mm(uv, base)) {
            return false;
        }
        mm = has_origin ? cv::Point2d(base.x - origin_mm.x, base.y - origin_mm.y) : base;
        return true;
    }
};

inline QString default_pixel_to_mm_path() {
    const QByteArray override_path = qgetenv("E_BTS_PIXEL_TO_MM");
    return override_path.isEmpty() ? QStringLiteral("calibration/pixel_to_mm.json")
                                   : QString::fromLocal8Bit(override_path);
}

inline QString default_elastomer_origin_path() {
    const QByteArray override_path = qgetenv("E_BTS_ELASTOMER_ORIGIN");
    return override_path.isEmpty() ? QStringLiteral("calibration/elastomer_origin.json")
                                   : QString::fromLocal8Bit(override_path);
}

inline bool read_json_object(const QString &path, QJsonObject &object, QString &error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        error = QStringLiteral("cannot open %1").arg(path);
        return false;
    }
    QJsonParseError parse_error{};
    const QJsonDocument document = QJsonDocument::fromJson(file.readAll(), &parse_error);
    if (document.isNull() || !document.isObject()) {
        error = QStringLiteral("%1 is not valid JSON (%2)").arg(path, parse_error.errorString());
        return false;
    }
    object = document.object();
    return true;
}

// Loads the elastomer centre. Absent is not an error -- the readout then reports
// robot-base millimetres and says so, which is honest; silently reporting
// pad-relative numbers against an origin of (0, 0) would not be.
inline void load_elastomer_origin(PixelToMm &calibration, const QString &path) {
    calibration.origin_source_path = path;
    QJsonObject root;
    if (!read_json_object(path, root, calibration.origin_error)) {
        return;
    }
    const QJsonArray origin = root.value(QStringLiteral("origin_mm")).toArray();
    if (origin.size() != 2) {
        calibration.origin_error = QStringLiteral("%1 has no 2-element origin_mm").arg(path);
        return;
    }
    calibration.origin_mm  = cv::Point2d(origin.at(0).toDouble(), origin.at(1).toDouble());
    calibration.has_origin = true;
    calibration.origin_error.clear();
}

inline PixelToMm load_pixel_to_mm(const QString &path, const QString &origin_path) {
    PixelToMm calibration;
    calibration.source_path = path;
    QJsonObject root;
    if (!read_json_object(path, root, calibration.error)) {
        return calibration;
    }

    const QJsonObject fit = root.value(QStringLiteral("pixel_to_mm")).toObject();
    const QString model   = fit.value(QStringLiteral("model")).toString();
    if (model != QStringLiteral("lattice_affine")) {
        calibration.error =
            QStringLiteral("%1 is a '%2' fit; only 'lattice_affine' is implemented live. Refusing to "
                           "approximate it -- re-fit with ml/fit_pixel_mm_warp.py or extend "
                           "src/pixel_to_mm.h.")
                .arg(path, model.isEmpty() ? QStringLiteral("(none)") : model);
        return calibration;
    }

    const QJsonObject lattice = fit.value(QStringLiteral("lattice")).toObject();
    calibration.rows     = lattice.value(QStringLiteral("rows")).toInt();
    calibration.cols     = lattice.value(QStringLiteral("cols")).toInt();
    calibration.origin_x = lattice.value(QStringLiteral("origin_x")).toDouble();
    calibration.origin_y = lattice.value(QStringLiteral("origin_y")).toDouble();
    calibration.pitch_x  = lattice.value(QStringLiteral("pitch_x")).toDouble();
    calibration.pitch_y  = lattice.value(QStringLiteral("pitch_y")).toDouble();
    if (calibration.rows < 2 || calibration.cols < 2 || calibration.pitch_x == 0.0 ||
        calibration.pitch_y == 0.0) {
        calibration.error = QStringLiteral("%1 has an unusable 'lattice' block").arg(path);
        return calibration;
    }

    const QJsonArray node_rows = fit.value(QStringLiteral("node_px")).toArray();
    if (node_rows.size() != calibration.rows) {
        calibration.error = QStringLiteral("%1: node_px has %2 rows, lattice says %3")
                                .arg(path)
                                .arg(node_rows.size())
                                .arg(calibration.rows);
        return calibration;
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    calibration.node_px.assign(static_cast<std::size_t>(calibration.rows) *
                                   static_cast<std::size_t>(calibration.cols),
                               cv::Point2d(nan, nan));
    for (int r = 0; r < calibration.rows; ++r) {
        const QJsonArray node_row = node_rows.at(r).toArray();
        if (node_row.size() != calibration.cols) {
            calibration.error = QStringLiteral("%1: node_px row %2 has %3 entries, lattice says %4")
                                    .arg(path)
                                    .arg(r)
                                    .arg(node_row.size())
                                    .arg(calibration.cols);
            return calibration;
        }
        for (int c = 0; c < calibration.cols; ++c) {
            const QJsonArray node = node_row.at(c).toArray();
            // JSON stores an unmeasured node as null (or [null, null]); numpy reads
            // that as NaN and so does this, so a hole stays a hole.
            if (node.size() == 2 && !node.at(0).isNull() && !node.at(1).isNull()) {
                calibration.node_px[static_cast<std::size_t>(r) * static_cast<std::size_t>(calibration.cols) +
                                    static_cast<std::size_t>(c)] =
                    cv::Point2d(node.at(0).toDouble(), node.at(1).toDouble());
            }
        }
    }

    const QJsonObject norm  = fit.value(QStringLiteral("norm")).toObject();
    const QJsonArray centre = norm.value(QStringLiteral("centre")).toArray();
    if (centre.size() != 2) {
        calibration.error = QStringLiteral("%1: norm.centre is not a pair").arg(path);
        return calibration;
    }
    calibration.norm_centre = cv::Point2d(centre.at(0).toDouble(), centre.at(1).toDouble());
    calibration.norm_scale  = norm.value(QStringLiteral("scale")).toDouble(1.0);
    if (calibration.norm_scale == 0.0) {
        calibration.norm_scale = 1.0;
    }

    const QJsonArray coefficients = fit.value(QStringLiteral("coeffs")).toArray();
    if (coefficients.size() != 3) {
        calibration.error = QStringLiteral("%1: an affine fit needs 3 coefficient rows, found %2")
                                .arg(path)
                                .arg(coefficients.size());
        return calibration;
    }
    for (int i = 0; i < 3; ++i) {
        const QJsonArray pair = coefficients.at(i).toArray();
        if (pair.size() != 2) {
            calibration.error = QStringLiteral("%1: coeffs row %2 is not a pair").arg(path).arg(i);
            return calibration;
        }
        calibration.coeffs[i][0] = pair.at(0).toDouble();
        calibration.coeffs[i][1] = pair.at(1).toDouble();
    }

    calibration.loaded = true;
    calibration.error.clear();
    load_elastomer_origin(calibration, origin_path);
    return calibration;
}

} // namespace e_bts

#endif // E_BTS_PIXEL_TO_MM_H
