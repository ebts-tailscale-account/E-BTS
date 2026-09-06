// E_BTS_contact_replay -- re-run the LIVE contact estimator over a recorded .raw.
//
//     ./build/E_BTS_contact_replay --raw recordings/<run>/camera.raw \
//         --time-ref recordings/<run>/contact.csv \
//         --out recordings/<run>/contact_replay.csv
//     python3 ml/contact_error.py recordings/<run> --contact contact_replay.csv
//
// WHY THIS EXISTS
// ---------------
// The ladder run of 2026-09-06 measured a positional GAIN of 0.59 (x) / 0.69 (y):
// move the indenter 1 mm and the reported contact moved 0.6 mm. The cause was
// found -- CircleDetector::detect_from_map() was searching a +-9 px square around
// each marker's REST position, while a marker beside a 5 mm indent moves 15-25 px,
// so the markers carrying the contact signal sat outside their own search window
// (see TemporalCircleTracker::circle_map_search_centers()). The fix is in.
//
// The question that then decides ~2.8 h of robot time is simply: did it work?
//
// The only evidence so far is E_BTS_marker_travel_test, which is honest about its
// own scope -- it measures marker TRACKING on a synthetic lattice of hard discs,
// not localisation on a real marker image. Re-running the campaign to find out
// costs 2.8 h of arm time and ~66 GB, and answers on a DIFFERENT sample of poke
// locations, so a better gain there is confounded with having drawn easier pokes.
//
// This replays the estimator over the .raw the run already wrote. That makes the
// comparison PAIRED: identical pokes, identical depths, identical baseline, the
// same 211k event windows -- the only thing that changes is the code. A gain that
// moves from 0.59 to ~1.0 on the very same windows is attributable to the fix and
// to nothing else. No robot, no new recording.
//
// It also answers a question the robot cannot. kContactMinimumDivergence is a
// starting value, not a calibrated one, and 17% of that run's pokes produced no
// estimate at all. A threshold cannot be swept with robot time -- you cannot
// re-poke to try 1.2 instead of 1.5. It can be swept over a recording, and this
// writes the peak strength for EVERY window (see --min-divergence) so a single
// pass supports any threshold offline.
//
// WHAT IS AND IS NOT THE SAME AS THE LIVE RUN
// -------------------------------------------
// Same: the estimator. CircleTrackingSource::process_window() is the one the GUI
// calls; this differs only in taking every window in order rather than the latest
// (pop_oldest vs pop_latest) and in not drawing a frame nobody will look at.
//
// Same: the baseline. It is rebuilt from the head of the recording rather than
// copied from the run -- the campaign opens with the pad unloaded, which is the
// condition a baseline needs.
//
// ⚠ THE EPOCH. This is the one that cost the most, so it is first.
//
// The .raw is REBASED TO ZERO when it is written; the GUI's contact.csv is not.
// On the ladder run the live log runs from window_end_us 90.75 s to 8545.19 s
// while the file runs 0 to 8454.4 s -- the SAME 8454.4 s of recording, offset by a
// constant 90.7 s. Feeding raw timestamps into a fit made against live ones
// therefore shifts every window 90.7 s from where it belongs.
//
// It hid well. The campaign's poke cycle is ~7.5 s and 90.7 s is 12.09 of them, so
// a 90.7 s error is only ~0.7 s from a whole number of cycles: the first minutes
// still lined up, and the alignment slipped away only as the cycle length varied
// with the depth ladder. The estimator looked correct for 150 s and then appeared
// to report contacts during out-of-contact holds and miss them during dwells --
// which reads exactly like a broken estimator and is nothing of the kind.
//
// So both series are anchored at their FIRST ROW, which is the same physical
// instant (recording start) on both clocks, and the constant never enters. The
// summary prints both spans as a standing check: same recording, same duration.
//
// Different, and it matters: window boundaries. The live buffer began
// accumulating when the tracking pane opened, which was before recording started,
// so its 40 ms grid has a phase offset this replay cannot reproduce (the file
// begins at a different event). Windows therefore do not correspond one-to-one
// with the live run's. Per-window rows are not comparable; per-poke aggregates
// over a ~1 s dwell are, which is what ml/contact_error.py reports.
//
// ⚠ THE CLOCK. ml/contact_error.py joins the camera to the robot on unix_time_s,
// and a replay has no wall clock of its own -- the recording is being read in
// 2026, not lived through. --time-ref recovers it: the run's own contact.csv
// carries both window_end_us (camera clock) and unix_time_s (workstation clock)
// for 211k windows, so a least-squares line through them IS the run's measured
// correspondence. On the ladder run the two clocks agree to 1 part in 1e6 over
// 8454 s and the fit's residual is 4 ms sd against 40 ms windows. Without
// --time-ref the output cannot be joined to franka.csv at all, and this says so
// rather than writing a plausible-looking column of nonsense.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <QCoreApplication>
#include <QString>

#include <metavision/sdk/driver/camera.h>

#include "circle_tracker_config.h"
#include "circle_tracking.h"
#include "pixel_to_mm.h"

namespace {

struct Options {
    std::string raw_path;
    std::string out_path;
    std::string time_ref_path;
    std::string calibration_path;
    std::string origin_path;
    Metavision::timestamp accumulation_time_us = 40'000;
    double minimum_circle_density              = e_bts::kDefaultMinimumCircleDensity;
    double minimum_divergence                  = e_bts::kContactMinimumDivergence;
    double progress_interval_s                 = 60.0;
    double limit_s                             = 0.0;   // 0 = the whole file
    Metavision::timestamp start_us             = -1;    // <0 = follow --time-ref
    bool legacy_search                         = false;
    double dump_field_at_s                     = -1.0;  // <0 = no dump
    std::string dump_field_path;
};

void print_usage() {
    std::cerr
        << "usage: E_BTS_contact_replay --raw <camera.raw> --out <contact.csv>\n"
           "                            [--time-ref <the run's original contact.csv>]\n"
           "                            [--accum-us 40000] [--density 0.50]\n"
           "                            [--min-divergence 1.5] [--limit-s 0]\n"
           "                            [--calib calibration/pixel_to_mm.json]\n"
           "                            [--origin calibration/elastomer_origin.json]\n"
           "                            [--progress-s 60]\n\n"
           "  --time-ref   recovers the wall clock from the run's own log; without it the\n"
           "               output cannot be joined to franka.csv (see the file header).\n"
           "  --accum-us   MUST match the run. The ladder run used 40000; the live default\n"
           "               is 10000, so this is not the value to leave alone.\n"
           "  --min-divergence  the accept threshold. The peak strength is written for every\n"
           "               window regardless, so one pass supports any threshold afterwards.\n"
           "  --limit-s    stop after N seconds of recording -- for a smoke test.\n"
           "  --start-us   begin at this camera timestamp. Defaults to where the live log\n"
           "               began; the footage before that is setup, not a settled pad.\n"
           "  --legacy-search  search around marker REST sites, as the code did before the\n"
           "               fix. Run it against the default to attribute a change to the fix\n"
           "               rather than to the pokes.\n"
           "  --dump-field-at-s / --dump-field-out   write the per-cell displacement field and\n"
           "               its divergence for the first accepted window at/after that RECORDING\n"
           "               second. For figures; see ml/contact_method_report.py.\n";
}

bool parse_args(int argc, char *argv[], Options &options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        auto next_value = [&](const char *name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << name << " needs a value\n";
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--raw") {
            options.raw_path = next_value("--raw");
        } else if (arg == "--out") {
            options.out_path = next_value("--out");
        } else if (arg == "--time-ref") {
            options.time_ref_path = next_value("--time-ref");
        } else if (arg == "--calib") {
            options.calibration_path = next_value("--calib");
        } else if (arg == "--origin") {
            options.origin_path = next_value("--origin");
        } else if (arg == "--accum-us") {
            options.accumulation_time_us = std::stoll(next_value("--accum-us"));
        } else if (arg == "--density") {
            options.minimum_circle_density = std::stod(next_value("--density"));
        } else if (arg == "--min-divergence") {
            options.minimum_divergence = std::stod(next_value("--min-divergence"));
        } else if (arg == "--progress-s") {
            options.progress_interval_s = std::stod(next_value("--progress-s"));
        } else if (arg == "--limit-s") {
            options.limit_s = std::stod(next_value("--limit-s"));
        } else if (arg == "--dump-field-at-s") {
            options.dump_field_at_s = std::stod(next_value("--dump-field-at-s"));
        } else if (arg == "--dump-field-out") {
            options.dump_field_path = next_value("--dump-field-out");
        } else if (arg == "--legacy-search") {
            options.legacy_search = true;
        } else if (arg == "--start-us") {
            options.start_us = std::stoll(next_value("--start-us"));
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(0);
        } else {
            std::cerr << "unknown argument: " << arg << "\n\n";
            print_usage();
            return false;
        }
    }
    if (options.raw_path.empty() || options.out_path.empty()) {
        print_usage();
        return false;
    }
    return true;
}

// unix_time_s = scale * window_end_us + offset, fitted from a run's own
// contact.csv. See the header for why this is a recovery of a measured
// correspondence rather than an assumption about either clock.
struct TimeBase {
    bool loaded  = false;
    double scale = 1e-6;
    double offset = 0.0;
    std::size_t sample_count = 0;
    double residual_sd_ms    = 0.0;
    double residual_max_ms   = 0.0;
    double live_span_s = 0.0;                   // the live log's duration
    Metavision::timestamp first_window_us = 0;  // the live log's first window
    double first_unix_s = 0.0;                  // ...and its wall clock

    // ⚠ The argument is a RAW-FILE timestamp, which is NOT the same clock as the
    // live log's window_end_us. See the epoch note in the file header: the .raw is
    // rebased to zero when it is written, the GUI's log is not, and on this run the
    // two differ by a constant 90.7 s. Anchoring both series at their first row --
    // both begin when recording begins -- is what removes that offset without
    // having to know it.
    double to_unix_s(Metavision::timestamp raw_window_end_us, Metavision::timestamp raw_origin_us) const {
        return first_unix_s + scale * static_cast<double>(raw_window_end_us - raw_origin_us);
    }
};

bool fit_time_base(const std::string &path, TimeBase &time_base, std::string &error) {
    std::ifstream input(path);
    if (!input.is_open()) {
        error = "cannot open " + path;
        return false;
    }
    std::string line;
    if (!std::getline(input, line)) {
        error = path + " is empty";
        return false;
    }
    // Locate the two columns by name rather than position: the header has grown
    // before and will again.
    int unix_column = -1;
    int window_column = -1;
    {
        std::stringstream header(line);
        std::string name;
        for (int index = 0; std::getline(header, name, ','); ++index) {
            if (name == "unix_time_s") {
                unix_column = index;
            } else if (name == "window_end_us") {
                window_column = index;
            }
        }
    }
    if (unix_column < 0 || window_column < 0) {
        error = path + " has no unix_time_s/window_end_us columns";
        return false;
    }

    // ⚠ CENTRED ON THE MEAN, and that is not a stylistic choice.
    //
    // window_end_us spans ~8.5e9 over a 2.3 h run. Fitting the normal equations
    // on the raw values makes the denominator n*Sxx - Sx^2 a difference of two
    // ~6e29 quantities that agree in their first several digits, so a double's
    // 16 available ones leave almost nothing. Centring on the FIRST sample does
    // not help -- x still spans the whole run. Centring on the MEAN makes Sx
    // exactly 0 and the denominator simply n*Sxx, with no cancellation at all.
    //
    // This was caught by comparing against numpy on the same file: the
    // first-sample version returned a residual of 289 ms where the true fit is
    // 4 ms, i.e. it silently mis-stated the clock by most of a window. Any
    // future change here should be checked the same way.
    std::vector<std::pair<double, double>> samples;
    std::size_t count = 0;
    while (std::getline(input, line)) {
        std::stringstream row(line);
        std::string field;
        double unix_time = 0.0, window_end = 0.0;
        bool have_unix = false, have_window = false;
        for (int index = 0; std::getline(row, field, ','); ++index) {
            if (field.empty()) {
                continue;
            }
            if (index == unix_column) {
                unix_time = std::atof(field.c_str());
                have_unix = true;
            } else if (index == window_column) {
                window_end  = std::atof(field.c_str());
                have_window = true;
            }
        }
        if (!have_unix || !have_window) {
            continue;
        }
        samples.emplace_back(window_end, unix_time);
        ++count;
    }
    if (count < 2) {
        error = path + " has fewer than 2 usable rows";
        return false;
    }
    const double n = static_cast<double>(count);

    double mean_x = 0.0, mean_y = 0.0;
    for (const auto &sample : samples) {
        mean_x += sample.first;
        mean_y += sample.second;
    }
    mean_x /= n;
    mean_y /= n;

    double sum_xx = 0.0, sum_xy = 0.0;
    for (const auto &sample : samples) {
        const double x = sample.first - mean_x;
        const double y = sample.second - mean_y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    if (sum_xx == 0.0) {
        error = path + " has no spread in window_end_us";
        return false;
    }
    const double slope = sum_xy / sum_xx;

    double sum_sq = 0.0, max_abs = 0.0;
    for (const auto &sample : samples) {
        const double residual = (sample.second - mean_y) - slope * (sample.first - mean_x);
        sum_sq += residual * residual;
        max_abs = std::max(max_abs, std::abs(residual));
    }

    const auto &first_sample = *std::min_element(samples.begin(), samples.end());
    time_base.first_window_us = static_cast<Metavision::timestamp>(first_sample.first);
    time_base.first_unix_s    = first_sample.second;
    time_base.live_span_s     = (std::max_element(samples.begin(), samples.end())->first
                                 - first_sample.first) * 1e-6;
    time_base.loaded          = true;
    time_base.scale           = slope;
    time_base.offset          = mean_y - slope * mean_x;
    time_base.sample_count    = count;
    time_base.residual_sd_ms  = std::sqrt(sum_sq / n) * 1e3;
    time_base.residual_max_ms = max_abs * 1e3;
    return true;
}

} // namespace

int main(int argc, char *argv[]) {
    QCoreApplication app(argc, argv);

    // ⚠ AFTER QCoreApplication, and not optional.
    //
    // QCoreApplication calls setlocale(LC_ALL, "") on Unix, which activates the
    // user's locale. This workstation runs LC_NUMERIC=kk_KZ.UTF-8, where the
    // decimal separator is a COMMA -- so strtod/atof stop at the '.' and read
    // "1788688925.731514" as 1788688925, silently discarding the fractional
    // second. The time base fitted from those values had a residual of 289 ms
    // against 40 ms windows, and 289 ms is 1/sqrt(12) s exactly: the standard
    // deviation of a uniform error over one whole second, which is the
    // fingerprint of truncation to integer seconds rather than of any noise.
    //
    // Nothing about the output would have looked wrong -- the timestamps are
    // plausible, monotonic, and correctly formatted (C++ streams are governed by
    // std::locale, which Qt does not touch, so WRITING was never affected). It
    // would simply have joined every camera window to the wrong robot pose by up
    // to half a second, in a run whose dwells last about one.
    std::setlocale(LC_NUMERIC, "C");

    Options options;
    if (!parse_args(argc, argv, options)) {
        return 2;
    }

    const QString calibration_path = options.calibration_path.empty()
                                         ? e_bts::default_pixel_to_mm_path()
                                         : QString::fromStdString(options.calibration_path);
    const QString origin_path      = options.origin_path.empty()
                                         ? e_bts::default_elastomer_origin_path()
                                         : QString::fromStdString(options.origin_path);

    const e_bts::PixelToMm calibration = e_bts::load_pixel_to_mm(calibration_path, origin_path);
    if (!calibration.loaded) {
        // Not a warning: without the calibration there is no x_robot_mm, and
        // x_robot_mm is the entire point of the comparison.
        std::cerr << "cannot load pixel->mm calibration: " << calibration.error.toStdString() << '\n'
                  << "  (run from the repo root, or pass --calib)\n";
        return 1;
    }
    std::cout << "calibration: " << calibration.source_path.toStdString();
    if (calibration.has_origin) {
        std::cout << "   origin (" << std::fixed << std::setprecision(2) << calibration.origin_mm.x << ", "
                  << calibration.origin_mm.y << ") mm\n";
    } else {
        std::cout << "   no elastomer origin (" << calibration.origin_error.toStdString()
                  << ") -- x_pad_mm will be empty, x_robot_mm is unaffected\n";
    }

    TimeBase time_base;
    if (options.time_ref_path.empty()) {
        std::cout << "\n⚠ no --time-ref: unix_time_s will be the CAMERA clock in seconds, which is\n"
                     "  NOT the workstation clock ml/contact_error.py joins on. This output can be\n"
                     "  inspected but not compared against franka.csv.\n\n";
    } else {
        std::string error;
        if (!fit_time_base(options.time_ref_path, time_base, error)) {
            std::cerr << "cannot fit the time base: " << error << '\n';
            return 1;
        }
        std::cout << "time base from " << options.time_ref_path << ": " << time_base.sample_count
                  << " rows, camera clock rate " << std::fixed << std::setprecision(7)
                  << time_base.scale * 1e6 << " (1.0 = identical to the workstation), residual sd "
                  << std::setprecision(2) << time_base.residual_sd_ms << " ms, max "
                  << time_base.residual_max_ms << " ms\n";
        // Two things a good fit must satisfy, checked rather than assumed. Both
        // were violated by the locale bug described in main(), which is the
        // reason they are here: it produced a fit that was wrong by half a
        // second while looking entirely ordinary.
        const double window_ms = static_cast<double>(options.accumulation_time_us) / 1e3;
        const double rate      = time_base.scale * 1e6;
        if (std::abs(rate - 1.0) > 1e-3) {
            std::cerr << "\ntime base REJECTED: the camera clock would have to run at " << rate
                      << "x the workstation's. Two free-running quartz clocks differ by parts per\n"
                         "million, not parts per thousand, so this is a parse or column error rather\n"
                         "than a measurement.\n";
            return 1;
        }
        if (time_base.residual_sd_ms > window_ms) {
            std::cerr << "\ntime base REJECTED: residual sd " << time_base.residual_sd_ms
                      << " ms exceeds one " << window_ms
                      << " ms window, so the recovered clock cannot place a window to better than\n"
                         "the window itself and the join to franka.csv would be meaningless.\n";
            return 1;
        }
        if (time_base.residual_sd_ms > 0.25 * window_ms) {
            std::cout << "  ⚠ that residual is a large fraction of one window; the join to franka.csv\n"
                         "    will be correspondingly blurred.\n";
        }
    }

    std::ofstream output(options.out_path);
    if (!output.is_open()) {
        std::cerr << "cannot open " << options.out_path << " for writing\n";
        return 1;
    }
    // The live header, unchanged, plus two appended columns. ml/contact_error.py
    // reads by name, so appending is invisible to it.
    output << "unix_time_s,window_end_us,valid,ambiguous,x_pad_mm,y_pad_mm,x_robot_mm,y_robot_mm,"
              "pixel_x,pixel_y,cell_col,cell_row,divergence_px_per_cell,coherence,tracked_markers,"
              "peak_found,second_divergence_px_per_cell\n";

    try {
        Metavision::Camera camera = Metavision::Camera::from_file(options.raw_path, false);
        const int width  = camera.geometry().width();
        const int height = camera.geometry().height();
        std::cout << "raw: " << options.raw_path << "   geometry " << width << "x" << height
                  << "   windows " << options.accumulation_time_us << " us   density "
                  << std::setprecision(2) << options.minimum_circle_density << "   min-divergence "
                  << options.minimum_divergence << "\n";

        // Start at the top of the file by default. An earlier version skipped to
        // the live log's first window_end_us on the theory that the opening ~90 s
        // were un-logged setup footage; that was wrong. Those 90 s are an EPOCH
        // difference, not footage -- the live log and the .raw cover exactly the
        // same 8454.4 s -- so skipping them discarded real data.
        Metavision::timestamp start_us = options.start_us > 0 ? options.start_us : 0;
        if (start_us > 0) {
            std::cout << "start: " << std::fixed << std::setprecision(1)
                      << static_cast<double>(start_us) / 1e6
                      << " s into the recording (--start-us). The wall-clock anchor then falls\n"
                         "       back to an assumed file origin of 0, which is right to within one "
                         "window.\n";
        }

        e_bts::CircleTrackingSource tracking(width, height, options.accumulation_time_us,
                                             options.minimum_circle_density);
        tracking.set_minimum_divergence(options.minimum_divergence);
        tracking.set_legacy_search_centers(options.legacy_search);

        // One window's internals, for the method figures. Written for the first
        // ACCEPTED window at or after the requested second, so the picture shows a
        // real contact rather than whatever the clock happened to land on.
        std::ofstream field_csv;
        bool field_written = false;
        if (options.dump_field_at_s >= 0.0 && !options.dump_field_path.empty()) {
            tracking.set_field_observer([&](const e_bts::DisplacementField &field,
                                            const std::vector<double> &divergence,
                                            const e_bts::ContactEstimate &estimate,
                                            Metavision::timestamp window_end_us) {
                if (field_written || !estimate.valid) {
                    return;
                }
                if (static_cast<double>(window_end_us) / 1e6 < options.dump_field_at_s) {
                    return;
                }
                field_csv.open(options.dump_field_path);
                if (!field_csv.is_open()) {
                    std::cerr << "cannot write " << options.dump_field_path << '\n';
                    field_written = true;
                    return;
                }
                field_csv << "# window_end_us=" << window_end_us << " rows=" << field.rows
                          << " cols=" << field.cols << " peak_row=" << estimate.cell_row
                          << " peak_col=" << estimate.cell_col << " peak_px_x=" << estimate.pixel.x
                          << " peak_px_y=" << estimate.pixel.y << " divergence=" << estimate.divergence
                          << " coherence=" << estimate.coherence
                          << " tracked=" << estimate.tracked_markers << '\n';
                field_csv << "row,col,ok,baseline_x,baseline_y,dx,dy,divergence\n";
                for (int r = 0; r < field.rows; ++r) {
                    for (int c = 0; c < field.cols; ++c) {
                        const std::size_t i = field.index(r, c);
                        field_csv << r << ',' << c << ',' << static_cast<int>(field.ok[i]) << ','
                                  << field.baseline[i].x << ',' << field.baseline[i].y << ','
                                  << field.dx[i] << ',' << field.dy[i] << ',';
                        if (std::isfinite(divergence[i])) {
                            field_csv << divergence[i];
                        }
                        field_csv << '\n';
                    }
                }
                field_csv.close();
                field_written = true;
                std::cout << "field dump: window " << std::fixed << std::setprecision(2)
                          << static_cast<double>(window_end_us) / 1e6 << " s -> "
                          << options.dump_field_path << '\n';
            });
        }
        if (options.legacy_search) {
            std::cout << "search centres: BASELINE REST SITES (--legacy-search, the pre-fix "
                         "behaviour)\n";
        }
        tracking.set_pixel_to_mm([calibration](const cv::Point2d &pixel, cv::Point2d &mm) {
            return calibration.pixel_to_pad_mm(pixel, mm);
        });

        std::uint64_t window_count = 0;
        std::uint64_t peak_count   = 0;
        std::uint64_t valid_count  = 0;
        Metavision::timestamp first_window_us = -1;
        Metavision::timestamp last_window_us  = 0;
        // The raw timestamp that corresponds to the live log's first row, i.e. the
        // moment recording began. Learned from the first window when replaying the
        // whole file; assumed to be the first window boundary when --start-us skips
        // past it (an error of at most one window).
        Metavision::timestamp raw_origin_us =
            start_us > 0 ? options.accumulation_time_us : 0;
        double next_progress_us = options.progress_interval_s * 1e6;
        bool limit_reached      = false;

        const auto sink = [&](const e_bts::ContactReading &reading) {
            if (limit_reached) {
                return;
            }
            if (first_window_us < 0) {
                first_window_us = reading.window_end_us;
                if (start_us == 0) {
                    raw_origin_us = reading.window_end_us;
                }
            }
            last_window_us = reading.window_end_us;
            const double elapsed_us =
                static_cast<double>(reading.window_end_us - first_window_us);
            if (options.limit_s > 0.0 && elapsed_us > options.limit_s * 1e6) {
                limit_reached = true;
                return;
            }

            const e_bts::ContactEstimate &contact = reading.estimate;
            ++window_count;
            peak_count += contact.peak_found ? 1 : 0;
            valid_count += contact.valid ? 1 : 0;

            // Robot-frame millimetres, computed exactly as CameraSessionWorker
            // does -- but gated on peak_found rather than valid, so that a
            // threshold lowered afterwards still has a position to use.
            cv::Point2d robot_mm(0.0, 0.0);
            const bool has_robot_mm =
                contact.peak_found && calibration.pixel_to_mm(contact.pixel, robot_mm);

            std::ostringstream line;
            const double unix_time_s =
                time_base.loaded ? time_base.to_unix_s(reading.window_end_us, raw_origin_us)
                                 : static_cast<double>(reading.window_end_us) * 1e-6;
            line << std::fixed << std::setprecision(6) << unix_time_s << ',' << reading.window_end_us
                 << ',' << (contact.valid ? 1 : 0) << ',' << (contact.ambiguous ? 1 : 0) << ',';
            line << std::setprecision(4);
            if (reading.has_mm) {
                line << reading.mm.x << ',' << reading.mm.y << ',';
            } else {
                line << ",,";
            }
            if (has_robot_mm) {
                line << robot_mm.x << ',' << robot_mm.y << ',';
            } else {
                line << ",,";
            }
            if (contact.peak_found) {
                line << std::setprecision(3) << contact.pixel.x << ',' << contact.pixel.y << ','
                     << contact.cell_col << ',' << contact.cell_row << ',' << contact.divergence << ','
                     << contact.coherence << ',';
            } else {
                line << ",,,,,,";
            }
            line << contact.tracked_markers << ',' << (contact.peak_found ? 1 : 0) << ','
                 << std::setprecision(3) << contact.second_divergence << '\n';
            output << line.str();

            if (options.progress_interval_s > 0.0 && elapsed_us >= next_progress_us) {
                next_progress_us += options.progress_interval_s * 1e6;
                std::cout << "  " << std::fixed << std::setprecision(0) << elapsed_us / 1e6
                          << " s of recording   " << window_count << " windows   " << valid_count
                          << " contacts (" << std::setprecision(1)
                          << 100.0 * static_cast<double>(valid_count) /
                                 static_cast<double>(std::max<std::uint64_t>(window_count, 1))
                          << "%)\n"
                          << std::flush;
            }
        };

        tracking.connect_to_camera_offline(camera, sink, start_us);

        std::atomic_bool camera_error{false};
        camera.add_runtime_error_callback([&camera_error](const Metavision::CameraException &error) {
            std::cerr << "Metavision runtime error: " << error.what() << '\n';
            camera_error = true;
        });

        camera.start();
        while (camera.is_running() && !camera_error && !limit_reached) {
            // The estimator runs on the reader's thread (see
            // connect_to_camera_offline), so there is nothing to do here but wait
            // for the file to be consumed.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        camera.stop();
        if (!limit_reached) {
            tracking.flush_offline(sink);
        }
        output.flush();
        output.close();

        const e_bts::TrackingBufferStats stats = tracking.buffer_stats();
        std::cout << "\ndone: " << window_count << " windows over "
                  << std::fixed << std::setprecision(1)
                  << static_cast<double>(last_window_us - std::max<Metavision::timestamp>(first_window_us, 0)) /
                         1e6
                  << " s of recording\n"
                  << "  peaks found   " << peak_count << " ("
                  << 100.0 * static_cast<double>(peak_count) /
                         static_cast<double>(std::max<std::uint64_t>(window_count, 1))
                  << "%)\n"
                  << "  above " << std::setprecision(2) << options.minimum_divergence << " px/cell  "
                  << valid_count << " (" << std::setprecision(1)
                  << 100.0 * static_cast<double>(valid_count) /
                         static_cast<double>(std::max<std::uint64_t>(window_count, 1))
                  << "%)\n"
                  << "  windows dropped by the buffer " << stats.dropped_windows
                  << "   (must be 0: a dropped window is a hole in the series)\n"
                  << "  -> " << options.out_path << '\n';
        // The check that would have caught the epoch bug immediately: the replay
        // and the live log describe the same recording, so they must span the same
        // amount of time. They did -- 8454.4 s each -- while their timestamps were
        // offset by a constant 90.7 s, which is exactly why the offset was invisible
        // until the tare rate was compared phase by phase.
        if (time_base.loaded && options.limit_s <= 0.0 && start_us == 0) {
            const double replay_span_s =
                static_cast<double>(last_window_us - std::max<Metavision::timestamp>(first_window_us, 0)) / 1e6;
            const double difference_s = std::abs(replay_span_s - time_base.live_span_s);
            std::cout << "  span vs the live log: " << std::setprecision(1) << replay_span_s << " s vs "
                      << time_base.live_span_s << " s (differ by " << std::setprecision(2) << difference_s
                      << " s)\n";
            if (difference_s > 5.0) {
                std::cerr << "\n⚠ the replay and the live log do not cover the same span. They are "
                             "supposed to be\n  the same recording, so the wall clock written here "
                             "cannot be trusted.\n";
            }
        }
        if (stats.dropped_windows != 0) {
            std::cerr << "\n⚠ windows were dropped. The synchronous drain in "
                         "connect_to_camera_offline() is supposed to make this impossible; "
                         "the result has gaps and should not be used.\n";
            return 1;
        }
        if (camera_error) {
            return 1;
        }
    } catch (const Metavision::CameraException &error) {
        std::cerr << "camera error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
