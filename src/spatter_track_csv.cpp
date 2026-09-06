// E_BTS_spatter_track_csv -- run the spatter tracker over a recorded .raw and
// write one CSV row per (event window, live track).
//
// WHY OFFLINE RATHER THAN EXPORTING FROM THE GUI
// ----------------------------------------------
// SpatterTrackingSource renders a cv::Mat and nothing else, so the live pane
// produces pictures, not measurements. Adding an export path there would couple
// the parameters to record time: every re-tune of cell size or the association
// gate would need a fresh indentation. Reading the .raw back instead makes the
// recording the fixed artefact and the tracking a re-runnable analysis, which is
// what an experiment wants -- and the .raw is already being written.
//
// WINDOW LOSS IS A CORRECTNESS ISSUE HERE, NOT A LATENCY ONE
// ----------------------------------------------------------
// The live path calls EventWindowBuffer::pop_latest(), which deliberately drops
// superseded windows to keep the UI responsive. Offline, windows ARE the sample
// series: a dropped one is a hole in a displacement curve. So this reads with
// pop_oldest() against a large queue cap, and asserts at the end that the buffer
// dropped nothing.
//
// The tracker itself is the same SpatterTracker the GUI runs, constructed with
// the same defaults (including the ROI), so numbers here match what was seen on
// screen when the parameters are left alone.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "event_window_buffer.h"
#include "spatter_render.h"
#include "spatter_tracker.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/driver/camera.h>
#include <opencv2/videoio.hpp>

namespace {

// Big enough that a file reader delivering a long burst of events in one
// callback cannot overflow the queue before the drain loop runs.
constexpr std::size_t kOfflineQueueCapacity = 1u << 20;

constexpr Metavision::timestamp kDefaultCollectionTimeUs = 40'000;

void print_usage(const char *program_name) {
    std::cerr
        << "Usage:\n  " << program_name << " <recording.raw> [--out tracks.csv] [options]\n\n"
        << "Runs the spatter tracker over a recorded .raw and writes one row per\n"
        << "(window, live track). Defaults match src/spatter_tracker_config.h, so\n"
        << "with no options the output matches the GUI pane at its defaults.\n\n"
        << "Options:\n"
        << "  --out PATH             output CSV (default: <recording>_spatter.csv)\n"
        << "  --accum-us N           event window length (default " << kDefaultCollectionTimeUs
        << ", = one 25 Hz\n"
        << "                         illumination cycle -- see HANDOFF section 12.5)\n"
        << "  --cell-width N         grid cell width  (default " << e_bts::kDefaultSpatterCellWidth << ")\n"
        << "  --cell-height N        grid cell height (default " << e_bts::kDefaultSpatterCellHeight << ")\n"
        << "  --activation N         distinct pixels to activate a cell (default "
        << e_bts::kDefaultSpatterActivationThreshold << ")\n"
        << "  --min-size N           min bounding-box side (default " << e_bts::kDefaultSpatterMinSizePx << ")\n"
        << "  --max-size N           max bounding-box side (default " << e_bts::kDefaultSpatterMaxSizePx << ")\n"
        << "  --max-distance N       association gate  (default " << e_bts::kDefaultSpatterMaxDistancePx << ")\n"
        << "  --untracked N          windows before a track retires (default "
        << e_bts::kDefaultSpatterUntrackedThreshold << ")\n"
        << "  --roi X Y W H          region of interest (default " << e_bts::kDefaultSpatterRoiX << " "
        << e_bts::kDefaultSpatterRoiY << " " << e_bts::kDefaultSpatterRoiWidth << " "
        << e_bts::kDefaultSpatterRoiHeight << ")\n"
        << "  --full-frame           track the whole sensor instead of the ROI\n"
        << "  --video PATH           also write the tracker's view as .mp4 -- the same\n"
        << "                         frame the GUI's Spatter Tracking pane draws\n"
        << "  --video-fps N          playback rate (default 25 = real time at 40000 us)\n"
        << "  --video-scale N        integer upscale, 1-8 (default 2)\n"
        << "  --quiet                suppress the per-1000-window progress line\n\n"
        << "For marker tracking the defaults are wrong -- markers sit ~29-31 px apart,\n"
        << "so the 50 px gate spans a neighbour. Try:\n"
        << "  " << program_name << " camera.raw --cell-width 3 --cell-height 3 \\\n"
        << "      --activation 4 --min-size 12 --max-size 28 --max-distance 12\n";
}

bool parse_int(const char *text, int &out) {
    try {
        std::size_t consumed = 0;
        const int value      = std::stoi(text, &consumed);
        if (consumed != std::string(text).size()) {
            return false;
        }
        out = value;
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace

int main(int argc, char *argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::filesystem::path input_raw_path;
    std::filesystem::path out_csv_path;
    e_bts::SpatterTrackerParams params;   // defaults, ROI included
    Metavision::timestamp collection_time_us = kDefaultCollectionTimeUs;
    bool full_frame = false;
    bool quiet      = false;
    std::filesystem::path video_path;
    double video_fps = 25.0;   // 40000 us windows played back 1:1
    int video_scale  = 2;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next_int = [&](int &target) {
            if (index + 1 >= argc || !parse_int(argv[index + 1], target)) {
                std::cerr << "Error: " << argument << " needs an integer value.\n";
                std::exit(1);
            }
            ++index;
        };

        if (argument == "-h" || argument == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (argument == "--out") {
            if (index + 1 >= argc) {
                std::cerr << "Error: --out needs a path.\n";
                return 1;
            }
            out_csv_path = argv[++index];
        } else if (argument == "--accum-us") {
            int value = 0;
            next_int(value);
            if (value <= 0) {
                std::cerr << "Error: --accum-us must be positive.\n";
                return 1;
            }
            collection_time_us = value;
        } else if (argument == "--cell-width") {
            next_int(params.cell_width);
        } else if (argument == "--cell-height") {
            next_int(params.cell_height);
        } else if (argument == "--activation") {
            next_int(params.activation_threshold);
        } else if (argument == "--min-size") {
            next_int(params.min_size);
        } else if (argument == "--max-size") {
            next_int(params.max_size);
        } else if (argument == "--max-distance") {
            next_int(params.max_distance);
        } else if (argument == "--untracked") {
            next_int(params.untracked_threshold);
        } else if (argument == "--roi") {
            if (index + 4 >= argc || !parse_int(argv[index + 1], params.roi_x) ||
                !parse_int(argv[index + 2], params.roi_y) || !parse_int(argv[index + 3], params.roi_width) ||
                !parse_int(argv[index + 4], params.roi_height)) {
                std::cerr << "Error: --roi needs four integers: X Y W H.\n";
                return 1;
            }
            index += 4;
        } else if (argument == "--full-frame") {
            full_frame = true;
        } else if (argument == "--video") {
            if (index + 1 >= argc) {
                std::cerr << "Error: --video needs a path.\n";
                return 1;
            }
            video_path = argv[++index];
        } else if (argument == "--video-fps") {
            int value = 0;
            next_int(value);
            if (value <= 0) {
                std::cerr << "Error: --video-fps must be positive.\n";
                return 1;
            }
            video_fps = value;
        } else if (argument == "--video-scale") {
            next_int(video_scale);
            if (video_scale < 1 || video_scale > 8) {
                std::cerr << "Error: --video-scale must be 1-8.\n";
                return 1;
            }
        } else if (argument == "--quiet") {
            quiet = true;
        } else if (!argument.empty() && argument[0] == '-') {
            std::cerr << "Error: unknown option " << argument << "\n\n";
            print_usage(argv[0]);
            return 1;
        } else if (input_raw_path.empty()) {
            input_raw_path = argument;
        } else {
            std::cerr << "Error: more than one input file given.\n";
            return 1;
        }
    }

    if (input_raw_path.empty()) {
        std::cerr << "Error: no input .raw given.\n\n";
        print_usage(argv[0]);
        return 1;
    }
    if (!std::filesystem::exists(input_raw_path)) {
        std::cerr << "Error: " << input_raw_path << " does not exist.\n";
        return 1;
    }
    if (out_csv_path.empty()) {
        out_csv_path = input_raw_path;
        out_csv_path.replace_extension();
        out_csv_path += "_spatter.csv";
    }

    try {
        Metavision::Camera camera = Metavision::Camera::from_file(input_raw_path.string(), false);
        const int width  = static_cast<int>(camera.geometry().width());
        const int height = static_cast<int>(camera.geometry().height());

        if (full_frame) {
            params.roi_x = params.roi_y = 0;
            params.roi_width  = width;
            params.roi_height = height;
        }

        e_bts::SpatterTracker tracker(width, height, params);
        const e_bts::SpatterTrackerParams &effective = tracker.params();
        const cv::Rect roi = tracker.roi();

        auto event_windows =
            std::make_shared<e_bts::EventWindowBuffer>(width, height, collection_time_us, kOfflineQueueCapacity);

        std::ofstream csv(out_csv_path);
        if (!csv.is_open()) {
            std::cerr << "Error: could not open " << out_csv_path << " for writing.\n";
            return 1;
        }
        // A comment header, so the parameters that produced the file travel with
        // it. Reproducing a plot six weeks later otherwise means guessing which
        // gate was in force. The plotting script skips '#' lines.
        csv << "# source=" << input_raw_path.filename().string() << "\n"
            << "# sensor=" << width << "x" << height << "\n"
            << "# accum_us=" << collection_time_us << "\n"
            << "# cell=" << effective.cell_width << "x" << effective.cell_height << "\n"
            << "# activation=" << effective.activation_threshold << "\n"
            << "# size=" << effective.min_size << "-" << effective.max_size << "\n"
            << "# max_distance=" << effective.max_distance << "\n"
            << "# untracked=" << effective.untracked_threshold << "\n"
            << "# roi=" << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << "\n";
        csv << "window_index,t_us,track_id,cx,cy,box_x,box_y,box_w,box_h,pixel_count,window_count\n";

        std::atomic_bool camera_error{false};
        camera.add_runtime_error_callback([&camera_error](const Metavision::CameraException &error) {
            std::cerr << "Metavision runtime error: " << error.what() << '\n';
            camera_error = true;
        });
        camera.cd().add_callback(
            [event_windows](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
                event_windows->add_events(begin, end);
            });

        // Opened before the stream starts so a bad codec/path fails immediately
        // rather than after minutes of reading.
        cv::VideoWriter video;
        const bool want_video = !video_path.empty();
        if (want_video) {
            const cv::Size frame_size(width * video_scale, height * video_scale);
            if (!video.open(video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            video_fps, frame_size, true)) {
                std::cerr << "Error: could not open " << video_path << " for writing.\n";
                return 1;
            }
            std::cout << "  video " << frame_size.width << "x" << frame_size.height << " @ "
                      << video_fps << " fps -> " << video_path.string() << '\n';
        }

        std::uint64_t window_index = 0;
        std::uint64_t row_count    = 0;
        std::uint64_t empty_windows = 0;
        std::uint64_t video_frames  = 0;
        Metavision::timestamp first_t = -1;
        Metavision::timestamp last_t  = 0;

        // Drains everything queued so far. Called both while the file streams and
        // once after it ends, so the tail of the recording is not left behind.
        const auto drain = [&]() {
            e_bts::EventWindow event_window;
            while (event_windows->pop_oldest(event_window)) {
                const std::vector<e_bts::SpatterCluster> &clusters = tracker.update(event_window);
                if (first_t < 0) {
                    first_t = event_window.start_us;
                }
                last_t = event_window.end_us;
                if (clusters.empty()) {
                    ++empty_windows;
                }
                for (const e_bts::SpatterCluster &cluster : clusters) {
                    csv << window_index << ',' << cluster.t << ',' << cluster.id << ','
                        << cluster.center.x << ',' << cluster.center.y << ',' << cluster.box.x << ','
                        << cluster.box.y << ',' << cluster.box.width << ',' << cluster.box.height << ','
                        << cluster.pixel_count << ',' << cluster.window_count << '\n';
                    ++row_count;
                }

                if (want_video) {
                    // Same renderer as the live pane (spatter_render.h), so the
                    // video is evidence about the tracker rather than a second
                    // drawing that could disagree with it.
                    const cv::Mat occupied_pixels =
                        make_occupied_pixel_frame(event_window, width, height);
                    cv::Mat frame = render_spatter_frame(
                        event_window, occupied_pixels, clusters, tracker.params(),
                        tracker.diagnostics(), tracker.track_count(), 0, roi);
                    if (video_scale != 1) {
                        // INTER_NEAREST: these are single-pixel events, and any
                        // smoothing invents brightness that was never measured.
                        cv::resize(frame, frame, cv::Size(), video_scale, video_scale,
                                   cv::INTER_NEAREST);
                    }
                    video.write(frame);
                    ++video_frames;
                }

                ++window_index;
                if (!quiet && window_index % 1000 == 0) {
                    std::cout << "  " << window_index << " windows, " << row_count << " rows\r" << std::flush;
                }
            }
        };

        std::cout << "Reading " << input_raw_path.filename().string() << " (" << width << "x" << height
                  << ") at " << collection_time_us << " us/window\n"
                  << "  roi " << roi.x << "," << roi.y << " " << roi.width << "x" << roi.height << " | cell "
                  << effective.cell_width << "x" << effective.cell_height << " | act "
                  << effective.activation_threshold << " | size " << effective.min_size << "-"
                  << effective.max_size << " | gate " << effective.max_distance << " | keep "
                  << effective.untracked_threshold << '\n';

        camera.start();
        while (camera.is_running() && !camera_error) {
            drain();
        }
        camera.stop();
        event_windows->flush();   // the final partial window
        drain();

        csv.flush();
        if (!csv) {
            std::cerr << "Error: writing " << out_csv_path << " failed.\n";
            return 1;
        }
        csv.close();

        if (want_video) {
            video.release();
            std::cout << "Wrote " << video_path.string() << " (" << video_frames << " frames, "
                      << video_frames / video_fps << " s at " << video_fps << " fps)\n";
        }

        const std::uint64_t dropped = event_windows->dropped_window_count();
        const double duration_s     = first_t < 0 ? 0.0 : (last_t - first_t) / 1e6;
        std::cout << "\nWrote " << out_csv_path.string() << '\n'
                  << "  " << window_index << " windows over " << duration_s << " s, " << row_count
                  << " track rows, " << empty_windows << " windows with no live track\n";
        if (dropped != 0) {
            // Cannot happen at the offline queue capacity, but a silent hole in a
            // displacement series is exactly the kind of thing that gets believed.
            std::cerr << "  WARNING: the buffer dropped " << dropped
                      << " windows -- the series has holes and the plots will be wrong.\n";
            return 1;
        }
        if (row_count == 0) {
            std::cerr << "  WARNING: no tracks at all. Check the ROI against the marker band,\n"
                      << "           and remember the size gate rejects on BOTH bounding-box sides.\n";
        }
        return camera_error ? 1 : 0;
    } catch (const Metavision::CameraException &error) {
        std::cerr << "Metavision error: " << error.what() << '\n';
        return 1;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
