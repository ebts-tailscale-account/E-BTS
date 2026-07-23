#ifndef E_BTS_CAMERA_UTILS_H
#define E_BTS_CAMERA_UTILS_H

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <string>
#include <thread>

#include "camera_feed_utils.h"
#include "path_utils.h"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/camera_exception.h>
#include <opencv2/videoio.hpp>

namespace e_bts {

struct RawToCsvResult {
    std::uint64_t cd_events      = 0;
    std::uint64_t trigger_events = 0;
    bool success                 = false;
};

struct RawToVideoResult {
    std::uint64_t frames_written = 0;
    bool success                 = false;
};

inline bool looks_like_raw_file(const std::string &source) {
    constexpr const char *kRawExtension = ".raw";
    return source.size() >= std::char_traits<char>::length(kRawExtension) &&
           source.compare(source.size() - std::char_traits<char>::length(kRawExtension),
                          std::char_traits<char>::length(kRawExtension), kRawExtension) == 0;
}

inline const char *source_type_label(Metavision::OnlineSourceType source_type) {
    switch (source_type) {
    case Metavision::OnlineSourceType::EMBEDDED:
        return "embedded";
    case Metavision::OnlineSourceType::USB:
        return "usb";
    case Metavision::OnlineSourceType::REMOTE:
        return "remote";
    }

    return "unknown";
}

inline void print_available_sources(std::ostream &out = std::cout) {
    const auto sources = Metavision::Camera::list_online_sources();

    out << "Available camera sources:\n";
    bool found = false;
    for (const auto &[source_type, serials] : sources) {
        for (const auto &serial : serials) {
            found = true;
            out << "  " << source_type_label(source_type) << ": " << serial << '\n';
        }
    }

    if (!found) {
        out << "  none\n";
    }
}

inline Metavision::Camera open_first_available_or_serial(int argc, char *argv[]) {
    if (argc > 1) {
        return Metavision::Camera::from_serial(argv[1]);
    }

    return Metavision::Camera::from_first_available();
}

inline Metavision::Camera open_first_available_serial_or_raw(int argc, char *argv[]) {
    if (argc <= 1) {
        return Metavision::Camera::from_first_available();
    }

    const std::string source = argv[1];
    if (looks_like_raw_file(source)) {
        return Metavision::Camera::from_file(source, true);
    }

    return Metavision::Camera::from_serial(source);
}

inline void set_camera_runtime_error_callback(Metavision::Camera &camera, std::atomic_bool &camera_error) {
    camera.add_runtime_error_callback([&camera_error](const Metavision::CameraException &error) {
        std::cerr << "Camera runtime error: " << error.what() << '\n';
        camera_error = true;
    });
}

inline RawToCsvResult convert_raw_to_csv(const std::filesystem::path &input_raw_path,
                                         const std::filesystem::path &cd_csv_path,
                                         const std::filesystem::path *trigger_csv_path = nullptr,
                                         std::ostream &out = std::cout,
                                         std::ostream &err = std::cerr) {
    RawToCsvResult result;

    try {
        std::ofstream cd_csv(cd_csv_path);
        if (!cd_csv.is_open()) {
            err << "Could not open CD output file: " << cd_csv_path << '\n';
            return result;
        }

        std::ofstream trigger_csv;
        const bool write_triggers = trigger_csv_path != nullptr;
        if (write_triggers) {
            trigger_csv.open(*trigger_csv_path);
            if (!trigger_csv.is_open()) {
                err << "Could not open trigger output file: " << *trigger_csv_path << '\n';
                return result;
            }
        }

        cd_csv << "x,y,polarity,timestamp_us\n";
        if (write_triggers) {
            trigger_csv << "value,id,timestamp_us\n";
        }

        Metavision::Camera camera = Metavision::Camera::from_file(input_raw_path.string(), false);

        std::atomic_bool camera_error{false};
        camera.add_runtime_error_callback([&camera_error, &err](const Metavision::CameraException &error) {
            err << "Metavision runtime error while reading RAW: " << error.what() << '\n';
            camera_error = true;
        });

        std::atomic<std::uint64_t> cd_count{0};
        camera.cd().add_callback([&cd_csv, &cd_count](const Metavision::EventCD *begin,
                                                       const Metavision::EventCD *end) {
            for (auto it = begin; it != end; ++it) {
                cd_csv << it->x << ',' << it->y << ',' << it->p << ',' << e_bts::timestamp_us(it->t) << '\n';
            }
            cd_count.fetch_add(static_cast<std::uint64_t>(std::distance(begin, end)), std::memory_order_relaxed);
        });

        std::atomic<std::uint64_t> trigger_count{0};
        if (write_triggers) {
            try {
                camera.ext_trigger().add_callback([&trigger_csv, &trigger_count](
                                                       const Metavision::EventExtTrigger *begin,
                                                       const Metavision::EventExtTrigger *end) {
                    for (auto it = begin; it != end; ++it) {
                        trigger_csv << it->p << ',' << it->id << ',' << e_bts::timestamp_us(it->t) << '\n';
                    }
                    trigger_count.fetch_add(static_cast<std::uint64_t>(std::distance(begin, end)),
                                            std::memory_order_relaxed);
                });
            } catch (const Metavision::CameraException &error) {
                err << "Trigger events are not available for this RAW file: " << error.what() << '\n';
            }
        }

        out << "Reading " << input_raw_path << " and writing " << cd_csv_path << "...\n";
        if (!camera.start()) {
            err << "Could not start RAW file reader.\n";
            return result;
        }

        while (camera.is_running() && !camera_error) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        camera.stop();
        cd_csv.flush();
        if (write_triggers) {
            trigger_csv.flush();
        }

        result.cd_events      = cd_count.load();
        result.trigger_events = trigger_count.load();
        result.success        = !camera_error;

        out << "Wrote " << result.cd_events << " CD events to " << cd_csv_path << '\n';
        if (write_triggers) {
            out << "Wrote " << result.trigger_events << " trigger events to " << *trigger_csv_path << '\n';
        }
    } catch (const Metavision::CameraException &error) {
        err << "Metavision error: " << error.what() << '\n';
    } catch (const std::exception &error) {
        err << "Error: " << error.what() << '\n';
    }

    return result;
}

inline void connect_cd_events_to_frame_generator(Metavision::Camera &camera,
                                                 Metavision::CDFrameGenerator &frame_generator) {
    camera.cd().add_callback([&frame_generator](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        frame_generator.add_events(begin, end);
    });
}

// Renders a RAW recording to an .mp4 by replaying it through the same
// CDFrameGenerator machinery the live Camera pane uses for its preview, then
// encoding each generated frame with OpenCV's VideoWriter. Frame generation
// is paced by the *event* timestamps embedded in the RAW file (see
// PeriodicFrameGenerationAlgorithm), not wall-clock time, so it stays
// correct regardless of from_file()'s replay speed. process_all_frames=true
// is required for the same reason: unlike the live view (which is fine
// dropping a frame under load), a file conversion must not silently skip
// frames as fast replay backs up the generator's internal queue.
//
// CDFrameGenerator::add_events() is synchronous (called straight from the
// RAW reader's callback), but the frames it produces are generated
// asynchronously on the generator's own worker thread -- for a small file
// replayed "as fast as possible", the reader can finish delivering every
// event (camera.is_running() goes false) well before that worker thread has
// rendered the corresponding frames. Calling frame_generator.stop() at that
// point does NOT drain the backlog; it just forces one partial frame and
// discards the rest, silently truncating the tail of the video. So once
// reading is done, this waits for the last generated frame's timestamp to
// catch up to the last observed event's timestamp before stopping.
inline RawToVideoResult convert_raw_to_video(const std::filesystem::path &input_raw_path,
                                             const std::filesystem::path &video_path,
                                             Metavision::timestamp accumulation_time_us = 10'000,
                                             CameraFeedMode camera_feed_mode = CameraFeedMode::Default,
                                             std::ostream &out = std::cout,
                                             std::ostream &err = std::cerr) {
    RawToVideoResult result;

    try {
        Metavision::Camera camera = Metavision::Camera::from_file(input_raw_path.string(), false);

        const int width  = camera.geometry().width();
        const int height = camera.geometry().height();

        Metavision::CDFrameGenerator frame_generator(width, height, /*process_all_frames=*/true);
        frame_generator.set_display_accumulation_time_us(accumulation_time_us);
        apply_camera_feed_mode(frame_generator, camera_feed_mode);

        std::atomic_bool camera_error{false};
        camera.add_runtime_error_callback([&camera_error, &err](const Metavision::CameraException &error) {
            err << "Metavision runtime error while reading RAW: " << error.what() << '\n';
            camera_error = true;
        });

        std::atomic<Metavision::timestamp> last_event_ts{0};
        camera.cd().add_callback([&frame_generator, &last_event_ts](const Metavision::EventCD *begin,
                                                                     const Metavision::EventCD *end) {
            if (begin != end) {
                last_event_ts.store(std::prev(end)->t, std::memory_order_relaxed);
            }
            frame_generator.add_events(begin, end);
        });

        const double fps = 1'000'000.0 / static_cast<double>(accumulation_time_us);

        cv::VideoWriter writer;
        std::atomic<std::uint64_t> frame_count{0};
        std::atomic<Metavision::timestamp> last_frame_ts{0};
        std::atomic_bool writer_error{false};

        const bool generator_started = frame_generator.start(
            static_cast<std::uint16_t>(fps), [&](Metavision::timestamp frame_ts, cv::Mat &frame) {
                if (!writer.isOpened()) {
                    const bool is_color = frame.channels() == 3;
                    if (!writer.open(video_path.string(), cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps,
                                     frame.size(), is_color)) {
                        err << "Could not open video output file: " << video_path << '\n';
                        writer_error = true;
                        return;
                    }
                }
                writer.write(frame);
                frame_count.fetch_add(1, std::memory_order_relaxed);
                last_frame_ts.store(frame_ts, std::memory_order_relaxed);
            });

        if (!generator_started) {
            err << "Could not start frame generator.\n";
            return result;
        }

        out << "Reading " << input_raw_path << " and writing " << video_path << "...\n";
        if (!camera.start()) {
            err << "Could not start RAW file reader.\n";
            frame_generator.stop();
            return result;
        }

        while (camera.is_running() && !camera_error && !writer_error) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Grace period for the frame-generation worker thread to catch up to
        // the last event before stopping it (see comment above); bounded so a
        // stalled generator can't hang the conversion forever.
        constexpr std::chrono::milliseconds kCatchUpTimeout{30'000};
        const auto catch_up_deadline = std::chrono::steady_clock::now() + kCatchUpTimeout;
        while (!camera_error && !writer_error &&
               last_frame_ts.load(std::memory_order_relaxed) + accumulation_time_us <
                   last_event_ts.load(std::memory_order_relaxed) &&
               std::chrono::steady_clock::now() < catch_up_deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }

        camera.stop();
        frame_generator.stop();
        if (writer.isOpened()) {
            writer.release();
        }

        result.frames_written = frame_count.load();
        result.success        = !camera_error && !writer_error && result.frames_written > 0;

        out << "Wrote " << result.frames_written << " frames to " << video_path << '\n';
    } catch (const Metavision::CameraException &error) {
        err << "Metavision error: " << error.what() << '\n';
    } catch (const std::exception &error) {
        err << "Error: " << error.what() << '\n';
    }

    return result;
}

} // namespace e_bts

#endif // E_BTS_CAMERA_UTILS_H
