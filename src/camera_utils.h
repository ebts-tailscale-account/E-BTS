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

#include "path_utils.h"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/events/event_ext_trigger.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/camera_exception.h>

namespace e_bts {

struct RawToCsvResult {
    std::uint64_t cd_events      = 0;
    std::uint64_t trigger_events = 0;
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

} // namespace e_bts

#endif // E_BTS_CAMERA_UTILS_H
