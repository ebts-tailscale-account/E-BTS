#ifndef E_BTS_SEQUENCE_RECORDING_CONTROLLER_H
#define E_BTS_SEQUENCE_RECORDING_CONTROLLER_H

// Records event-camera data for one or more robotic-hand task sequences,
// triggered *externally* by a companion Python script that drives the
// robotic hand: the Python script drops a small "command file" into a
// control directory, this controller polls for it, and reacts
// (start / stop / quit). All recordings are written to the recordings
// directory. This is the same protocol the standalone E_BTS_record_sequence
// CLI has always used (see record_sequence.cpp), extracted here so the
// E_BTS_GUI's Sequence Recording pane can drive the same watcher against a
// camera session shared with the Camera/Circle Tracking panes.
//
// Protocol:
//
//   control/start.cmd  -> created by Python right before the hand begins a
//                          task. Its contents (optional) are used as the
//                          base name for the output file, e.g. writing
//                          "grasp_01" produces
//                          recordings/grasp_01_<timestamp>.raw. An empty
//                          file falls back to the base name "sequence".
//   control/stop.cmd   -> created by Python once the hand has finished the
//                         task. Recording is stopped immediately.
//   control/quit.cmd   -> created by Python to end the watch session. The
//                         standalone CLI exits entirely; the GUI just closes
//                         the Sequence Recording pane and leaves the rest of
//                         the app running (see camera_session_worker.cpp).
//
// Each command file is deleted once acted on, so the same three files can be
// reused for every sequence in a session. See record_sequence.cpp for the
// Python-side example.

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>

#include "path_utils.h"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/driver/camera.h>

namespace e_bts {

class SequenceRecordingController {
public:
    explicit SequenceRecordingController(std::filesystem::path control_dir    = "control",
                                         std::filesystem::path recordings_dir = "recordings") :
        control_dir_(std::move(control_dir)), recordings_dir_(std::move(recordings_dir)),
        start_file_(control_dir_ / kStartFileName), stop_file_(control_dir_ / kStopFileName),
        quit_file_(control_dir_ / kQuitFileName) {}

    void set_log_callback(std::function<void(const std::string &)> log_callback) {
        log_callback_ = std::move(log_callback);
    }

    // Registers this controller's CD-event counter with the shared camera.
    // Safe to call once per camera connection; independent of whatever other
    // callbacks CameraSource/CircleTrackingSource register on the same
    // camera.
    void connect_to_camera(Metavision::Camera &camera) {
        camera.cd().add_callback([this](const Metavision::EventCD *begin, const Metavision::EventCD *end) {
            total_events_.fetch_add(static_cast<std::uint64_t>(std::distance(begin, end)),
                                    std::memory_order_relaxed);
        });
    }

    // Creates control_dir/recordings_dir and clears any stale command files
    // left over from a previous watch session.
    void start_watching() {
        std::filesystem::create_directories(control_dir_);
        std::filesystem::create_directories(recordings_dir_);

        std::error_code ignore_ec;
        std::filesystem::remove(start_file_, ignore_ec);
        std::filesystem::remove(stop_file_, ignore_ec);
        std::filesystem::remove(quit_file_, ignore_ec);

        watching_ = true;
        log("Watching for commands in '" + control_dir_.string() + "' (start.cmd / stop.cmd / quit.cmd)...");
    }

    // Does not stop an in-progress recording; callers that want that should
    // call stop_recording_if_active() first.
    void stop_watching() {
        watching_ = false;
    }

    bool is_watching() const {
        return watching_;
    }

    bool is_recording() const {
        return recording_;
    }

    // Call periodically (e.g. every 50ms) while watching. Returns true if a
    // quit.cmd was consumed -- the caller decides what "quit" means for it
    // (exit the process, or just close the pane).
    bool poll(Metavision::Camera &camera) {
        if (!watching_) {
            return false;
        }

        if (consume_command_file(quit_file_)) {
            log("Quit signal received.");
            return true;
        }

        if (!recording_) {
            std::string requested_name;
            if (consume_command_file(start_file_, &requested_name)) {
                begin_recording(camera, requested_name, "Start signal received.");
            }
        } else if (consume_command_file(stop_file_)) {
            stop_recording_if_active(camera);
        }

        return false;
    }

    // Starts a recording immediately rather than waiting for start.cmd, e.g.
    // from the Camera ribbon's manual "Record" button. Shares this
    // controller's single recording state with the start.cmd/stop.cmd
    // watcher, so a manual recording and an externally-triggered one can
    // never both be active at once. No-op (returns false) if a recording is
    // already in progress.
    bool start_recording(Metavision::Camera &camera, const std::string &base_name = "manual") {
        if (recording_) {
            return false;
        }
        begin_recording(camera, base_name, "Manual recording started.");
        return true;
    }

    // Stops an in-progress recording regardless of how it was started
    // (start.cmd or a manual ribbon trigger sharing this controller).
    void stop_recording_if_active(Metavision::Camera &camera) {
        if (!recording_) {
            return;
        }

        camera.stop_recording();
        recording_ = false;

        const auto elapsed_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - recording_started_at_).count();
        const auto captured_events = total_events_.load(std::memory_order_relaxed) - events_before_recording_;
        std::ostringstream message;
        message << "Stopped. Saved " << current_output_path_ << " (" << elapsed_s << "s, " << captured_events
                << " CD events observed).";
        log(message.str());
    }

private:
    static constexpr const char *kStartFileName       = "start.cmd";
    static constexpr const char *kStopFileName        = "stop.cmd";
    static constexpr const char *kQuitFileName        = "quit.cmd";
    static constexpr const char *kDefaultSequenceName = "sequence";

    static std::string read_and_trim(const std::filesystem::path &path) {
        std::ifstream in(path, std::ios::binary);
        std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

        const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
        contents.erase(contents.begin(), std::find_if(contents.begin(), contents.end(), not_space));
        contents.erase(std::find_if(contents.rbegin(), contents.rend(), not_space).base(), contents.end());
        return contents;
    }

    static bool consume_command_file(const std::filesystem::path &path, std::string *contents_out = nullptr) {
        std::error_code exists_ec;
        if (!std::filesystem::exists(path, exists_ec)) {
            return false;
        }

        if (contents_out != nullptr) {
            *contents_out = read_and_trim(path);
        }

        std::error_code remove_ec;
        std::filesystem::remove(path, remove_ec);
        return true;
    }

    std::filesystem::path make_recording_path(const std::string &base_name) const {
        const std::string sanitized =
            sanitized_filename_component(base_name.empty() ? kDefaultSequenceName : base_name);

        const auto now      = std::chrono::system_clock::now();
        const auto now_time = std::chrono::system_clock::to_time_t(now);
        std::tm local_time{};
        if (const std::tm *time_info = std::localtime(&now_time)) {
            local_time = *time_info;
        }

        std::ostringstream stamped_name;
        stamped_name << sanitized << "_" << std::put_time(&local_time, "%Y%m%d_%H%M%S");

        std::filesystem::path path = recordings_dir_ / (stamped_name.str() + ".raw");
        int suffix                 = 1;
        while (std::filesystem::exists(path)) {
            path = recordings_dir_ / (stamped_name.str() + "_" + std::to_string(suffix++) + ".raw");
        }

        return path;
    }

    void begin_recording(Metavision::Camera &camera, const std::string &base_name, const std::string &reason) {
        current_output_path_     = make_recording_path(base_name);
        events_before_recording_ = total_events_.load(std::memory_order_relaxed);
        recording_started_at_    = std::chrono::steady_clock::now();

        log(reason + " Recording " + current_output_path_.string() + "...");
        camera.start_recording(current_output_path_.string());
        recording_ = true;
    }

    void log(const std::string &message) const {
        if (log_callback_) {
            log_callback_(message);
        }
    }

    std::filesystem::path control_dir_;
    std::filesystem::path recordings_dir_;
    std::filesystem::path start_file_;
    std::filesystem::path stop_file_;
    std::filesystem::path quit_file_;
    std::function<void(const std::string &)> log_callback_;

    bool watching_  = false;
    bool recording_ = false;
    std::filesystem::path current_output_path_;
    std::uint64_t events_before_recording_ = 0;
    std::chrono::steady_clock::time_point recording_started_at_;
    std::atomic<std::uint64_t> total_events_{0};
};

} // namespace e_bts

#endif // E_BTS_SEQUENCE_RECORDING_CONTROLLER_H
