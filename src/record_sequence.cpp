// ============================================================================
// record_sequence.cpp
//
// Thin CLI wrapper around SequenceRecordingController (see
// sequence_recording_controller.h for the full start.cmd/stop.cmd/quit.cmd
// protocol and a Python-side example). Preserved as a standalone executable
// so existing robot-control scripts and docker/record_sequence.sh keep
// working unchanged; E_BTS_GUI's Sequence Recording pane drives the same
// controller class against its shared camera session instead of a second
// camera handle.
// ============================================================================

#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <thread>

#include "camera_utils.h"
#include "sequence_recording_controller.h"
#include "signal_utils.h"

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/camera_exception.h>

namespace {

constexpr auto kPollInterval = std::chrono::milliseconds(50);

} // namespace

int main(int argc, char *argv[]) {
    e_bts::install_stop_signal_handlers();

    try {
        std::cout << "Opening EVK1";
        if (argc > 1) {
            std::cout << " with serial '" << argv[1] << "'";
        }
        std::cout << "...\n";

        Metavision::Camera camera = e_bts::open_first_available_or_serial(argc, argv);
        std::cout << "Camera opened: " << camera.geometry().width() << "x" << camera.geometry().height() << '\n';

        std::atomic_bool camera_error{false};
        e_bts::set_camera_runtime_error_callback(camera, camera_error);

        e_bts::SequenceRecordingController controller;
        controller.set_log_callback([](const std::string &message) { std::cout << message << '\n'; });
        controller.connect_to_camera(camera);

        if (!camera.start()) {
            std::cerr << "Camera did not start.\n";
            return 1;
        }

        controller.start_watching();

        bool quit_received = false;
        while (!e_bts::stop_requested() && !camera_error && !quit_received) {
            quit_received = controller.poll(camera);
            std::this_thread::sleep_for(kPollInterval);
        }

        if (controller.is_recording()) {
            std::cout << "Interrupted mid-recording; stopping and saving...\n";
            controller.stop_recording_if_active(camera);
        }
        camera.stop();

        return camera_error ? 1 : 0;
    } catch (const Metavision::CameraException &error) {
        std::cerr << "Metavision camera error: " << error.what() << '\n';
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
    }

    return 1;
}
