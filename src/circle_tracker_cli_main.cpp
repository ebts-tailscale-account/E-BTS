// E_BTS_event_circle_tracker: headless-friendly marker overlay tool. Single
// window, no live camera view, no GUI dependency -- just the tracker output,
// for debugging the detector/tracker without building the full E_BTS_GUI.

#include <atomic>
#include <exception>
#include <iostream>
#include <limits>

#include "camera_utils.h"
#include "circle_tracker_config.h"
#include "circle_tracking.h"
#include "console_input_utils.h"
#include "display_utils.h"
#include "signal_utils.h"
#include "tracking_render_utils.h"
#include "window_icon_utils.h"

#include <metavision/sdk/driver/camera.h>
#include <metavision/sdk/driver/camera_exception.h>
#include <metavision/sdk/ui/utils/event_loop.h>

int main(int argc, char *argv[]) {
    e_bts::install_stop_signal_handlers();

    try {
        std::cout << "Opening Metavision camera";
        if (argc > 1) {
            std::cout << " from '" << argv[1] << "'";
        }
        std::cout << "...\n";

        Metavision::Camera camera = e_bts::open_first_available_serial_or_raw(argc, argv);
        const int width           = camera.geometry().width();
        const int height          = camera.geometry().height();
        std::cout << "Camera opened: " << width << 'x' << height << '\n';

        std::cout << "Enter accumulation time in microseconds (press Enter or enter 0 for default = "
                  << e_bts::kDefaultEventCollectionTimeUs << "): " << std::flush;
        std::uint64_t requested_collection_time_us = 0;
        if (!e_bts::read_value_or_default(static_cast<std::uint64_t>(e_bts::kDefaultEventCollectionTimeUs),
                                          requested_collection_time_us)) {
            std::cerr << "Accumulation time must be a non-negative integer.\n";
            return 1;
        }
        if (requested_collection_time_us == 0) {
            requested_collection_time_us = static_cast<std::uint64_t>(e_bts::kDefaultEventCollectionTimeUs);
        }
        const std::uint64_t maximum_collection_time_us =
            static_cast<std::uint64_t>(std::numeric_limits<Metavision::timestamp>::max()) /
            static_cast<std::uint64_t>(e_bts::kTemporalFilterWindowCount);
        if (requested_collection_time_us > maximum_collection_time_us) {
            std::cerr << "Accumulation time is too large for the temporal filter.\n";
            return 1;
        }
        const auto collection_time_us = static_cast<Metavision::timestamp>(requested_collection_time_us);

        std::cout << "Enter minimum circle pixel density as a percentage (press Enter or enter 0 for default = "
                  << e_bts::kDefaultMinimumCircleDensity * 100.0 << "): " << std::flush;
        double requested_density_percent = 0.0;
        if (!e_bts::read_value_or_default(e_bts::kDefaultMinimumCircleDensity * 100.0, requested_density_percent)) {
            std::cerr << "Circle pixel density must be a number from 0 to 100.\n";
            return 1;
        }
        if (requested_density_percent == 0.0) {
            requested_density_percent = e_bts::kDefaultMinimumCircleDensity * 100.0;
        }
        if (!std::isfinite(requested_density_percent) || requested_density_percent < 0.0 ||
            requested_density_percent > 100.0) {
            std::cerr << "Circle pixel density must be between 0 and 100 percent.\n";
            return 1;
        }
        const double minimum_circle_density = requested_density_percent / 100.0;

        e_bts::CircleTrackingSource tracker(width, height, collection_time_us, minimum_circle_density);

        std::cout << "Accumulation time: " << tracker.collection_time_us() << " us\n"
                  << "Estimated marker radius: " << tracker.expected_radius_px() << " px\n"
                  << "Minimum occupied-pixel density: " << tracker.minimum_circle_density() * 100.0 << "%\n"
                  << "Keep the sensor unloaded during startup baseline collection.\n"
                  << "Press B with no load applied to restart baseline collection.\n"
                  << "Press Escape or Q to quit.\n";

        std::atomic_bool camera_error{false};
        e_bts::set_camera_runtime_error_callback(camera, camera_error);
        tracker.connect_to_camera(camera);

        if (!camera.start()) {
            std::cerr << "Camera did not start.\n";
            return 1;
        }

        e_bts::CameraFeedWindow window("EVK1 event circles and displacement", width, height,
                                       Metavision::BaseWindow::RenderMode::BGR);
        e_bts::LetterboxedFramePresenter presenter(window);
        std::atomic_bool baseline_requested{false};
        e_bts::set_tracking_window_keyboard_callback(window, baseline_requested);

        while (camera.is_running() && !window.should_close() && !camera_error && !e_bts::stop_requested()) {
            Metavision::EventLoop::poll_and_dispatch(5);
            window.poll_events();

            if (baseline_requested.exchange(false)) {
                tracker.request_baseline_restart();
                std::cout << "Baseline collection restart requested; keep the sensor unloaded.\n";
            }

            cv::Mat output;
            if (tracker.process_next_window(output)) {
                presenter.show_async(output);
            }
        }

        camera.stop();
        return camera_error ? 1 : 0;
    } catch (const Metavision::CameraException &error) {
        std::cerr << "Metavision camera error: " << error.what() << '\n';
        e_bts::print_available_sources();
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
    }

    return 1;
}
