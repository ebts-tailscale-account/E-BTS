#ifndef E_BTS_CAMERA_FEED_UTILS_H
#define E_BTS_CAMERA_FEED_UTILS_H

#include <iostream>
#include <sstream>
#include <string>

#include <metavision/sdk/core/algorithms/base_frame_generation_algorithm.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <opencv2/core.hpp>

namespace e_bts {

enum class CameraFeedMode {
    Default = 1,
    Monochrome = 2,
};

inline CameraFeedMode choose_camera_feed_mode(CameraFeedMode default_mode = CameraFeedMode::Default) {
    const int default_option = static_cast<int>(default_mode);

    while (true) {
        std::cout << "Choose camera feed mode:\n"
                  << "  1) Default polarity colors\n"
                  << "  2) Black/white events\n"
                  << "Enter 1 or 2 (press Enter for default = " << default_option << "): " << std::flush;

        std::string choice;
        if (!std::getline(std::cin, choice)) {
            return default_mode;
        }

        std::istringstream choice_stream(choice);
        choice_stream >> std::ws;
        if (choice_stream.eof()) {
            return default_mode;
        }

        int selected_option = 0;
        if (!(choice_stream >> selected_option)) {
            std::cout << "Invalid option. Try again.\n";
            continue;
        }
        choice_stream >> std::ws;
        if (!choice_stream.eof()) {
            std::cout << "Invalid option. Try again.\n";
            continue;
        }

        if (selected_option == 1) {
            return CameraFeedMode::Default;
        }
        if (selected_option == 2) {
            return CameraFeedMode::Monochrome;
        }

        std::cout << "Invalid option. Try again.\n";
    }
}

inline void apply_camera_feed_mode(Metavision::CDFrameGenerator &frame_generator, CameraFeedMode mode) {
    if (mode == CameraFeedMode::Monochrome) {
        frame_generator.set_colors(cv::Scalar::all(0), cv::Scalar::all(255), cv::Scalar::all(255), true);
        return;
    }

    frame_generator.set_color_palette(Metavision::BaseFrameGenerationAlgorithm::default_palette());
}

} // namespace e_bts

#endif // E_BTS_CAMERA_FEED_UTILS_H
