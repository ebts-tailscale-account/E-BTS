#ifndef E_BTS_WINDOW_ICON_UTILS_H
#define E_BTS_WINDOW_ICON_UTILS_H

#include <array>
#include <filesystem>
#include <iostream>

#include <metavision/sdk/ui/utils/mt_window.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace e_bts {

inline std::filesystem::path find_camera_feed_icon_path() {
    const std::filesystem::path header_path = std::filesystem::path(__FILE__).parent_path() / "icon.png";
    const std::array<std::filesystem::path, 4> candidates{
        header_path,
        std::filesystem::current_path() / "icon.png",
        std::filesystem::current_path() / "src" / "icon.png",
        std::filesystem::current_path() / ".." / "src" / "icon.png",
    };

    for (const auto &candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return header_path;
}

inline cv::Mat load_glfw_icon_pixels(const std::filesystem::path &icon_path) {
    const cv::Mat icon = cv::imread(icon_path.string(), cv::IMREAD_UNCHANGED);
    if (icon.empty()) {
        return {};
    }

    cv::Mat rgba_icon;
    if (icon.channels() == 4) {
        cv::cvtColor(icon, rgba_icon, cv::COLOR_BGRA2RGBA);
    } else if (icon.channels() == 3) {
        cv::cvtColor(icon, rgba_icon, cv::COLOR_BGR2RGBA);
    } else if (icon.channels() == 1) {
        cv::cvtColor(icon, rgba_icon, cv::COLOR_GRAY2RGBA);
    }

    if (!rgba_icon.isContinuous()) {
        rgba_icon = rgba_icon.clone();
    }

    return rgba_icon;
}

class CameraFeedWindow : public Metavision::MTWindow {
public:
    CameraFeedWindow(const std::string &title, int width, int height, RenderMode mode) :
        Metavision::MTWindow(title, width, height, mode) {
        set_icon(find_camera_feed_icon_path());
    }

private:
    void set_icon(const std::filesystem::path &icon_path) {
        cv::Mat icon = load_glfw_icon_pixels(icon_path);
        if (icon.empty()) {
            std::cerr << "Could not load window icon from " << icon_path << '\n';
            return;
        }

        GLFWimage image{};
        image.width  = icon.cols;
        image.height = icon.rows;
        image.pixels = icon.data;

        glfwSetWindowIcon(glfwWindow_, 1, &image);
    }
};

} // namespace e_bts

#endif // E_BTS_WINDOW_ICON_UTILS_H
