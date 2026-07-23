#ifndef E_BTS_DISPLAY_UTILS_H
#define E_BTS_DISPLAY_UTILS_H

#include <algorithm>

#include <metavision/sdk/ui/utils/mt_window.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace e_bts {

inline cv::Rect centered_four_by_three_rect(int window_width, int window_height) {
    constexpr int kAspectWidth  = 4;
    constexpr int kAspectHeight = 3;

    int output_width  = window_width;
    int output_height = output_width * kAspectHeight / kAspectWidth;
    if (output_height > window_height) {
        output_height = window_height;
        output_width  = output_height * kAspectWidth / kAspectHeight;
    }

    return cv::Rect((window_width - output_width) / 2, (window_height - output_height) / 2, output_width,
                    output_height);
}

inline void render_letterboxed_display_frame(const cv::Mat &frame, cv::Mat &display, int window_width,
                                             int window_height) {
    window_width  = std::max(1, window_width);
    window_height = std::max(1, window_height);

    display.create(window_height, window_width, frame.type());
    display.setTo(cv::Scalar::all(0));

    const cv::Rect output_rect = centered_four_by_three_rect(window_width, window_height);
    if (output_rect.width > 0 && output_rect.height > 0) {
        cv::resize(frame, display(output_rect), output_rect.size(), 0, 0, cv::INTER_NEAREST);
        cv::rectangle(display, output_rect, cv::Scalar(0, 255, 0), 2);
    }
}

class LetterboxedFramePresenter {
public:
    explicit LetterboxedFramePresenter(Metavision::MTWindow &window) : window_(window) {}

    void show_async(const cv::Mat &frame) {
        int window_width  = 0;
        int window_height = 0;
        window_.get_size(window_width, window_height);

        render_letterboxed_display_frame(frame, display_frame_, window_width, window_height);
        window_.show_async(display_frame_, false);
    }

private:
    Metavision::MTWindow &window_;
    cv::Mat display_frame_;
};

} // namespace e_bts

#endif // E_BTS_DISPLAY_UTILS_H
