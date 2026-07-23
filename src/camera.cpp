#include "camera.h"

#include "camera_utils.h"
#include "circle_tracker_config.h"

namespace e_bts {

CameraSource::CameraSource(int width, int height, Metavision::timestamp accumulation_time_us,
                           CameraFeedMode camera_feed_mode) :
    frame_generator_(width, height) {
    frame_generator_.set_display_accumulation_time_us(accumulation_time_us);
    apply_camera_feed_mode(frame_generator_, camera_feed_mode);
}

bool CameraSource::start(std::function<void(const cv::Mat &)> frame_callback) {
    frame_callback_ = std::move(frame_callback);
    return frame_generator_.start(kLiveDisplayFps, [this](Metavision::timestamp, cv::Mat &frame) {
        if (frame_callback_) {
            frame_callback_(frame);
        }
    });
}

void CameraSource::stop() {
    frame_generator_.stop();
}

void CameraSource::connect_to_camera(Metavision::Camera &camera) {
    connect_cd_events_to_frame_generator(camera, frame_generator_);
}

void CameraSource::set_display_accumulation_time_us(Metavision::timestamp accumulation_time_us) {
    frame_generator_.set_display_accumulation_time_us(accumulation_time_us);
}

} // namespace e_bts
