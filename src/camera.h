#ifndef E_BTS_CAMERA_H
#define E_BTS_CAMERA_H

#include <functional>

#include "camera_feed_utils.h"

#include <metavision/sdk/base/utils/timestamp.h>
#include <metavision/sdk/core/utils/cd_frame_generator.h>
#include <metavision/sdk/driver/camera.h>
#include <opencv2/core.hpp>

namespace e_bts {

// Owns the live-view rendering branch of a shared camera session: a
// CDFrameGenerator that turns decoded CD events into display frames. The
// Metavision::Camera itself is owned by the caller (CameraSessionWorker in
// the GUI, shared with CircleTrackingSource and SequenceRecordingController)
// since the EVK1 only supports one open handle at a time. Manual RAW
// recording is not this class's concern -- it goes through the same
// SequenceRecordingController the Sequence Recording pane uses, so there is
// exactly one recorder per camera session regardless of which UI element
// triggers it.
class CameraSource {
public:
    CameraSource(int width, int height, Metavision::timestamp accumulation_time_us,
                 CameraFeedMode camera_feed_mode = CameraFeedMode::Monochrome);

    // frame_callback runs on the frame generator's own internal thread
    // whenever a new display frame is ready.
    bool start(std::function<void(const cv::Mat &)> frame_callback);
    void stop();

    void connect_to_camera(Metavision::Camera &camera);
    void set_display_accumulation_time_us(Metavision::timestamp accumulation_time_us);

private:
    Metavision::CDFrameGenerator frame_generator_;
    std::function<void(const cv::Mat &)> frame_callback_;
};

} // namespace e_bts

#endif // E_BTS_CAMERA_H
