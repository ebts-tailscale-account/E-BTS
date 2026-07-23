#ifndef E_BTS_EVENT_WINDOW_BUFFER_H
#define E_BTS_EVENT_WINDOW_BUFFER_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

#include "circle_tracker_config.h"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/base/utils/timestamp.h>
#include <opencv2/core.hpp>

namespace e_bts {

struct EventWindow {
    Metavision::timestamp start_us = 0;
    Metavision::timestamp end_us   = 0;
    std::vector<std::uint32_t> occupied_pixel_indices;
};

// Accumulates direct CD events into non-overlapping timestamp windows. A pixel
// is binary: receiving multiple events during one window does not increase its
// contribution to the density calculation.
class EventWindowBuffer {
public:
    EventWindowBuffer(int width, int height, Metavision::timestamp collection_time_us) :
        width_(width), height_(height), collection_time_us_(collection_time_us),
        pixel_window_ids_(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), 0) {}

    void add_events(const Metavision::EventCD *begin, const Metavision::EventCD *end) {
        std::lock_guard<std::mutex> lock(mutex_);

        for (const Metavision::EventCD *event = begin; event != end; ++event) {
            if (!window_started_) {
                window_started_  = true;
                window_start_us_ = event->t;
                window_end_us_   = window_start_us_ + collection_time_us_;
            }

            if (event->t >= window_end_us_) {
                queue_current_window();
                advance_pixel_window_id();
                const Metavision::timestamp elapsed_windows =
                    (event->t - window_start_us_) / collection_time_us_;
                window_start_us_ += elapsed_windows * collection_time_us_;
                window_end_us_ = window_start_us_ + collection_time_us_;
            }

            if (event->x < width_ && event->y < height_) {
                const std::uint32_t pixel_index =
                    static_cast<std::uint32_t>(event->y * width_ + event->x);
                if (pixel_window_ids_[pixel_index] != current_pixel_window_id_) {
                    pixel_window_ids_[pixel_index] = current_pixel_window_id_;
                    current_occupied_pixel_indices_.push_back(pixel_index);
                }
            }
        }
    }

    // Detection cannot run at 10,000-100,000 frames per second for very short
    // collection intervals. Consume the newest completed window so capture and
    // the UI stay responsive, and report superseded windows as dropped.
    bool pop_latest(EventWindow &event_window) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ready_windows_.empty()) {
            return false;
        }

        dropped_window_count_ += ready_windows_.size() - 1;
        event_window = std::move(ready_windows_.back());
        ready_windows_.clear();
        return true;
    }

    std::uint64_t dropped_window_count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return dropped_window_count_;
    }

    // Changes the collection window length for windows started from this point
    // on; a window already being accumulated keeps using the length it started
    // with. Shares add_events'/pop_latest's mutex since the camera callback
    // thread reads collection_time_us_ concurrently with this main-thread call
    // (e.g. from the settings overlay slider).
    void set_collection_time_us(Metavision::timestamp collection_time_us) {
        std::lock_guard<std::mutex> lock(mutex_);
        collection_time_us_ = std::clamp(collection_time_us, kMinimumCollectionTimeUs, kMaximumCollectionTimeUs);
    }

    Metavision::timestamp collection_time_us() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return collection_time_us_;
    }

private:
    void queue_current_window() {
        if (current_occupied_pixel_indices_.empty()) {
            return;
        }
        if (ready_windows_.size() == kMaximumQueuedWindows) {
            ready_windows_.pop_front();
            ++dropped_window_count_;
        }

        ready_windows_.push_back(
            EventWindow{window_start_us_, window_end_us_, std::move(current_occupied_pixel_indices_)});
        current_occupied_pixel_indices_.clear();
    }

    void advance_pixel_window_id() {
        ++current_pixel_window_id_;
        if (current_pixel_window_id_ == 0) {
            std::fill(pixel_window_ids_.begin(), pixel_window_ids_.end(), 0);
            current_pixel_window_id_ = 1;
        }
    }

    int width_;
    int height_;
    Metavision::timestamp collection_time_us_;
    mutable std::mutex mutex_;
    bool window_started_ = false;
    Metavision::timestamp window_start_us_ = 0;
    Metavision::timestamp window_end_us_   = 0;
    std::vector<std::uint32_t> pixel_window_ids_;
    std::uint32_t current_pixel_window_id_ = 1;
    std::vector<std::uint32_t> current_occupied_pixel_indices_;
    std::deque<EventWindow> ready_windows_;
    std::uint64_t dropped_window_count_ = 0;
};

inline cv::Mat make_occupied_pixel_frame(const EventWindow &event_window, int width, int height) {
    cv::Mat occupied_pixels = cv::Mat::zeros(height, width, CV_8UC1);
    auto *pixels = occupied_pixels.ptr<std::uint8_t>(0);
    for (const std::uint32_t pixel_index : event_window.occupied_pixel_indices) {
        pixels[pixel_index] = 255;
    }
    return occupied_pixels;
}

} // namespace e_bts

#endif // E_BTS_EVENT_WINDOW_BUFFER_H
