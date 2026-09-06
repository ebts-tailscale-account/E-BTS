cmake_minimum_required(VERSION 3.16)
project(E_BTS LANGUAGES CXX)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MetavisionSDK COMPONENTS core driver ui REQUIRED)
find_package(OpenCV COMPONENTS core imgcodecs imgproc videoio REQUIRED)
find_package(Qt5 COMPONENTS Quick QuickControls2 Gui Network REQUIRED)

# Headers pulled in by circle_tracking.cpp/circle_tracker_cli_main.cpp; not
# compiled on their own, but listed as sources so IDEs index them next to
# the .cpp files that use them.
set(E_BTS_CIRCLE_TRACKER_HEADERS
    circle_detector.h
    circle_lattice.h
    circle_tracker_config.h
    circle_tracking.h
    console_input_utils.h
    contact_localiser.h
    event_window_buffer.h
    temporal_circle_tracker.h
    tracking_render_utils.h
)

add_executable(E_BTS_event_circle_tracker
    circle_tracker_cli_main.cpp
    circle_tracking.cpp
    display_utils.h
    window_icon_utils.h
    ${E_BTS_CIRCLE_TRACKER_HEADERS}
)
target_link_libraries(E_BTS_event_circle_tracker PRIVATE
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::ui
    ${OpenCV_LIBS}
)

add_executable(E_BTS_record_sequence record_sequence.cpp sequence_recording_controller.h path_utils.h)
target_link_libraries(E_BTS_record_sequence PRIVATE
    MetavisionSDK::driver
)

add_executable(E_BTS_raw_to_csv raw_to_csv.cpp camera_utils.h path_utils.h)
target_link_libraries(E_BTS_raw_to_csv PRIVATE
    MetavisionSDK::driver
)

# Offline spatter tracking: .raw -> per-window per-track CSV. Links OpenCV only
# for cv::Point2f/cv::Rect in the tracker's types; it draws nothing.
add_executable(E_BTS_spatter_track_csv
    spatter_track_csv.cpp
    spatter_tracker.h
    spatter_tracker_config.h
    spatter_render.h
    event_window_buffer.h
)
target_link_libraries(E_BTS_spatter_track_csv PRIVATE
    MetavisionSDK::driver
    ${OpenCV_LIBS}
)

# Verification harness for the contact estimator's positional gain; see the file
# header. No Qt, no camera -- synthetic frames through the real detector/tracker.
add_executable(E_BTS_marker_travel_test marker_travel_test.cpp contact_localiser.h
               temporal_circle_tracker.h circle_detector.h)
target_link_libraries(E_BTS_marker_travel_test PRIVATE MetavisionSDK::base ${OpenCV_LIBS})

# Verification harness for src/pixel_to_mm.h; see ml/check_pixel_to_mm_port.py.
# Qt5::Core only -- it needs QJsonDocument and nothing else from Qt.
add_executable(E_BTS_pixel_to_mm_probe pixel_to_mm_probe.cpp pixel_to_mm.h)
target_link_libraries(E_BTS_pixel_to_mm_probe PRIVATE Qt5::Core opencv_core)

# Offline replay of the LIVE contact estimator over a recorded .raw; see the file
# header. Shares circle_tracking.cpp with the GUI on purpose -- the point is to
# measure the same code, so a second copy of the estimator loop would defeat it.
# Qt5::Core for pixel_to_mm.h's JSON loader, as E_BTS_pixel_to_mm_probe does.
add_executable(E_BTS_contact_replay
    contact_replay.cpp
    circle_tracking.cpp
    pixel_to_mm.h
    ${E_BTS_CIRCLE_TRACKER_HEADERS}
)
target_link_libraries(E_BTS_contact_replay PRIVATE
    MetavisionSDK::core
    MetavisionSDK::driver
    Qt5::Core
    ${OpenCV_LIBS}
)

add_executable(E_BTS_GUI
    combined_main.cpp
    camera.cpp
    camera.h
    circle_tracking.cpp
    circle_tracking.h
    spatter_tracking.cpp
    spatter_tracking.h
    spatter_tracker.h
    spatter_tracker_config.h
    sequence_recording_controller.h
    contact_localiser.h
    pixel_to_mm.h
    gui/camera_session_worker.cpp
    gui/camera_session_worker.h
    gui/export_bridge.h
    gui/export_worker.cpp
    gui/export_worker.h
    gui/franka_pose_client.h
    gui/frame_view.cpp
    gui/frame_view.h
    gui/ft_graph_view.cpp
    gui/ft_graph_view.h
    gui/gui_bridge.h
    gui/log_stream.h
    gui/wittenstein_bridge.h
    gui/wittenstein_worker.cpp
    gui/wittenstein_worker.h
    qml.qrc
)
target_link_libraries(E_BTS_GUI PRIVATE
    Qt5::Quick
    Qt5::QuickControls2
    Qt5::Gui
    # QtNetwork is only for gui/franka_pose_client.h, which polls
    # franka/pose_server.py on tactile for the robot's live EE pose.
    Qt5::Network
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::ui
    ${OpenCV_LIBS}
)
# Scoped to this target only: moc for CameraSessionWorker/FrameView's
# Q_OBJECT macros, rcc for qml.qrc. The other targets above have neither.
set_target_properties(E_BTS_GUI PROPERTIES AUTOMOC ON AUTORCC ON)
