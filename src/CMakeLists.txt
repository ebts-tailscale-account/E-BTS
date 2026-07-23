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
find_package(Qt5 COMPONENTS Quick QuickControls2 Gui REQUIRED)

# Headers pulled in by circle_tracking.cpp/circle_tracker_cli_main.cpp; not
# compiled on their own, but listed as sources so IDEs index them next to
# the .cpp files that use them.
set(E_BTS_CIRCLE_TRACKER_HEADERS
    circle_detector.h
    circle_lattice.h
    circle_tracker_config.h
    circle_tracking.h
    console_input_utils.h
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

add_executable(E_BTS_GUI
    combined_main.cpp
    camera.cpp
    camera.h
    circle_tracking.cpp
    circle_tracking.h
    sequence_recording_controller.h
    gui/camera_session_worker.cpp
    gui/camera_session_worker.h
    gui/export_bridge.h
    gui/export_worker.cpp
    gui/export_worker.h
    gui/frame_view.cpp
    gui/frame_view.h
    gui/gui_bridge.h
    gui/log_stream.h
    qml.qrc
)
target_link_libraries(E_BTS_GUI PRIVATE
    Qt5::Quick
    Qt5::QuickControls2
    Qt5::Gui
    MetavisionSDK::core
    MetavisionSDK::driver
    MetavisionSDK::ui
    ${OpenCV_LIBS}
)
# Scoped to this target only: moc for CameraSessionWorker/FrameView's
# Q_OBJECT macros, rcc for qml.qrc. The other targets above have neither.
set_target_properties(E_BTS_GUI PROPERTIES AUTOMOC ON AUTORCC ON)
