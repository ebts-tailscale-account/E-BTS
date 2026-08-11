cmake_minimum_required(VERSION 3.16)
project(bias_tuner LANGUAGES CXX)

# Standalone one-off calibration tool. Completely separate from the main E-BTS
# build -- has its own project()/targets and does not reference ../../src.

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)

find_package(MetavisionSDK COMPONENTS core driver REQUIRED)
find_package(OpenCV COMPONENTS core imgproc REQUIRED)
find_package(Qt5 COMPONENTS Widgets Gui REQUIRED)

add_executable(bias_tuner bias_tuner.cpp)
target_link_libraries(bias_tuner PRIVATE
    Qt5::Widgets
    Qt5::Gui
    MetavisionSDK::core
    MetavisionSDK::driver
    ${OpenCV_LIBS}
)
