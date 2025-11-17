cmake_minimum_required(VERSION 3.16)

add_library(tools OBJECT 
    exiter/exiter.cpp
    extended_kalman_filter/extended_kalman_filter.cpp
    ransac_sine_fitter/ransac_sine_fitter.cpp
    img_tools/img_tools.cpp
    math_tools/math_tools.cpp
    plotter/plotter.cpp
    trajectory/trajectory.cpp
    recorder/recorder.cpp
    logger/logger.cpp
    pid/pid.cpp
    crc/crc.cpp
)