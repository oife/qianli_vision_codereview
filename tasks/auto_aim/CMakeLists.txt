cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(Eigen3 REQUIRED)
find_package(OpenVINO QUIET COMPONENTS Runtime)
find_package(ONNXRuntime QUIET)

if(NOT ONNXRuntime_FOUND)
    set(ONNXRUNTIME_ROOT "/opt/onnxruntime/onnxruntime-linux-x64-gpu-1.20.0" CACHE PATH "Path to ONNX Runtime installation")
    
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS ${ONNXRUNTIME_ROOT}/include
              /usr/include
              /usr/local/include
    )
    
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS ${ONNXRUNTIME_ROOT}/lib
              /usr/lib
              /usr/local/lib
    )
    
    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        set(ONNXRuntime_FOUND TRUE)
        set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
        set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})
    endif()
endif()

add_subdirectory(planner/tinympc)

set(AUTO_AIM_SOURCES
    armor/armor.cpp
    classifier/classifier.cpp 
    detector/detector.cpp
    solver/solver.cpp
    aimer/aimer.cpp
    target/target.cpp
    tracker/tracker.cpp
    voter/voter.cpp
    shooter/shooter.cpp
    yolos/yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    yolos/yolo26n.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
    planner/quintic_polynomial/quintic_polynomial.cpp
    backend/backend.cpp
)

if(OpenVINO_FOUND)
    list(APPEND AUTO_AIM_SOURCES backend/openvino_backend.cpp)
    add_definitions(-DOPENVINO_AVAILABLE)
endif()

if(ONNXRuntime_FOUND)
    list(APPEND AUTO_AIM_SOURCES backend/ort_backend.cpp)
    add_definitions(-DORT_AVAILABLE)
endif()

add_library(auto_aim OBJECT 
    ${AUTO_AIM_SOURCES}
)

set(BACKEND_FOUND FALSE)

if(OpenVINO_FOUND)
    message(STATUS "OpenVINO found")
    target_link_libraries(auto_aim openvino::runtime)
    set(BACKEND_FOUND TRUE)
else()
    message(STATUS "OpenVINO not found")
endif()

if(ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime found")
    include_directories(${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(auto_aim ${ONNXRUNTIME_LIBRARIES})
    set(BACKEND_FOUND TRUE)
else()
    message(STATUS "ONNX Runtime not found")
endif()

if(NOT BACKEND_FOUND)
    message(FATAL_ERROR "Backend not found")
endif()

target_link_libraries(auto_aim io tinympcstatic)