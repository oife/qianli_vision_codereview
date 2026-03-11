cmake_minimum_required(VERSION 3.16)

find_package(Eigen3 REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include
)
set(CUDA_TARGET_INCLUDE_DIR /usr/local/cuda/targets/aarch64-linux/include)
if(NOT EXISTS ${CUDA_TARGET_INCLUDE_DIR}/cuda_runtime_api.h)
    set(CUDA_TARGET_INCLUDE_DIR /usr/local/cuda-12.6/targets/aarch64-linux/include)
endif()
find_library(TENSORRT_LIBRARY nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
)
find_library(NVONNXPARSER_LIBRARY nvonnxparser
    PATHS /usr/lib/aarch64-linux-gnu
)
find_library(CUDART_LIBRARY cudart
    PATHS /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda-12.6/targets/aarch64-linux/lib
)
find_library(CUDLA_LIBRARY cudla
    PATHS /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda-12.6/targets/aarch64-linux/lib
)
find_library(NVDLA_COMPILER_LIBRARY nvdla_compiler
    PATHS /usr/lib/aarch64-linux-gnu/nvidia
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT EXISTS ${CUDA_TARGET_INCLUDE_DIR}/cuda_runtime_api.h OR NOT TENSORRT_LIBRARY OR NOT NVONNXPARSER_LIBRARY OR NOT CUDART_LIBRARY OR NOT CUDLA_LIBRARY OR NOT NVDLA_COMPILER_LIBRARY)
    message(FATAL_ERROR "TensorRT/CUDA development files not found")
endif()

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
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
    backend/tensorrt_backend.cpp
)

target_include_directories(auto_aim PUBLIC ${TENSORRT_INCLUDE_DIR} ${CUDA_TARGET_INCLUDE_DIR})
target_link_libraries(auto_aim io tinympcstatic ${TENSORRT_LIBRARY} ${NVONNXPARSER_LIBRARY} ${CUDART_LIBRARY} ${CUDLA_LIBRARY} ${NVDLA_COMPILER_LIBRARY})
