cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(Eigen3 REQUIRED)
find_package(OpenVINO QUIET COMPONENTS Runtime)
find_package(ONNXRuntime QUIET)
find_package(CUDAToolkit QUIET COMPONENTS cudart cublas)
find_package(TensorRT QUIET)

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

if(NOT TensorRT_FOUND)
    set(TENSORRT_ROOT "/usr/local/TensorRT-8.6.1.6" CACHE PATH "Path to TensorRT installation")

    find_path(TENSORRT_INCLUDE_DIR
        NAMES NvInfer.h NvOnnxParser.h
        PATHS ${TENSORRT_ROOT}/include
            /usr/include
            /usr/local/include
    )

    find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    PATHS ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
        /usr/lib
        /usr/local/lib
    )

    find_library(TENSORRT_NVONNXPARSER_LIB
        NAMES nvonnxparser
        PATHS ${TENSORRT_ROOT}/lib
            ${TENSORRT_ROOT}/lib64
            /usr/lib
            /usr/local/lib
    )

    find_library(TENSORRT_NVINFER_PLUGIN_LIB
        NAMES nvinfer_plugin
        PATHS ${TENSORRT_ROOT}/lib
            ${TENSORRT_ROOT}/lib64
            /usr/lib
            /usr/local/lib
    )

    if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB)
        set(TensorRT_FOUND TRUE)
        set(TENSORRT_INCLUDE_DIRS ${TENSORRT_INCLUDE_DIR})
        set(TENSORRT_LIBRARIES 
            ${TENSORRT_NVINFER_LIB}
            ${TENSORRT_NVONNXPARSER_LIB}
        )
        if(TENSORRT_NVINFER_PLUGIN_LIB)
            list(APPEND TENSORRT_LIBRARIES ${TENSORRT_NVINFER_PLUGIN_LIB})
        endif()
    endif()
endif()

set(BACKEND_SOURCES
    backend.cpp
)

if(OpenVINO_FOUND)
    list(APPEND BACKEND_SOURCES openvino_backend.cpp)
    add_definitions(-DOPENVINO_AVAILABLE)
endif()

if(ONNXRuntime_FOUND)
    list(APPEND BACKEND_SOURCES ort_backend.cpp)
    add_definitions(-DORT_AVAILABLE)
endif()

if (CUDAToolkit_FOUND AND TensorRT_FOUND)
    list(APPEND BACKEND_SOURCES tensorrt_backend.cpp)
    add_definitions(-DTENSORRT_AVAILABLE)
endif()

add_library(backend STATIC 
    ${BACKEND_SOURCES}
)

set(BACKEND_FOUND FALSE)

if(OpenVINO_FOUND)
    message(STATUS "OpenVINO found")
    target_link_libraries(backend openvino::runtime)
    set(BACKEND_FOUND TRUE)
else()
    message(STATUS "OpenVINO not found")
endif()

if(ONNXRuntime_FOUND)
    message(STATUS "ONNX Runtime found")
    target_include_directories(backend PUBLIC ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(backend ${ONNXRUNTIME_LIBRARIES})
    set(BACKEND_FOUND TRUE)
else()
    message(STATUS "ONNX Runtime not found")
endif()

if (CUDAToolkit_FOUND AND TensorRT_FOUND)
    message(STATUS "TensorRT found")
    target_include_directories(backend PUBLIC ${TENSORRT_INCLUDE_DIRS})
    target_link_libraries(backend
        ${TENSORRT_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
    )
    set(BACKEND_FOUND TRUE)
else()
    message(STATUS "TensorRT not found")
endif()

if(NOT BACKEND_FOUND)
    message(FATAL_ERROR "Backend not found")
endif()