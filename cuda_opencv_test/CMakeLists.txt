cmake_minimum_required(VERSION 3.16)
project(CudaVsCpuDemo LANGUAGES CXX CUDA)

set(CMAKE_AUTOMOC ON)
# set(CMAKE_CXX_STANDARD 20)
 
find_package(Qt6 COMPONENTS Widgets REQUIRED)
find_package(OpenCV REQUIRED)
set(CUDAToolKit_ROOT "/opt/cuda")
find_package(CUDA REQUIRED)


set(ONNXRUNTIME_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/onnxruntime_gpu/include")
set(ONNXRUNTIME_LIB_DIR     "${CMAKE_SOURCE_DIR}/onnxruntime_gpu/lib")
set(CUDA_INCLUDE_DIR        "/opt/cuda/include")
set(OPENCV_INCLUDE_DIR      "/usr/include/opencv4")

include_directories(${CUDA_INCLUDE_DIR})
include_directories(${OPENCV_INCLUDE_DIR})

link_directories(/usr/lib)
link_directories(/usr/cuda/lib64)

add_executable(${PROJECT_NAME}
    main.cpp
    mainwindow.cpp
    cuda_processor.cu
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE 
    Qt6::Widgets
    ${OpenCV_LIBS}
    cudart
)