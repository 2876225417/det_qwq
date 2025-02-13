cmake_minimum_required(VERSION 3.16)

set(CUDA_INCLUDE_DIR        "/opt/cuda/include")

include_directories(${CUDA_INCLUDE_DIR})
link_directories(/usr/lib)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS
    Widgets
    OpenGL
    OpenGLWidgets
    Charts
    Sql
)
find_package(OpenGL REQUIRED)

add_library(gpu_info
    STATIC
    gpu_performance_monitor.cpp
)

target_link_libraries(gpu_info PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(gpu_info PRIVATE Qt${QT_VERSION_MAJOR}::OpenGL)
target_link_libraries(gpu_info PRIVATE Qt${QT_VERSION_MAJOR}::OpenGLWidgets)
target_link_libraries(gpu_info PRIVATE Qt${QT_VERSION_MAJOR}::Gui)
target_link_libraries(gpu_info PRIVATE Qt${QT_VERSION_MAJOR}::Charts)
target_link_libraries(gpu_info PRIVATE Qt${QT_VERSION_MAJOR}::Sql)


target_link_libraries(gpu_info PRIVATE OpenGL::GL nvidia-ml)
