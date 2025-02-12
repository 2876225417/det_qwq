cmake_minimum_required(VERSION 3.16)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS
    Widgets
)

add_subdirectory(gpu_info)


add_library(utils
    STATIC
    test.cpp
)

target_link_libraries(utils PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
