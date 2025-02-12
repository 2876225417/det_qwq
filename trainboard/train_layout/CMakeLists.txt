cmake_minimum_required(VERSION 3.16)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS
    Widgets
)

find_package(fmt CONFIG REQUIRED)

add_library(train_layout
    STATIC
    train_configuration_panel.cpp
    cuda_t.cu
)

target_link_libraries(train_layout PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(train_layout PRIVATE fmt::fmt)

