

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(single_opencv)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON) 


find_package(OpenCV REQUIRED)

find_package(OpenGL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Sql
    OpenGL
    OpenGLWidgets
    SerialPort
)
add_executable(single_opencv
    main.cpp
)


target_link_libraries(single_opencv PRIVATE
    Qt6::Widgets
    Qt6::Sql
    Qt6::Core
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
    Qt6::Gui
    Qt6::SerialPort
    ${OpenCV_LIBS}
)