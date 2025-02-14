cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(det_qwq LANGUAGES CXX CUDA)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(utils)
add_subdirectory(trainboard)
# add_subdirectory(det_qwq_layout)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(FindPythonLocal)

set(ONNXRUNTIME_GPU_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/onnxruntime_gpu/include")
set(ONNXRUNTIME_GPU_LIB_DIR     "${CMAKE_SOURCE_DIR}/onnxruntime_gpu/lib")
set(ONNXRUNTIME_CPU_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/onnxruntime_gpu/include")
set(ONNXRUNTIME_CPU_LIB_DIR     "${CMAKE_SOURCE_DIR}/onnxruntime_gpu/lib")
set(CUDA_INCLUDE_DIR        "/opt/cuda/include")
set(OPENCV_INCLUDE_DIR      "/usr/include/opencv4")

# include_directories(${ONNXRUNTIME_GPU_INCLUDE_DIR})
include_directories(${ONNXRUNTIME_CPU_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${OPENCV_INCLUDE_DIR})

link_directories(
#    ${ONNXRUNTIME_GPU_LIB_DIR}
    ${ONNXRUNTIME_CPU_LIB_DIR}
    /usr/lib
)


find_package(OpenGL REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS
    Core
    Gui
    Widgets
    Sql
    OpenGL
    OpenGLWidgets
    SerialPort
    WebSockets
)

find_package(OpenCV REQUIRED)
if(OpenCV_FOUND AND OpenCV_CUDA_FOUND)
    add_definitions(-DHAVE_OPENCV_CUDA)
endif()

# add_definitions(-DENABLE_SR)

find_package(Boost 1.87.0 REQUIRED COMPONENTS system filesystem)

if(NOT Python_EXECUTABLE)
    find_package(Python REQUIRED COMPONENTS Interpreter)
endif()

file(COPY
    ${CMAKE_SOURCE_DIR}/pyscripts
    DESTINATION ${CMAKE_BINARY_DIR}
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wuninitialized
    -Wmaybe-uninitialized
)

add_executable(det_qwq
    main.cpp
    mainwindow.h mainwindow.cpp
    configuration.h configuration.cpp
    customstatusbar.h customstatusbar.cpp
    dashboard.h dashboard.cpp
    database.h database.cpp 
    dbconn.h dbconn.cpp
    detectionboard.h detectionboard.cpp
    onnxruntime_inference_session.h onnxruntime_inference_session.cpp
    gpuinfo.h gpuinfo.cpp
    serial_port_configuration_board.h serial_port_configuration_board.cpp
    sidebar.h sidebar.cpp
    trainboard.h trainboard.cpp
    yolotrainer.h yolotrainer.cpp
    iteminfo.h iteminfo.cpp 
    # algorithm.cu
    denoise_cuda.h denoise_kernels.cu   
    sr_cuda.h sr_kernels.cu
    
    resources.qrc
)


option(ENABLE_VENV "Create isolated Python venv for YOLO training" OFF)
set(PIP_INDEX_URL "https://pypi.tuna.tsinghua.edu.cn/simple" CACHE STRING "PyPI mirror URL")
set(HTTP_PROXY "" CACHE STRING "HTTP proxy configuration")


if (ENABLE_VENV AND NOT EXISTS ${VENV_MARKER})
    set(YOLO_VENV_DIR "${CMAKE_BINARY_DIR}/venv")
    set(VENV_MARKER "${YOLO_VENV_DIR}/.venv_ready")

    add_definitions(-DENABLE_VENV)

    if(WIN32)
        set(PYTHON_VENV_EXE "${YOLO_VENV_DIR}/Scripts/python.exe")
        set(PIP_EXE "${YOLO_VENV_DIR}/Scripts/pip.exe")
        set(VENV_MARKER "${YOLO_VENV_DIR}/Scripts/.venv_ready")
    else()
        set(PYTHON_VENV_EXE "${YOLO_VENV_DIR}/bin/python3")
        set(PIP_EXE "${YOLO_VENV_DIR}/bin/pip")
        set(VENV_MARKER "${YOLO_VENV_DIR}/.venv_ready")
    endif()

    set(PROXY_SETUP "")
    if(HTTP_PROXY)
        if(WIN32)
            set(PROXY_SETUP "set http_proxy=${HTTP_PROXY} && set https_proxy=${HTTP_PROXY} && ")
        else()
            set(PROXY_SETUP "export http_proxy=${HTTP_PROXY} && export https_proxy=${HTTP_PROXY} && ")
        endif()
    endif()

    if(NOT EXISTS ${VENV_MARKER})
        add_custom_target(
            create_venv
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${YOLO_VENV_DIR}"
            COMMAND ${Python_EXECUTABLE} -m venv "${YOLO_VENV_DIR}"
            COMMAND ${PROXY_SETUP} "${PIP_EXE}" install --upgrade pip
                -i "${PIP_INDEX_URL}"
                --trusted-host=pypi.tuna.tsinghua.edu.cn
            COMMAND ${PROXY_SETUP} "${PIP_EXE}" install ultralytics
                -i "${PIP_INDEX_URL}"
                --trusted-host=pypi.tuna.tsinghua.edu.cn
            COMMAND ${CMAKE_COMMAND} -E touch "${VENV_MARKER}"
            COMMENT "Creating venv with mirror ${PIP_INDEX_URL}" 
        )
    else()
        add_custom_target(create_venv
            COMMAND ${CMAKE_COMMAND} -E echo "Virtual environment already exists"
            COMMENT "Skipping venv creating (marker exists)"
        )
    endif()

    add_dependencies(det_qwq create_venv)

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/include")
    configure_file(
        ${CMAKE_SOURCE_DIR}/config.h.in
        ${CMAKE_BINARY_DIR}/include/config.h
        @ONLY
    )
    include_directories(${CMAKE_BINARY_DIR}/include)

    if(MSVC)
        set_target_properties(det_qwq PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        )
    endif()
endif()



target_link_libraries(det_qwq 
    PRIVATE 
    Boost::filesystem
    onnxruntime
    Qt6::Widgets
    Qt6::Sql
    Qt6::Core
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
    Qt6::Gui
    Qt6::SerialPort
    Qt6::WebSockets
    ${OpenCV_LIBS}
    nvidia-ml
    cudart
)

target_link_libraries(det_qwq PRIVATE gpu_info train_layout)

