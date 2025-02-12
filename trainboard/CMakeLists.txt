cmake_minimum_required(VERSION 3.16)



set(TRAINBOARD_DEBUG ON)
set(USE_FMT ON)

if(TRAINBOARD_DEBUG)
    add_definitions(-DTRAINBOARD_DEBUG)
endif()

if(USE_FMT)
    add_definitions(-DUSE_FMT)
endif()



add_subdirectory(train_layout)
add_subdirectory(train_utils)
