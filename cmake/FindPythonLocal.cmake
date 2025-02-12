
# cmake/FindPythonLocal.cmake


function(download_pyton)
    set(PY_VERSION "3.13.1")
    set(PY_DIR "${CMAKE_BINARY_DIR}/python")
    
    if(NOT EXISTS "${PY_DIR}/.success")
        if(WIN32)
            set(PY_URL "https://www.python.org/ftp/python/${PY_VERSION}/python-${PY_VERSION}-embed-amd64.zip")
        else()
            set(PY_URL "")
        endif()

        file(DOWNLOAD
            ${PY_URL}
            "${CMAKE_BINARY_DIR}/python.zip"
            SHOW_PROGRESS
            TLS_VERIFY ON
        )

        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${CMAKE_BINARY_DIR}/python.zip"
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        )

        file(WRITE "${PY_DIR}/.success" "Download completed")
    endif()

    if(WIN32)
        set(Python_EXECUTABLE "${PY_DIR}/python.exe" PARENT_SCOPE)
    else()
        set(Python_EXECUTABLE "${PY_DIR}/bin/python3" PARENT_SCOPE)
    endif()
endfunction()

if (NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter)
    if (NOT Python_FOUND)
        message(STATUS "System Python not found, downloading portable version.")
        download_pyton()
        set(Python_REQUIRED_AUTO_DOWNLOAD TRUE CACHE BOOL "Enable Python auto-download")
    endif()
endif()