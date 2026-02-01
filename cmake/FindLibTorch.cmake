#[[
================================================================================
cmake/FindLibTorch.cmake - LibTorch Finder Module
================================================================================
PURPOSE: Locates LibTorch (PyTorch C++ distribution) on the system.

HOW TO USE:
  cmake -DLIBTORCH_PATH="C:/path/to/libtorch" ..

SETS THESE VARIABLES:
  LibTorch_FOUND        - TRUE if found
  LibTorch_INCLUDE_DIRS - Header directories
  LibTorch_LIBRARIES    - Libraries to link (.lib or .so files)
  LibTorch_LIB_DIR      - Directory containing DLLs/shared libs
================================================================================
]]

# Get path from -D flag or environment variable
if(NOT DEFINED LIBTORCH_PATH)
    if(DEFINED ENV{LIBTORCH_PATH})
        set(LIBTORCH_PATH $ENV{LIBTORCH_PATH})
    else()
        message(FATAL_ERROR "LIBTORCH_PATH not set. Use: cmake -DLIBTORCH_PATH=...")
    endif()
endif()

message(STATUS "Looking for LibTorch in: ${LIBTORCH_PATH}")

# Find headers
find_path(LibTorch_INCLUDE_DIR NAMES torch/torch.h
    PATHS ${LIBTORCH_PATH}/include ${LIBTORCH_PATH}/include/torch/csrc/api/include
    NO_DEFAULT_PATH
)

# Find library directory
find_path(LibTorch_LIBRARY_DIR NAMES torch.lib libtorch.so libtorch.dylib
    PATHS ${LIBTORCH_PATH}/lib
    NO_DEFAULT_PATH
)

if(LibTorch_INCLUDE_DIR AND LibTorch_LIBRARY_DIR)
    set(LibTorch_FOUND TRUE)
    set(LibTorch_INCLUDE_DIRS
        ${LIBTORCH_PATH}/include
        ${LIBTORCH_PATH}/include/torch/csrc/api/include
    )
    
    # Collect all libraries
    if(WIN32)
        file(GLOB LibTorch_LIBRARIES "${LibTorch_LIBRARY_DIR}/*.lib")
    else()
        file(GLOB LibTorch_LIBRARIES "${LibTorch_LIBRARY_DIR}/*.so" "${LibTorch_LIBRARY_DIR}/*.dylib")
    endif()
    
    set(LibTorch_LIB_DIR ${LibTorch_LIBRARY_DIR})
    message(STATUS "Found LibTorch: ${LIBTORCH_PATH}")
    message(STATUS "Found LibTorch LIBRARY DIR: ${LibTorch_LIBRARY_DIR}")
else()
    set(LibTorch_FOUND FALSE)
    message(FATAL_ERROR "Could not find LibTorch. Check LIBTORCH_PATH.")
endif()

mark_as_advanced(LibTorch_INCLUDE_DIR LibTorch_LIBRARY_DIR)
