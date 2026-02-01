cmake_minimum_required(VERSION 3.18)
project(ABQnn 
    VERSION 1.0.0
    DESCRIPTION "PyTorch Neural Network integration with Abaqus UMAT"
    LANGUAGES CXX
)

#[[
================================================================================
CMakeLists.txt (Root) - Project Configuration
================================================================================
PURPOSE: Main entry point for CMake build system. Configures project-wide
         settings, finds dependencies, and orchestrates the build.

WHAT IT DOES:
  1. Sets C++17 standard and compiler options
  2. Finds LibTorch library using cmake/FindLibTorch.cmake
  3. Defines configurable paths (LIBTORCH_PATH, MODEL_PATH, etc.)
  4. Generates abqnn_config.h with path constants for C++ code
  5. Generates UMAT_base.for with path constants for Fortran code
  6. Includes src/ and tests/ subdirectories
  7. Configures installation targets
================================================================================
]]

# Build options (ON/OFF switches)
option(BUILD_TESTS "Build test executables" ON)
option(ENABLE_DEBUG_OUTPUT "Enable debug output to log files" OFF)
option(USE_SCCACHE "Use sccache to accelerate compilation if available" ON)

# C++ configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Platform-specific flags
if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS -DNOMINMAX)
    # Enable MSVC parallel compilation across translation units
    add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Prefer faster builds with Ninja if user chooses that generator; enable parallel builds
# Users can also pass --parallel to cmake --build to increase parallelism

# -----------------------------------------------------------------------------
# Find LibTorch
# -----------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(LibTorch REQUIRED)

# -----------------------------------------------------------------------------
# Configurable Paths (can be overridden with -D flags)
# -----------------------------------------------------------------------------
set(LIBTORCH_LIB_PATH "${LibTorch_LIB_DIR}" CACHE PATH "LibTorch DLL directory")
set(ABAQUS_LIB_PATH "" CACHE PATH "Abaqus library directory (for installation)")
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/models" CACHE PATH "PyTorch models directory")
set(LOG_PATH "${CMAKE_SOURCE_DIR}/log" CACHE PATH "Debug log directory")

# Platform flag for config header
set(ABQNN_PLATFORM_WINDOWS ${WIN32})

# -----------------------------------------------------------------------------
# Generate Configuration Files
# -----------------------------------------------------------------------------
# C/C++ config header
configure_file(
    ${CMAKE_SOURCE_DIR}/cmake/config.h.in
    ${CMAKE_BINARY_DIR}/include/abqnn_config.h
    @ONLY
)

# Optional: use sccache (or clcache) as a compiler launcher to cache builds
if(USE_SCCACHE)
    find_program(SCCACHE_PROGRAM sccache)
    if(SCCACHE_PROGRAM)
        message(STATUS "Using sccache: ${SCCACHE_PROGRAM}")
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
        set(CMAKE_Fortran_COMPILER_LAUNCHER ${SCCACHE_PROGRAM})
    else()
        if(WIN32)
            find_program(CLCACHE_PROGRAM clcache)
            if(CLCACHE_PROGRAM)
                message(STATUS "Using clcache: ${CLCACHE_PROGRAM}")
                set(CMAKE_CXX_COMPILER_LAUNCHER ${CLCACHE_PROGRAM})
            endif()
        endif()
    endif()
endif()

# Fortran UMAT file
configure_file(
    ${CMAKE_SOURCE_DIR}/fortran/UMAT_base.for.in
    ${CMAKE_BINARY_DIR}/fortran/UMAT_base.for
    @ONLY
)

# -----------------------------------------------------------------------------
# Build Libraries
# -----------------------------------------------------------------------------
add_subdirectory(src)

# -----------------------------------------------------------------------------
# Tests (optional)
# -----------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# Installation
# -----------------------------------------------------------------------------
include(GNUInstallDirs)

# Install to custom paths if specified
if(ABAQUS_LIB_PATH)
    install(TARGETS umat_auxlib ARCHIVE DESTINATION ${ABAQUS_LIB_PATH})
endif()
if(LIBTORCH_LIB_PATH)
    install(TARGETS umat_pt_caller RUNTIME DESTINATION ${LIBTORCH_LIB_PATH})
endif()

# -----------------------------------------------------------------------------
# Build Summary
# -----------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== ABQnn Configuration ===")
message(STATUS "  LIBTORCH_PATH:    ${LIBTORCH_PATH}")
message(STATUS "  MODEL_PATH:       ${MODEL_PATH}")
message(STATUS "  LOG_PATH:         ${LOG_PATH}")
message(STATUS "  Debug output:     ${ENABLE_DEBUG_OUTPUT}")
message(STATUS "")
