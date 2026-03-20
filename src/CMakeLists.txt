#[[
================================================================================
src/CMakeLists.txt - Library Build Configuration
================================================================================
PURPOSE: Defines how to build the two main libraries.

LIBRARIES BUILT:
    1. umat_auxlib (STATIC)
     - Linked into Abaqus UMAT
         - Abaqus-facing IPC client bridge
     - Exports: invoke_pt() - called from Fortran

    2. abqnn_inference_server (EXE)
      - Links against LibTorch
      - Runs Torch inference out-of-process
================================================================================
]]

# -----------------------------------------------------------------------------
# abqnn_inference_server.exe - Torch inference server (out-of-process)
# -----------------------------------------------------------------------------
add_executable(abqnn_inference_server ABQnn_inference_server.cpp abqnn_ipc_common.cpp)

target_include_directories(abqnn_inference_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${LibTorch_INCLUDE_DIRS}
)

target_link_libraries(abqnn_inference_server PRIVATE ${LibTorch_LIBRARIES})

target_compile_definitions(abqnn_inference_server PRIVATE
    $<$<BOOL:${ENABLE_DEBUG_OUTPUT}>:ENABLE_DEBUG_OUTPUT>
)

target_precompile_headers(abqnn_inference_server PRIVATE
    <torch/torch.h>
    <torch/script.h>
)

# -----------------------------------------------------------------------------
# umat_auxlib.lib - Static library for Abaqus linking
# -----------------------------------------------------------------------------
add_library(umat_auxlib STATIC UMAT_auxlib.cpp abqnn_ipc_common.cpp)

target_include_directories(umat_auxlib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(umat_auxlib PRIVATE
    $<$<BOOL:${ENABLE_DEBUG_OUTPUT}>:ENABLE_DEBUG_OUTPUT>
)

# target_link_libraries(umat_pt_caller PRIVATE ucrt.lib vcruntime.lib msvcrt.lib)