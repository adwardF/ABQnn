#[[
================================================================================
src/CMakeLists.txt - Library Build Configuration
================================================================================
PURPOSE: Defines how to build the two main libraries.

LIBRARIES BUILT:
  1. umat_pt_caller (SHARED/DLL)
     - Links against LibTorch
     - Loads and runs PyTorch models
     - Exports: pt_module_invoke()

  2. umat_auxlib (STATIC)
     - Linked into Abaqus UMAT
     - Handles DLL loading at runtime
     - Exports: invoke_pt() - called from Fortran
================================================================================
]]

# -----------------------------------------------------------------------------
# umat_pt_caller.dll - PyTorch model inference
# -----------------------------------------------------------------------------
add_library(umat_pt_caller SHARED UMAT_pt_caller.cpp)

target_include_directories(umat_pt_caller PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${LibTorch_INCLUDE_DIRS}
)

target_link_libraries(umat_pt_caller PRIVATE ${LibTorch_LIBRARIES})

target_compile_definitions(umat_pt_caller PRIVATE
    UMAT_PT_CALLER_EXPORTS
    $<$<BOOL:${ENABLE_DEBUG_OUTPUT}>:ENABLE_DEBUG_OUTPUT>
)

# Precompiled headers to speed heavy Torch includes
target_precompile_headers(umat_pt_caller PRIVATE
    <torch/torch.h>
    <torch/script.h>
)

# -----------------------------------------------------------------------------
# umat_auxlib.lib - Static library for Abaqus linking
# -----------------------------------------------------------------------------
add_library(umat_auxlib STATIC UMAT_auxlib.cpp)

target_include_directories(umat_auxlib PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_compile_definitions(umat_auxlib PRIVATE
    $<$<BOOL:${ENABLE_DEBUG_OUTPUT}>:ENABLE_DEBUG_OUTPUT>
)

# target_link_libraries(umat_pt_caller PRIVATE ucrt.lib vcruntime.lib msvcrt.lib)