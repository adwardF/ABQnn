#[[
================================================================================
tests/CMakeLists.txt - Test Executables
================================================================================
PURPOSE: Builds test programs to verify the libraries work correctly.

TESTS:
  - pt_caller_test (C++) - Tests pt_module_invoke directly
  - umat_fortest (Fortran) - Tests invoke_pt from Fortran (if compiler available)
================================================================================
]]

# -----------------------------------------------------------------------------
# C++ Test
# -----------------------------------------------------------------------------
add_executable(load_test load_test.cpp)

target_include_directories(load_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
    ${LibTorch_INCLUDE_DIRS}
)

target_link_libraries(load_test PRIVATE ${LibTorch_LIBRARIES})

# Precompiled headers for Torch in load_test to reduce compile time
target_precompile_headers(load_test PRIVATE
    <torch/torch.h>
    <torch/script.h>
)

add_executable(pt_caller_test pt_caller_test.cpp)

target_include_directories(pt_caller_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(pt_caller_test PRIVATE umat_pt_caller)

add_test(NAME cpp_test COMMAND pt_caller_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# -----------------------------------------------------------------------------
# Fortran Test (optional - only if compiler found)
# -----------------------------------------------------------------------------
include(CheckLanguage)
check_language(Fortran)

if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")
    
    add_executable(umat_fortest UMAT_fortest.f90)
    target_link_libraries(umat_fortest PRIVATE umat_auxlib)
    set_target_properties(umat_fortest PROPERTIES LINKER_LANGUAGE Fortran)
    
    add_test(NAME fortran_test COMMAND umat_fortest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
else()
    message(STATUS "No Fortran compiler - skipping Fortran tests")
endif()
