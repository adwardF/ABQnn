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

target_link_libraries(pt_caller_test PRIVATE umat_auxlib)

add_executable(pt_caller_concurrency_test pt_caller_concurrency_test.cpp)

target_include_directories(pt_caller_concurrency_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/include
)

target_link_libraries(pt_caller_concurrency_test PRIVATE umat_auxlib)

add_test(NAME cpp_test COMMAND pt_caller_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME cpp_concurrency_test COMMAND pt_caller_concurrency_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
set_tests_properties(cpp_concurrency_test PROPERTIES TIMEOUT 70)

if(WIN32)
    find_program(ABQNN_PWSH_EXE NAMES pwsh powershell REQUIRED)

    set(ABQNN_TEST_PID_FILE "${CMAKE_BINARY_DIR}/abqnn_inference_server.pid")

    add_test(
        NAME ipc_server_setup
        COMMAND ${ABQNN_PWSH_EXE} -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/start_ipc_server.ps1
                -ServerExe $<TARGET_FILE:abqnn_inference_server>
                -TorchLibDir ${LIBTORCH_LIB_PATH}
                -PidFile ${ABQNN_TEST_PID_FILE}
    )
    set_tests_properties(ipc_server_setup PROPERTIES FIXTURES_SETUP ipc_server)

    add_test(
        NAME ipc_server_cleanup
        COMMAND ${ABQNN_PWSH_EXE} -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/stop_ipc_server.ps1
                -PidFile ${ABQNN_TEST_PID_FILE}
    )
    set_tests_properties(ipc_server_cleanup PROPERTIES FIXTURES_CLEANUP ipc_server)

    set_tests_properties(cpp_test PROPERTIES FIXTURES_REQUIRED ipc_server)
    set_tests_properties(cpp_concurrency_test PROPERTIES FIXTURES_REQUIRED ipc_server)
endif()

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

    add_executable(vumat_fortest VUMAT_fortest.f90)
    target_link_libraries(vumat_fortest PRIVATE umat_auxlib)
    set_target_properties(vumat_fortest PROPERTIES LINKER_LANGUAGE Fortran)
    
    add_test(NAME fortran_test       COMMAND umat_fortest  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    add_test(NAME fortran_vumat_test COMMAND vumat_fortest WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    if(WIN32)
        set_tests_properties(fortran_test       PROPERTIES FIXTURES_REQUIRED ipc_server)
        set_tests_properties(fortran_vumat_test PROPERTIES FIXTURES_REQUIRED ipc_server)
    endif()
else()
    message(STATUS "No Fortran compiler - skipping Fortran tests")
endif()
