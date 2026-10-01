cmake_minimum_required(VERSION 3.16)
project(fx_toxicity_realtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native is a deliberate choice: this binary is built on the box it runs
# on. Drop it (or pin an ISA) if you ship a portable artifact.
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fno-math-errno")

add_library(fxtox_core INTERFACE)
target_include_directories(fxtox_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# ---- realtime engine --------------------------------------------------------
add_executable(vpin_realtime src/vpin_realtime.cpp)
target_link_libraries(vpin_realtime PRIVATE fxtox_core)
target_compile_options(vpin_realtime PRIVATE -Wall -Wextra)

# ---- C++ unit tests ---------------------------------------------------------
add_executable(fxtox_tests tests/test_core.cpp)
target_link_libraries(fxtox_tests PRIVATE fxtox_core)
target_compile_options(fxtox_tests PRIVATE -Wall -Wextra)
enable_testing()
add_test(NAME fxtox_core_tests COMMAND fxtox_tests)

# ---- Python extension -------------------------------------------------------
# Optional: configure with -DFXTOX_BUILD_PYTHON=OFF to skip it.
option(FXTOX_BUILD_PYTHON "Build the fxtox_native Python extension" ON)

if(FXTOX_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c
                    "import pybind11, sys; sys.stdout.write(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            RESULT_VARIABLE PYBIND11_RESULT
            ERROR_QUIET)
        if(PYBIND11_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 CONFIG REQUIRED)
            pybind11_add_module(fxtox_native src/bindings.cpp)
            target_link_libraries(fxtox_native PRIVATE fxtox_core)
            # Land the .so next to the Python package so `import fxtox_native`
            # works straight out of the build, with no install step.
            set_target_properties(fxtox_native PROPERTIES
                LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src")
        else()
            message(WARNING "pybind11 not importable; skipping fxtox_native. "
                            "The Python pipeline will fall back to NumPy.")
        endif()
    else()
        message(WARNING "Python3 development headers not found; skipping fxtox_native.")
    endif()
endif()
