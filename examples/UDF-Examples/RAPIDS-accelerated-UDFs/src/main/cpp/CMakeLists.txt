#=============================================================================
# Copyright (c) 2021-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Keep the same with https://github.com/rapidsai/rapids-cmake/blob/branch-xx.yy/RAPIDS.cmake
# the branch-xx.xx in the above line is the branch version
cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

# set to the rapids-cmake version
set(rapids-cmake-version "25.06")

file(DOWNLOAD https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-${rapids-cmake-version}/RAPIDS.cmake
     ${CMAKE_BINARY_DIR}/RAPIDS.cmake)
include(${CMAKE_BINARY_DIR}/RAPIDS.cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)

# Use GPU_ARCHS if it is defined
if(DEFINED GPU_ARCHS)
  set(CMAKE_CUDA_ARCHITECTURES "${GPU_ARCHS}")
endif()
rapids_cuda_init_architectures(UDFEXAMPLESJNI)

project(UDFEXAMPLESJNI VERSION 25.06.0 LANGUAGES C CXX CUDA)

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
option(BUILD_UDF_BENCHMARKS "Build the benchmarks" OFF)

###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

###################################################################################################
# - compiler options ------------------------------------------------------------------------------

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
endif(CMAKE_COMPILER_IS_GNUCXX)

if(CMAKE_CUDA_COMPILER_VERSION)
  # Compute the version. from  CMAKE_CUDA_COMPILER_VERSION
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\1" CUDA_VERSION_MAJOR ${CMAKE_CUDA_COMPILER_VERSION})
  string(REGEX REPLACE "([0-9]+)\\.([0-9]+).*" "\\2" CUDA_VERSION_MINOR ${CMAKE_CUDA_COMPILER_VERSION})
  set(CUDA_VERSION "${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}" CACHE STRING "Version of CUDA as computed from nvcc.")
  mark_as_advanced(CUDA_VERSION)
endif()

message(STATUS "CUDA_VERSION_MAJOR: ${CUDA_VERSION_MAJOR}")
message(STATUS "CUDA_VERSION_MINOR: ${CUDA_VERSION_MINOR}")
message(STATUS "CUDA_VERSION: ${CUDA_VERSION}")

# Always set this convenience variable
set(CUDA_VERSION_STRING "${CUDA_VERSION}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w --expt-extended-lambda --expt-relaxed-constexpr")

####################################################################################################
# - cudf -------------------------------------------------------------------------------------------

# Ensure CUDA runtime is dynamic despite statically linking Arrow in libcudf
set(CUDA_USE_STATIC_CUDA_RUNTIME ON)

rapids_cpm_init()
rapids_cpm_find(cudf 25.06.00
        CPM_ARGS
        GIT_REPOSITORY  https://github.com/rapidsai/cudf.git
        GIT_TAG         branch-25.06
        GIT_SHALLOW     TRUE
        SOURCE_SUBDIR   cpp
        OPTIONS         "BUILD_TESTS OFF"
                        "BUILD_BENCHMARKS OFF"
                        "CUDF_USE_ARROW_STATIC ON"
                        "JITIFY_USE_CACHE ON"
                        "CUDA_STATIC_RUNTIME ${CUDA_USE_STATIC_CUDA_RUNTIME}"
                        "DISABLE_DEPRECATION_WARNING ON"
                        "AUTO_DETECT_CUDA_ARCHITECTURES OFF"
                        "CUDF_KVIKIO_REMOTE_IO OFF"
    )

###################################################################################################
# - benchmarks ------------------------------------------------------------------------------------

if(BUILD_UDF_BENCHMARKS)
    # Find or install GoogleBench
    CPMFindPackage(NAME benchmark
        VERSION         1.5.2
        GIT_REPOSITORY  https://github.com/google/benchmark.git
        GIT_TAG         v1.5.2
        GIT_SHALLOW     TRUE
        OPTIONS         "BENCHMARK_ENABLE_TESTING OFF"
                        "BENCHMARK_ENABLE_INSTALL OFF")
    add_subdirectory(benchmarks)
endif()

###################################################################################################
# - find JNI -------------------------------------------------------------------------------------

find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JDK with JNI not found, please check your settings.")
endif(JNI_FOUND)

###################################################################################################
# - library paths ---------------------------------------------------------------------------------

# CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES is an undocumented/unsupported variable containing the link directories for nvcc
link_directories("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}"
                 "${CMAKE_BINARY_DIR}/lib")


###################################################################################################
# - library targets -------------------------------------------------------------------------------

set(SOURCE_FILES
    "src/CosineSimilarityJni.cpp"
    "src/StringWordCountJni.cpp"
    "src/cosine_similarity.cu"
    "src/string_word_count.cu")

add_library(udfexamplesjni SHARED ${SOURCE_FILES})

#Override RPATH for udfexamplesjni
SET_TARGET_PROPERTIES(udfexamplesjni PROPERTIES BUILD_RPATH "\$ORIGIN")

###################################################################################################
# - build options ---------------------------------------------------------------------------------

option(PER_THREAD_DEFAULT_STREAM "Build with per-thread default stream" OFF)
if(PER_THREAD_DEFAULT_STREAM)
    message(STATUS "Using per-thread default stream")
    target_compile_definitions(udfexamplesjni PRIVATE CUDA_API_PER_THREAD_DEFAULT_STREAM)
endif(PER_THREAD_DEFAULT_STREAM)

target_include_directories(udfexamplesjni PRIVATE ${JNI_INCLUDE_DIRS})

###################################################################################################
# - rmm logging level -----------------------------------------------------------------------------

set(RMM_LOGGING_LEVEL "OFF" CACHE STRING "Choose the logging level.")
# Set the possible values of build type for cmake-gui
set_property(CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS
        "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF")
message(STATUS "RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

target_compile_definitions(udfexamplesjni
    PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_${RMM_LOGGING_LEVEL})

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(udfexamplesjni cudf::cudf)
