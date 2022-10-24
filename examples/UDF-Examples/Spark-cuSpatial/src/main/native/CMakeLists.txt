#=============================================================================
# Copyright (c) 2022, NVIDIA CORPORATION.
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

cmake_minimum_required(VERSION 3.20.1 FATAL_ERROR)

project(SPATIALUDJNI VERSION 22.10.0 LANGUAGES C CXX CUDA)

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

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=deprecated-declarations")
endif(CMAKE_COMPILER_IS_GNUCXX)

####################################################################################################
# - find cuda toolkit ------------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

####################################################################################################
# - find cudf --------------------------------------------------------------------------------------

find_path(
    CUDF_INCLUDE "cudf"
    HINTS "$ENV{CONDA_PREFIX}/include"
          "${CONDA_PREFIX}/include"
          "$ENV{CUDF_ROOT}/include"
          "${CUDF_ROOT}/include"
    REQUIRED)

find_library(
    CUDF_LIB "cudf"
    HINTS "$ENV{CONDA_PREFIX}/lib"
          "${CONDA_PREFIX}/lib"
          "$ENV{CUDF_ROOT}/lib"
          "${CUDF_ROOT}/lib"
    REQUIRED)

set(CUDACXX_INCLUDE "$ENV{CONDA_PREFIX}/include/rapids/libcudacxx")

####################################################################################################
# - find cuspatial --------------------------------------------------------------------------------------

find_path(
    CUSPATIAL_INCLUDE "cuspatial"
    HINTS "$ENV{CONDA_PREFIX}/include"
          "${CONDA_PREFIX}/include"
          "$ENV{CUSPATIAL_ROOT}/include"
          "${CUSPATIAL_ROOT}/include"
    REQUIRED)

find_library(
    CUSPATIAL_LIB "cuspatial"
    HINTS "$ENV{CONDA_PREFIX}/lib"
          "${CONDA_PREFIX}/lib"
          "$ENV{CUSPATIAL_ROOT}/lib"
          "${CUSPATIAL_ROOT}/lib"
    REQUIRED)

###################################################################################################
# - find JNI -------------------------------------------------------------------------------------

find_package(JNI REQUIRED)
if(JNI_FOUND)
    message(STATUS "JDK with JNI in ${JNI_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "JDK with JNI not found, please check your environment")
endif(JNI_FOUND)

###################################################################################################
# - library targets -------------------------------------------------------------------------------

add_library(spatialudfjni SHARED "src/PointInPolygonJni.cpp")

# Override RPATH
set_target_properties(spatialudfjni PROPERTIES BUILD_RPATH "\$ORIGIN")

# include directories
target_include_directories(
    spatialudfjni
    PUBLIC "${JNI_INCLUDE_DIRS}"
           "${CUDF_INCLUDE}"
           "${CUDACXX_INCLUDE}"
           "${CUDAToolkit_INCLUDE_DIRS}"
           "${CUSPATIAL_INCLUDE}")

###################################################################################################
# - link libraries --------------------------------------------------------------------------------

target_link_libraries(spatialudfjni ${CUDF_LIB} ${CUSPATIAL_LIB})
