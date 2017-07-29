# Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

cmake_minimum_required(VERSION 3.0)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJ_BINARY_ROOT ${CMAKE_CURRENT_BINARY_DIR})

include(system)

project(paddle CXX C)

find_package(CUDA QUIET)
find_package(Git REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost QUIET)

include(simd)

################################ Configurations #######################################
option(WITH_GPU         "Compile PaddlePaddle with NVIDIA GPU"          ${CUDA_FOUND})
option(WITH_AVX         "Compile PaddlePaddle with AVX intrinsics"      ${AVX_FOUND})
option(WITH_DSO         "Compile PaddlePaddle with dynamic linked CUDA" ON)
option(WITH_TESTING     "Compile PaddlePaddle with unit testing"        ON)
option(WITH_STYLE_CHECK "Compile PaddlePaddle with style check"         ON)
option(WITH_PYTHON      "Compile PaddlePaddle with python interpreter"  ON)

# CMAKE_BUILD_TYPE
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel"
      FORCE)
endif()



set(THIRD_PARTY_PATH "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
  "A path setting third party libraries download & build directories.")


########################################################################################

include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
include(external/gtest)     # download, build, install gtest
include(external/protobuf)  # download, build, install protobuf
include(external/python)    # download, build, install python
include(external/openblas)  # download, build, install openblas
include(external/any)       # download libn::any
include(external/eigen)     # download eigen3
include(external/pybind11)    # download pybind11

include(cudnn)              # set cudnn libraries, must before configure
include(configure)          # add paddle env configuration
include(generic)            # simplify cmake module
include(package)            # set paddle packages
include(cpplint)            # set paddle c++ style
include(ccache)             # set ccache for compilation
include(util)               # set unittest and link libs
include(flags)              # set paddle compile flags
include(version)            # set PADDLE_VERSION


include_directories("${PROJ_ROOT}")
include_directories(${Boost_INCLUDE_DIRS})

set(EXTERNAL_LIBS
    ${GFLAGS_LIBRARIES}
    ${GLOG_LIBRARIES}
    ${CBLAS_LIBRARIES}
    ${PROTOBUF_LIBRARY}
    ${ZLIB_LIBRARIES}
    ${PYTHON_LIBRARIES}
)

add_subdirectory(paddle)

if(WITH_PYTHON)
  add_subdirectory(python)
endif()
