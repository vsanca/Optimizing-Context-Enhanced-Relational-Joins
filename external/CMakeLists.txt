set(EXTERNAL_LIB_PATH "/scratch4/sanca/opt")
set(SRC_LIB_PATH "/scratch4/sanca/libs")

list(APPEND CMAKE_PREFIX_PATH ${EXTERNAL_LIB_PATH})

find_path(MLPACK_INCLUDE_DIR NAMES mlpack/core.hpp mlpack/prereqs.hpp PATHS "${CMAKE_PREFIX_PATH}/mlpack/include/" REQUIRED)
find_path(FASTTEXT_INCLUDE_DIR NAMES fasttext/fasttext.h fasttext/vector.h fasttext/model.h fasttext/autotune.h fasttext/args.h PATHS "${CMAKE_PREFIX_PATH}/fasttext/include" REQUIRED)
find_path(SIMD_INCLUDE_DIR NAMES Simd/SimdLib.h PATHS "${SRC_LIB_PATH}/Simd/src" REQUIRED)

include_directories(${SRC_LIB_PATH}/mlpack/build/deps/armadillo-10.3.0/include)
include_directories(${SRC_LIB_PATH}/mlpack/build/deps/stb/include)
include_directories(${SRC_LIB_PATH}/mlpack/build/deps/ensmallen-2.19.0/include)
include_directories(${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include)
include_directories(${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include)

#message(STATUS ${MLPACK_INCLUDE_DIR})
#message(STATUS ${FASTTEXT_INCLUDE_DIR})

find_library(SIMD_LIBRARY NAMES Simd PATHS "/scratch4/sanca/libs/Simd/build" REQUIRED)
find_library(FASTTEXT_LIBRARY NAMES fasttext PATHS "${CMAKE_PREFIX_PATH}/fasttext/lib" REQUIRED)

include(FetchContent)
cmake_minimum_required(VERSION 3.14)

# Externally provided libraries
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.12.0)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.7.0)

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)