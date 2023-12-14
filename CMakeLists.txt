cmake_minimum_required(VERSION 3.14)
project(j2v)

set(CMAKE_CXX_STANDARD 23)

set(CMAKE_CXX_FLAGS "-O3 -march=native -DMKL_ILP64 -I/opt/intel/oneapi/mkl/latest/include -L/scratch4/sanca/opt/OpenBLAS/lib/ -L/opt/intel/oneapi/mkl/latest/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -fopenmp -lpthread -lm -ldl -L/usr/lib/gcc/x86_64-linux-gnu/7/ -ffast-math -DEIGEN_NO_DEBUG")


add_subdirectory(external)
add_subdirectory(src)


include_directories(${MLPACK_INCLUDE_DIR})
include_directories(${FASTTEXT_INCLUDE_DIR})
include_directories(${SIMD_INCLUDE_DIR})
include_directories(${SRC_LIB_PATH}/libeigen/include/eigen3)

add_executable(j2v src/main.cpp src/join2vec/join2vec.cpp src/join2vec/hashjoin2vec.cpp include/util/TimeBlock.cpp src/join2vec/cluster.cpp
        include/join2vec/scan.h
        src/join2vec/scan.cpp
        include/util/Tests.h
        include/util/Tests.cpp)

target_link_libraries(j2v ${FASTTEXT_LIBRARY} ${SIMD_LIBRARY} benchmark::benchmark stdc++fs /scratch4/sanca/opt/OpenBLAS/lib/libopenblas.so gomp)
target_include_directories(j2v PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/join2vec ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/join2vec /scratch4/sanca/libs/libeigen/include/eigen3 ${SRC_LIB_PATH}/mlpack/build/deps/armadillo-10.3.0/include ${SRC_LIB_PATH}/mlpack/build/deps/stb/include ${SRC_LIB_PATH}/mlpack/build/deps/ensmallen-2.19.0/include ${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include ${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include)


add_library(libj2v src/join2vec/join2vec.cpp src/join2vec/hashjoin2vec.cpp include/util/TimeBlock.cpp src/join2vec/cluster.cpp
        include/join2vec/scan.h
        src/join2vec/scan.cpp
        include/util/Tests.h
        include/util/Tests.cpp)

target_link_libraries(libj2v ${FASTTEXT_LIBRARY} ${SIMD_LIBRARY} benchmark::benchmark stdc++fs /scratch4/sanca/opt/OpenBLAS/lib/libopenblas.so gomp)
target_include_directories(libj2v PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/join2vec ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/join2vec /scratch4/sanca/libs/libeigen/include/eigen3 ${SRC_LIB_PATH}/mlpack/build/deps/armadillo-10.3.0/include ${SRC_LIB_PATH}/mlpack/build/deps/stb/include ${SRC_LIB_PATH}/mlpack/build/deps/ensmallen-2.19.0/include ${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include ${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include)



add_executable(perf_j2v perf/bench.cpp src/join2vec/cluster.cpp
        include/join2vec/scan.h
        src/join2vec/scan.cpp
        include/util/Tests.h
        include/util/Tests.cpp)

target_link_libraries(perf_j2v ${FASTTEXT_LIBRARY} ${SIMD_LIBRARY} benchmark::benchmark stdc++fs libj2v /scratch4/sanca/opt/OpenBLAS/lib/libopenblas.so gomp)
target_include_directories(perf_j2v PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/src/join2vec ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/include/join2vec /scratch4/sanca/libs/libeigen/include/eigen3 ${SRC_LIB_PATH}/mlpack/build/deps/armadillo-10.3.0/include ${SRC_LIB_PATH}/mlpack/build/deps/stb/include ${SRC_LIB_PATH}/mlpack/build/deps/ensmallen-2.19.0/include ${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include ${SRC_LIB_PATH}/mlpack/build/deps/cereal-1.3.0/include)
