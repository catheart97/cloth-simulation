cmake_minimum_required(VERSION 3.12)
project(ClothSimulation)

enable_language(CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_AVX "Enables AVX256 and AVX512 support." OFF)
if (!APPLE AND ENABLE_AVX) 
    add_definitions(-DENABLE_AVX)
    add_compile_options(-mavx2 -mfma )
endif()

add_compile_options(-O2 -Wall -Wextra -Wpedantic -fopenmp)

find_package(OpenMP REQUIRED)

add_executable(
    ClothSimulation
    src/Timer.h
    src/LinearAlgebra.h
    src/main.cpp
)

target_link_libraries(ClothSimulation PUBLIC OpenMP::OpenMP_CXX)