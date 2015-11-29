cmake_minimum_required(VERSION 3.3)
project(knn2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp ProbabilityDistribution.h matrix.h)
add_executable(knn2 ${SOURCE_FILES})