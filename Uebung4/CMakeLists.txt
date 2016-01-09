cmake_minimum_required(VERSION 3.3)
project(knn4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp MLP.h matrix.h)
add_executable(knn4 ${SOURCE_FILES})