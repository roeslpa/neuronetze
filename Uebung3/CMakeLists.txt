cmake_minimum_required(VERSION 3.3)
project(knn2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp GradientDescent.h matrix.h)
add_executable(knn3 ${SOURCE_FILES})