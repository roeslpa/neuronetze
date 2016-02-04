cmake_minimum_required(VERSION 3.3)
project(knn6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp KM.h matrix.h)
add_executable(knn6 ${SOURCE_FILES})