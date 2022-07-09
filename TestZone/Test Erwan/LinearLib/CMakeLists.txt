cmake_minimum_required(VERSION 3.20)
project(Library)

set(CMAKE_CXX_STANDARD 14)

SET(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")
SET(CMAKE_MODULE_PATH "$ENV{EIGEN3_DIR}")

find_package(Eigen3 3.3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

add_library(Library SHARED LinearClassificationLib.cpp)
