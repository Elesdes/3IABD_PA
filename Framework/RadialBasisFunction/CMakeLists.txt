cmake_minimum_required(VERSION 3.22)
project(RadialBasisFunction)

set(CMAKE_CXX_STANDARD 14)

add_library(RadialBasisFunction SHARED radialBasisFunction.cpp)
