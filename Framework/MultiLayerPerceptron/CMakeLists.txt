cmake_minimum_required(VERSION 3.20)
project(MultiLayerPerceptron)

set(CMAKE_CXX_STANDARD 14)

add_library(MultiLayerPerceptron SHARED multiLayerPerceptron.cpp)
