cmake_minimum_required(VERSION 3.10)
project(lstm)

set(CMAKE_CXX_STANDARD 11)

include_directories(./include)

add_executable(lstm main.cpp include/fast_grnn.h)