cmake_minimum_required(VERSION 3.10)

# Project name
project(MicrogradCpp VERSION 1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set project source files
set(SOURCE_FILES
    src/engine/engine.cc
)

# Include header directories
include_directories(src/engine)

# Add executable file
add_executable(MicrogradCpp main.cpp ${SOURCE_FILES})
