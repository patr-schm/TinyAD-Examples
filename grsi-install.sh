#!/bin/bash

# Install dependencies in a first step.
sudo apt-get install cmake
sudo apt-get build-dep glfw3
sudo apt install cmake libgl1-mesa-dev mesa-utils

# Build project.
mkdir build
cd build
cmake ..
make -j4
