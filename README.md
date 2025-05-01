# TinyAD-Examples

This repository contains **code examples** for **automatic differentiation with TinyAD**. Please see the [**README in the main TinyAD repository**](https://github.com/patr-schm/TinyAD) for an overview.

Note that this repository comes with a massive amount of dependencies as submodules. However, [TinyAD](https://github.com/patr-schm/TinyAD) itself is a lightweight header-only library that only depends on Eigen and has been tested on Linux, Windows, and Mac.

## Linux

These examples have been tested on Linux and require:
* A C++17 compiler
* CMake >= 3.9 (e.g. `sudo apt-get install cmake`)
* GLFW build dependencies (e.g. `sudo apt-get build-dep glfw3`)
* OpenGL >= 4.5 (e.g. `sudo apt-get install libgl1-mesa-dev mesa-utils`)

Build the examples via:
```
git clone --recursive https://github.com/patr-schm/TinyAD-Examples.git
cd TinyAD-Examples
mkdir build
cd build
cmake ..
make 
```

## Windows
The branch `windows` compiled successfully with MSVC in 2025.