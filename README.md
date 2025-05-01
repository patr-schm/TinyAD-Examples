# TinyAD-Examples

This repository provides **examples** for automatic differentiation using **TinyAD**. For an overview, visit the [**main TinyAD repository**](https://github.com/patr-schm/TinyAD).

While this repository includes many dependencies as submodules, [TinyAD](https://github.com/patr-schm/TinyAD) itself is a lightweight, header-only library requiring only Eigen. It has been tested on Linux, Windows, and Mac.

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