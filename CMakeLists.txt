cmake_minimum_required(VERSION 3.9)

# ===============================================
# Project name
project(TinyAD-Examples)

# ===============================================
# Global settings

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Add -fPIC flag (required to build predicates.c)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set glow bin dir
set(GLOW_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin/glow)

# Set output dir for results, screenshots, etc...
set(TINYAD_EXAMPLES_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/output)

# Disable glfw additionals
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
option(GLFW_INSTALL "" OFF)

# Enable TinyAD unit tests
option(TINYAD_UNIT_TESTS "" ON)

# Enable optional dependencies
option(TINYAD_GEOMETRYCENTRAL "Build Geometry Central examples" ON)

# ===============================================
# Find packages

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Find OpenMP (optional)
if (NOT TARGET OpenMP::OpenMP_CXX)
    find_package(OpenMP)
endif()

if (NOT TARGET OpenMP::OpenMP_CXX)
    message("WARNING: Building TinyAD-Examples without OpenMP!")
endif()

# Output build type
message("Building TinyAD-Examples in " ${CMAKE_BUILD_TYPE} " mode")

# ===============================================
# Add submodules

add_subdirectory(libs/eigen-lean)
add_subdirectory(libs/googletest)
add_subdirectory(libs/TinyAD)

add_subdirectory(libs/OpenMesh)
add_subdirectory(libs/typed-geometry)
add_subdirectory(libs/polymesh)
add_subdirectory(libs/glfw)
add_subdirectory(libs/glow)
add_subdirectory(libs/imgui-lean)
add_subdirectory(libs/glow-extras)
add_subdirectory(libs/geometry-central)
add_subdirectory(libs/polyscope)

# Supress an OpenMesh warning
if (${QT_QMAKE_EXECUTABLE})
endif()

# ===============================================
# Target TinyAD-Examples

file(GLOB_RECURSE TINYAD_EXAMPLES_HEADERS include/*.hh include/*.h src/*hh src/*.h)
file(GLOB_RECURSE TINYAD_EXAMPLES_SOURCES src/*.cc src/*.c)
add_library(TinyAD-Examples ${TINYAD_EXAMPLES_SOURCES} ${TINYAD_EXAMPLES_HEADERS})

target_compile_features(TinyAD-Examples PUBLIC cxx_std_17)
target_compile_definitions(TinyAD-Examples PUBLIC SOURCE_PATH_STR="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(TinyAD-Examples PUBLIC DATA_PATH_STR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_compile_definitions(TinyAD-Examples PUBLIC OUTPUT_PATH_STR="${TINYAD_EXAMPLES_OUTPUT_DIR}")

target_include_directories(TinyAD-Examples PUBLIC
    libs/libigl/include
    libs/Directional/include
    ${OPENMESH_INCLUDE_DIRS}
    include
    src
)
target_link_libraries(TinyAD-Examples PUBLIC
    eigen
    OpenMeshCore
    typed-geometry
    polymesh
    glfw
    glow
    imgui
    glow-extras
    stdc++fs
    TinyAD
)
if (TARGET OpenMP::OpenMP_CXX)
    target_link_libraries(TinyAD-Examples PUBLIC OpenMP::OpenMP_CXX)
endif()


function(add_app app_name main)
    message(STATUS "Adding app ${app_name}")
    add_executable(${app_name} ${main})
    add_dependencies(${app_name} TinyAD-Examples)
    target_link_libraries(${app_name} PUBLIC TinyAD-Examples)
endfunction()

# ===============================================
# Example targets

file(GLOB_RECURSE TINYAD_EXAMPLE_APPS apps/*.cc)
foreach(main ${TINYAD_EXAMPLE_APPS})
    get_filename_component(app_name ${main} NAME_WE)
    get_filename_component(dir ${main} DIRECTORY)
    get_filename_component(dir_name ${dir} NAME)

    if (${app_name} STREQUAL "parametrization_geometrycentral")
        if (${TINYAD_GEOMETRYCENTRAL})
            add_app(${app_name} ${main})
            target_link_libraries(${app_name} PUBLIC geometry-central polyscope)
        endif()
    else()
        add_app(${app_name} ${main})
    endif()

endforeach()
