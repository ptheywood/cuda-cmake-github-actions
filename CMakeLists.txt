# Cmake for simple cuda project which uses NVTX markers as an optional thing.

# Set the minimum cmake version to that which supports cuda natively.
cmake_minimum_required(VERSION VERSION 3.10 FATAL_ERROR)


# Name the project and set languages
project(main CUDA CXX)


# Set a default build type if not passed
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release")
endif()

if(CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_CONFIGURATION_TYPES Debug Release)
  set(CMAKE_CONFIGURATION_TYPES "${CMAKE_CONFIGURATION_TYPES}" CACHE STRING
    "Reset the configurations to what we need"
    FORCE)
endif()


# Update the documentation string of CMAKE_BUILD_TYPE for GUIs
SET( CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING
    "Choose the type of build, options are: None Debug Release."
    FORCE )

# Specify using C++11 standard
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED true)
endif()

# Define which source files are required for the target executable

set(SRC_INCLUDE
    include/test.cuh
)

set(SRC
    src/main.cu
)

set(ALL_SRC
    ${SRC_INCLUDE}
    ${SRC}
)


set(PROJECTNAME "main")

add_executable(${PROJECTNAME} ${ALL_SRC})

target_include_directories(${PROJECTNAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)