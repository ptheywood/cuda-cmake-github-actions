# Cmake for simple cuda project which uses NVTX markers as an optional thing.

# Set the minimum cmake version to that which provides modern CMake CUDA support (CMAKE_CUDA_ARCHITECTURES)
cmake_minimum_required(VERSION VERSION 3.18 FATAL_ERROR)

option(USE_CHECK_LANGUAGE "Control use of check_languge vs project(... LANGUAGES CUDA)" ON)

if (USE_CHECK_LANGUAGE)
    # Declare a project without languages, so we can use check_langauge to provide a more helpful error message in some cases
    project(main LANGUAGES NONE)
    # Check for CXX support, which is required
    include(CheckLanguage)
    check_language(CXX)
    if(CMAKE_CXX_COMPILER)
        enable_language(CXX)
    else()
        message(FATAL_ERROR "CXX Language support is required but not found")
    endif()
    # Check for CUDA support, once the host compiler has already been found.
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        # If using MSVC >= 1940 then CUDA <= 12.3 support requires -allow-unsupported-compiler, so warn about this
        if(MSVC AND MSVC_VERSION VERSION_GREATER_EQUAL "1941")
            if(CMAKE_VERSION VERSION_LESS "3.29.4")
                message(FATAL_ERROR 
                    " CUDA Language support detection failed with MSVC ${MSVC_VERSION}\n"
                    " \n"
                    " If you have CUDA <= 12.3 installed:\n"
                    "  - You must upgrade CMake to be >= 3.29.4\n"
                    "    The CUDAFLAGS environment variable must include '-allow-unsupported-compiler'\n"
                    "    @todo - something about defining _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH\n"
                    "    You must clear the CMake Cache before reconfiguring this project\n"
                    " \n"
                    "  - Alternatively you may upgrade CUDA to >= 12.4 and clear the CMake Cache before reconfiguring"
                )
            else()
                message(FATAL_ERROR 
                    " CUDA Language support detection failed with MSVC ${MSVC_VERSION}\n"
                    " \n"
                    " If you have CUDA <= 12.3 installed:\n"
                    "  - The CUDAFLAGS environment variable must include '-allow-unsupported-compiler'\n"
                    "    @todo - something about defining _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH\n"
                    "    You must clear the CMake Cache before reconfiguring this project\n"
                    " \n"
                    "  - Alternatively you may upgrade CUDA to >= 12.4 and clear the CMake Cache before reconfiguring"
                )
            if()
        elseif(MSVC AND MSVC_VERSION VERSION_GREATER_EQUAL "1940")
            # If this is the case, then CMake >= 3.29.4 is also required, otherwise CMake does not pass -allow-unsupported-compiler along, warn as appropriate
            if(CMAKE_VERSION VERSION_LESS "3.29.4")
                message(FATAL_ERROR 
                    " CUDA Language support detection failed with MSVC ${MSVC_VERSION}\n"
                    " \n"
                    " If you have CUDA <= 12.3 installed:\n"
                    "  - You must upgrade CMake to be >= 3.29.4\n"
                    "    The CUDAFLAGS environment variable must include '-allow-unsupported-compiler'\n"
                    "    You must clear the CMake Cache before reconfiguring this project\n"
                    " \n"
                    "  - Alternatively you may upgrade CUDA to >= 12.4 and clear the CMake Cache before reconfiguring"
                )
            else()
                message(FATAL_ERROR 
                    " CUDA Language support detection failed with MSVC ${MSVC_VERSION}\n"
                    " \n"
                    " If you have CUDA <= 12.3 installed:\n"
                    "  - The CUDAFLAGS environment variable must include '-allow-unsupported-compiler'\n"
                    "    You must clear the CMake Cache before reconfiguring this project\n"
                    " \n"
                    "  - Alternatively you may upgrade CUDA to >= 12.4 and clear the CMake Cache before reconfiguring"
                )
            endif()
        endif()
        # CUDA support is required
        message(FATAL_ERROR "CUDA Language support is required but not found")
    endif()
else()
    # Declare a project with LANGUAGES, which cannot provide more useful error messages than given by CMake (but might work)
    project(main LANGUAGES CXX CUDA)
endif()

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