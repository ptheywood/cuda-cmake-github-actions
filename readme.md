# CUDA + Cmake example using Github Actions

This repo contains CI installation scripts, workflow examples and a very simple CUDA application, to demonstrate the installation of the CUDA toolkit (but not driver) on GitHub Hosted runners.
Execution of CUDA code is not (yet) possible on GitHub hosted runners.

Network CUDA installers are used, with a hard-coded set of subpackages to install within each installation script, to avoid large and slow installation processes.

[![Ubuntu](https://github.com/ptheywood/cuda-cmake-github-actions/workflows/Ubuntu/badge.svg)](https://github.com/ptheywood/cuda-cmake-github-actions/actions?query=workflow%3AUbuntu)
[![Windows](https://github.com/ptheywood/cuda-cmake-github-actions/workflows/Windows/badge.svg)](https://github.com/ptheywood/cuda-cmake-github-actions/actions?query=workflow%3AWindows)

## CUDA and GitHub Actions Version Compatibility

CUDA is only supported with appropriate host compilers and host operating systems.

This support matrix can be found in the CUDA documentation, but to summarise (at the time of writing): 


| Runner | Host Compiler | CUDA |
|--------|------|---------------|
| [ubuntu-2204] | GCC 12 | >= `12.0` |
| [ubuntu-2204] | GCC 6 - 11 | >= `11.7` |
| [ubuntu-2004] | GCC 10 | >= `11.4`  (`11.4.1`) |
| [ubuntu-2004] | GCC 6 - 9 | >= `11.0` |
| [windows-2022] | Visual Studio 17 2022 | >= `11.6.0`  |
| [windows-2019] | Visual Studio 16 2019 | >= `10.1.243` |

Deprecated/Removed Runners previously supported:

| Runner | Host Compiler | CUDA |
|--------|------|---------------|
| [ubuntu-1804] | GCC 10 | >= `11.4`  (`11.4.1`) |
| [ubuntu-1804] | GCC 6 - 9 | >= `10.0` |

[ubuntu-2204]: https://github.com/actions/runner-images/blob/main/images/linux/Ubuntu2204-Readme.md
[ubuntu-2004]: https://github.com/actions/runner-images/blob/main/images/linux/Ubuntu2004-Readme.md
[ubuntu-1804]: https://github.com/actions/runner-images/blob/main/images/linux/Ubuntu1804-Readme.md
[windows-2022]: https://github.com/actions/runner-images/blob/main/images/win/Windows2022-Readme.md
[windows-2019]: https://github.com/actions/runner-images/blob/master/images/win/Windows2019-Readme.md

## Sample application

To ensure the installed compilers are usable, a very simple CUDA C++ test application is included, requiring CMake >= 3.10 for native CUDA support. 3.18+ has much improved CUDA support.

It simply prints `hello world` from the host and from a single thread on the device.

This does not specify any cuda architectures to target, using the nvcc defaults. From CMake 3.18, use `CMAKE_CUDA_ARCHITECTURES`.

### Compilation

```bash
mkdir -p build && cd build
cmake .. 
cmake --build .
```

### Execution

```bash
cd build
./main
```
