# CUDA + Cmake example using Github Actions

This repo is a very simple CUDA application, used to GitHub Actions as a CI service for CUDA compilation. 

This will **potentially** be expanded to investigate self-hosted runner(s) for running tests locally.


## Sample application.

To be representative of a real world example this should include:

+ `cmake` for cross platform build tooling
+ `nvcc` to compile .cu code. 
+ some form of test script?

## Compilation


```bash
mkdir -p build
cd build
cmake .. 
make
```

## Execution

```bash
cd build
./main
```