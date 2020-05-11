#pragma once

#include <stdio.h>

// example function from a header.

__global__ void helloWorldDevice(){
    if(threadIdx.x + blockIdx.x * blockDim.x < 1){
        printf("Hello World from the device\n");
    }
}

void helloWorld(){
    printf("helloWorld Host\n");

    helloWorldDevice<<<1, 1>>>();
    cudaDeviceSynchronize();
}