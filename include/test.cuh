#pragma once

#include <stdio.h>

// example function from a header.

__global__ void helloWorldDevice(){
    double x = 0;
    if(threadIdx.x + blockIdx.x * blockDim.x < 1){
        printf("Hello World from the device\n");
        atomicAdd(&x, 1);
    }
}

void helloWorld(){
    printf("helloWorld Host\n");

    helloWorldDevice<<<1, 1>>>();
    cudaDeviceSynchronize();
}