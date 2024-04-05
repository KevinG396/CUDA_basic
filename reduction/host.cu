#include <stdio.h>
#include <cuda_runtime.h>
#include "kernel.h"

int main() {
    int size = 1024;
    int bytes = size * sizeof(int); // array mem use
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    int *h_idata = (int*) malloc(bytes); // inp host
    int *h_odata = (int*) malloc(blocksPerGrid*sizeof(int)); // oup host
    int *d_idata, *d_odata;
    int sum = 0;
    // initialize
    for(int i = 0; i < size; i++) {
        h_idata[i] = 1;
    }
    // allocate mem on device
    cudaMalloc((void**)&d_idata, bytes);
    cudaMalloc((void**)&d_odata, blocksPerGrid*sizeof(int));

    // cp from host 2 device
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);

    // execute kernel
    //int threadsPerBlock = 256; // 256 threads per block
    //int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    reduce0<<<blocksPerGrid, threadsPerBlock>>>(d_idata, d_odata);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    //  cp result from device 2 host
    cudaMemcpy(h_odata, d_odata, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    // finish reduction

    for(int i = 0; i < blocksPerGrid; i++) {
        printf("sum: %d\n", sum);
        sum += h_odata[i];
    }

    printf("Total Sum = %d\n", sum);

    // free
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}