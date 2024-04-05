#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce_first_add(int *g_idata, int *g_odata) {
    extern __shared__ int sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
    // do reduction in shared mem
    for(unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {  // will cause divergence, for the result of "if" is different
            sdata[tid] += sdata[tid + s];
        }
        //printf("res: %d\n", sdata[tid]);
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0){
        g_odata[blockIdx.x] = sdata[0];
    }
}