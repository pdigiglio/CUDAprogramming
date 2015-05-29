#ifndef CUDAINTEGRATOR_H_
#define CUDAINTEGRATOR_H_

#include <stdio.h>

__global__ void trial ();

template <size_t D, typename T>
__host__ __device__ 
void distance ( const T *x_i, const T *x_j, T *x_ij ) {
    x_ij[0] = x_i[0] - x_j[0];
    x_ij[1] = x_i[1] - x_j[1];
    x_ij[2] = x_i[2] - x_j[2];
};

template <size_t N, size_t D, typename T>
__global__ 
void cudaLeapFrogVerlet( T* x, T* v ) {
    unsigned threadID = blockDim.x * blockIdx.x + threadIdx.x;

    printf( "Args [%u]: x %p y %p N %zu D %zu\n", threadID, (void*) x, (void*) v, N , D );
    
};


#endif /* CUDAINTEGRATOR_H_ */

