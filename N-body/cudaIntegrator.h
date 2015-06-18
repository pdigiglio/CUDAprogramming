#ifndef CUDAINTEGRATOR_H_
#define CUDAINTEGRATOR_H_

#include <stdio.h>
#include "cudaArrayOperationsHelper.h"

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
	__shared__ T    evolvingParticle[ D * 32 ];
//	__shared__ T surroungingParticle[ D * 32 ];

    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	fetchFromGlobalMemory <D> ( evolvingParticle + i, x + i ); 

	evolvingParticle[i] ++;

	writeToGlobalMemory<D>( x + i, evolvingParticle + i );



//    printf( "Args [%u]: x %p y %p N %zu D %zu\n", threadID, (void*) x, (void*) v, N , D );
    // if you do like that then it may be that some other thread
    // writes in-between
//    printf( "block %u (of %u), %u\n", blockIdx.x, gridDim.x, threadIdx.x );
//    printf( "block %u (of %u), %u\n", blockIdx.y, gridDim.y, threadIdx.y );
//    printf( "block %u (of %u), %u\n", blockIdx.z, gridDim.z, threadIdx.z );

};


#endif /* CUDAINTEGRATOR_H_ */

