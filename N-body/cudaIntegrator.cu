#include <stdio.h>

//#include "integrator.h"
#include "cudaIntegrator.h"

__global__ void trial () {
	unsigned threadID = blockDim.x * blockIdx.x + threadIdx.x;

	printf( "Hello from kernel %u\n", threadID );

	__syncthreads();
}
