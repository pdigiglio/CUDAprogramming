#include <stdio.h>
#include <stdlib.h>

//#include <cuda.h>
#include <cuda_runtime.h>

/**
 * @brief Simple kernel to print out something
 * 
 * `printf()` is only supported in Compute Capability >= 2
 */
    __global__ void
helloWorldKernel () {
    printf( "Hi from CUDA: %u\n", threadIdx.x );
}

    int
main () {

    /**
     * @attention Order of threads is not kept!
     */
    helloWorldKernel <<<1,1>>> ();
    printf( "Hi from main\n" );
    helloWorldKernel <<<1,1>>> ();
    
    /** Synchronize, othrewise kernels don't print. */
    cudaDeviceSynchronize();

    return 0;
}
