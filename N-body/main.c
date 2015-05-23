#include "integrator.h"

#include <stdlib.h>
#include <stdio.h>

const unsigned int numOfParticles = 1;

	int
main () {

//    printf("%s Starting...\n\n", argv[0]);

	/* taken from 0_Simple/cudaOpenMp/cudaOpenMP.cu
    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
	
	cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");

	*/

	float x[numOfParticles] = {};
	float v[numOfParticles] = {};

	for ( unsigned t = 0; t < 1000000; ++ t ) {
//		leapfrogVerlet( x, v, numOfParticles );
		rungeKutta( x, v, numOfParticles );
		printf( "%u %.6g %.6g\n", t, x[0], v[0] );
	}

	/*
     * `cudaDeviceReset()` causes the driver to clean up all state. While
     * not mandatory in normal operation, it is good practice.  It is also
     * needed to ensure correct operation when the application is being
     * profiled. Calling `cudaDeviceReset()` causes all profile data to be
     * flushed before the application exits.
	 */
//    cudaDeviceReset();

	return 0;
}
