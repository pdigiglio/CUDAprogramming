/**
 * @file main.c
 * @brief
 */

#include "integrator.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

const unsigned int spaceDimension = 3;
const unsigned int numOfParticles = 2; /* XXX this must be even! */

	int
main ( int argc, char *argv[] ) {
	
	printf("%s Starting...\n\n", argv[0]);

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

	double x[ spaceDimension * numOfParticles ] = {
		1., 0., 0.,
//		-1., 0., 0.,
//		1., 0., 0.,
		-1., 0., 0.
	};
	double v[ spaceDimension * numOfParticles ] = {
		0., 1., 0.,
//		0., 0., 0.,
//		0., 1., 0.,
		0., 0., 0.
	};

	for ( unsigned t = 0; t < 1000; t += 10 ) {

		for ( unsigned int j = 0; j < 100; ++ j ) {
			leapfrogVerlet < numOfParticles, spaceDimension > ( x, v );
	//		rungeKutta( x, v, numOfParticles );
		}

		printf( "%u ", t );
		for( unsigned int i = 0; i < numOfParticles * spaceDimension;  i += 6 ) {
			printf( "%.6g %.6g %.6g ", x[i ], x[i+1], x[i+2] );
			printf( "%.6g %.6g %.6g ", x[i+3], x[i+4], x[i+5] );
		}
		printf( "\n" );
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
