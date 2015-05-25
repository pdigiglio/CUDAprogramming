#include "integrator.h"
#include "cudaIntegrator.h"

#include <stdlib.h>
#include <stdio.h>

#include <omp.h>

#include <cuda_runtime.h>
// header located in /usr/local/cuda/samples/common/inc
#include <helper_cuda.h>

//const unsigned int numOfParticles = 1;

// TODO consider accepting these parameters as cmd-line args
const unsigned short int BLOCK_SIZE = 16;
const unsigned short int  GRID_SIZE = 1;
	int
main () {

//    printf("%s Starting...\n\n", argv[0]);

	/* taken from 0_Simple/cudaOpenMp/cudaOpenMP.cu */
    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
	
	int numGPUs = 0;
	cudaGetDeviceCount( &numGPUs );
    if ( numGPUs < 1 ) {
        fprintf( stderr, "no CUDA capable devices were detected\n");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", numGPUs);

    for ( int i = 0; i < numGPUs; ++ i) {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");


//	float x[numOfParticles] = {};
//	float v[numOfParticles] = {};

    /* event to get CUDA execution time */
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    /* TODO checkError */
    cudaEventCreate( &stop );
    /* TODO checkError */

	/* variable to control errors in CUDA calls */
	cudaError_t errorCode = cudaSuccess;

    /* record start (0 = default stream) */
    cudaEventRecord( start, 0 );

    // variables to control block and grid dimension
    dim3  dimBlock( BLOCK_SIZE, BLOCK_SIZE );
    dim3   dimGrid( GRID_SIZE,  GRID_SIZE );
	trial <<< dimGrid, dimBlock >>> ();
	errorCode = cudaGetLastError();
	if( errorCode != cudaSuccess ) {
		fprintf( stderr, "%s\n", cudaGetErrorString( errorCode ) );
		exit( EXIT_FAILURE );
	}

    /* record stop on the same stream as start */
    cudaEventRecord( stop, 0 );
    /* wait till every thread is done */
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );


    fprintf( stderr, "CUDA kernel execution time: %g ms\n", elapsedTime );

    cudaEventDestroy( start );
    /* TODO checkError */
    cudaEventDestroy( stop );
    /* TODO checkError */

//	for ( unsigned t = 0; t < 1000000; ++ t ) {
//		leapfrogVerlet( x, v, numOfParticles );
//		rungeKutta( x, v, numOfParticles );
//		printf( "%u %.6g %.6g\n", t, x[0], v[0] );
//	}

	cudaDeviceSynchronize();

	/*
     * `cudaDeviceReset()` causes the driver to clean up all state. While
     * not mandatory in normal operation, it is good practice.  It is also
     * needed to ensure correct operation when the application is being
     * profiled. Calling `cudaDeviceReset()` causes all profile data to be
     * flushed before the application exits.
	 */
    cudaDeviceReset();

	return 0;
}
