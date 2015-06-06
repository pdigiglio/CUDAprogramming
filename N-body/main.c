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

void cudaCheckError( cudaError_t errorCode ) {
//	errorCode = cudaGetLastError();

	if( errorCode != cudaSuccess ) {
		fprintf( stderr, "%s\n", cudaGetErrorString( errorCode ) );
		exit( EXIT_FAILURE );
	}
}

void cudaPrintDeviceInfo() {
	/* taken from 0_Simple/cudaOpenMp/cudaOpenMP.cu */
    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
	
	int numGPUs = 0;
	cudaGetDeviceCount( &numGPUs );
    if ( numGPUs < 1 ) {
        fprintf( stderr, "no CUDA capable devices were detected\n");
        exit( EXIT_FAILURE );
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
}
	int
main () {

//    printf("%s Starting...\n\n", argv[0]);
    cudaPrintDeviceInfo();


	const size_t xEntries    = BLOCK_SIZE;
	const size_t xMemorySize = xEntries * sizeof( float );
	float *x = (float *) calloc( xEntries, xMemorySize );

	const size_t vEntries    = BLOCK_SIZE;
	const size_t vMemorySize = vEntries * sizeof( float );
	float *v = (float *) calloc( vEntries, vMemorySize );

	/* variable to control errors in CUDA calls */
	cudaError_t errorCode = cudaSuccess;

    // allocate device memory
    float  *device_x = NULL;
    errorCode = cudaMalloc( &device_x, xMemorySize );
    cudaCheckError( errorCode );

    float  *device_v = NULL;
    errorCode = cudaMalloc( &device_v, vMemorySize );
    cudaCheckError( errorCode );

    // copy meory from host to device
    errorCode = cudaMemcpy( device_x, x, xMemorySize, cudaMemcpyHostToDevice );
    cudaCheckError( errorCode );
    errorCode = cudaMemcpy( device_v, v, vMemorySize, cudaMemcpyHostToDevice );
    cudaCheckError( errorCode );


    /* event to get CUDA execution time */
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    /* TODO checkError */
    cudaEventCreate( &stop );
    /* TODO checkError */

    /* record start (0 = default stream) */
    cudaEventRecord( start, 0 );

    // variables to control block and grid dimension
    dim3  dimBlock( BLOCK_SIZE /*, BLOCK_SIZE */ );
    dim3   dimGrid( GRID_SIZE,  GRID_SIZE );
//	trial <<< dimGrid, dimBlock >>> ();

	for ( unsigned t = 0; t < 10; t += 10 ) {

		for ( unsigned int j = 0; j < 1; ++ j ) {
            cudaLeapFrogVerlet<1,1,float> <<< dimGrid, dimBlock >>> ( device_x, device_v );
	//		leapfrogVerlet < numOfParticles, spaceDimension > ( x, v );
	//		rungeKutta( x, v, numOfParticles );
		}

//		printf( "%u\t", t );
//		for( unsigned int i = 0; i < numOfParticles * spaceDimension;  i += 6 ) {
//			printf( "%.6g\t%.6g\t%.6g\t", x[i ], x[i+1], x[i+2] );
//			printf( "%.6g\t%.6g\t%.6g\t"  , x[i+3], x[i+4], x[i+5] );
//		}
//		printf( "\n" );
	}

    errorCode = cudaGetLastError();
    cudaCheckError( errorCode );

    // copy meory from host to device
    errorCode = cudaMemcpy( x, device_x, xMemorySize, cudaMemcpyDeviceToHost );
    cudaCheckError( errorCode );
    errorCode = cudaMemcpy( v, device_v, vMemorySize, cudaMemcpyDeviceToHost );
    cudaCheckError( errorCode );

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

    for( size_t i = 0; i < xEntries; ++ i ) {
        printf( "x[ %zu ] = %g\n", i, x[i] );
    }

	cudaDeviceSynchronize();

    errorCode = cudaFree( device_x );
    cudaCheckError( errorCode );

    errorCode = cudaFree( device_v );
    cudaCheckError( errorCode );

    free( x );
    free( v );

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
