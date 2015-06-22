//#include "integrator.h"
#include "cudaIntegrator.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <omp.h>

#include <cuda_runtime.h>
// header located in /usr/local/cuda/samples/common/inc
#include <helper_cuda.h>

#include "systemInitializationHelper.h"
#include "genericHelperFunctions.h"

/**
 * @brief Number of space dimensions.
 */
const unsigned int spaceDimension = 3;

/**
 * @brief Size of the blocks
 */
const unsigned int BLOCK_SIZE     = 32;
const unsigned int GRID_SIZE      = 1;

/**
 * @brief Number of particles in the system.
 *
 * @attention This __must be a multiple of 4__ since in later functions a 4x manual loop 
 * unrolling is performed.
 */
const unsigned int numOfParticles = GRID_SIZE * BLOCK_SIZE; /* XXX this must be even! */

template <size_t N, typename T>
void printVector( const T *x, FILE *stream = stdout ) {
	for( size_t i = 0; i < N; ++ i ) {
		fprintf( stream, "%.6g\t", x[i] );
	}
}

	int
main ( int argc, char *argv[] ) {

	if ( argc > 1 )
		fprintf( stderr, "Too many arguments: program doesn't accept any!\n" );

	fprintf(stderr, "%s Starting...\n\n", argv[0]);;
    cudaPrintDeviceInfo();

	float *x = NULL, *v = NULL, *m = NULL;
	initializeSystem < float, spaceDimension, numOfParticles > ( x, v, m );

//	printVector<spaceDimension * numOfParticles>(x);
//	return 0;

	size_t xMemorySize = spaceDimension * numOfParticles * sizeof( x[0] );
	size_t vMemorySize = spaceDimension* numOfParticles * sizeof( v[0] );
	size_t mMemorySize = numOfParticles * sizeof( m[0] );

	float *device_x = NULL, *device_v = NULL, *device_m = NULL;
	copyConfigurationToDevice < float, spaceDimension, numOfParticles > (
			x, &device_x, xMemorySize,
			v, &device_v, vMemorySize,
			m, &device_m, mMemorySize );

	// just for debug: to check they're not NULL
//	printf( "%p %p %p\n", (void *) device_x, (void *) device_v, (void *) device_m );

	/* variable to control errors in CUDA calls */
	cudaError_t errorCode = cudaSuccess;

    /* event to get CUDA execution time */
    cudaEvent_t start, stop;
    cudaCheckError( cudaEventCreate( &start ) );
    cudaCheckError( cudaEventCreate( &stop ) );

    /* record start (0 = default stream) */
    cudaEventRecord( start, 0 );

    // variables to control block and grid dimension
    dim3  dimBlock( BLOCK_SIZE /*, BLOCK_SIZE */ );
    dim3   dimGrid( GRID_SIZE /*,  GRID_SIZE */ );

	const unsigned int MaxNumberOfTimeSteps = 10000;
	const unsigned int TimeStepIncrement    = 1;
	for ( unsigned t = 0; t < MaxNumberOfTimeSteps; t += TimeStepIncrement ) {

		fprintf( stderr, "Evolving particles... [step %u of %u]\r", t , MaxNumberOfTimeSteps );

		printf( "%u\t", t );
		for( unsigned int i = 0; i < numOfParticles * spaceDimension;  i += 6 ) {
			printf( "%.6g\t%.6g\t%.6g\t", x[i  ], x[i+1], x[i+2] );
			printf( "%.6g\t%.6g\t%.6g\t", x[i+3], x[i+4], x[i+5] );
		}
		printf( "\n" );

		for ( unsigned int j = 0; j < TimeStepIncrement; ++ j ) {
			cudaUpdateSystemGlobalPositions<numOfParticles,spaceDimension,float> <<<dimGrid,dimBlock>>>( device_x, device_v );
			// implicit synchronization here!!
			cudaLeapFrogVerlet<numOfParticles,spaceDimension,float> <<< dimGrid, dimBlock, 10 * BLOCK_SIZE * sizeof( float ) >>> ( device_x, device_v, device_m );
		}

		// collect results
		errorCode = cudaMemcpy( x, device_x, xMemorySize, cudaMemcpyDeviceToHost );
		cudaCheckError( errorCode );
	}

	fprintf( stderr, "Evolving particles... done!                                 \n" );
//    errorCode = cudaGetLastError();
//    cudaCheckError( errorCode );

    // copy meory from host to device
//    errorCode = cudaMemcpy( x, device_x, xMemorySize, cudaMemcpyDeviceToHost );
//    cudaCheckError( errorCode );
//    errorCode = cudaMemcpy( v, device_v, vMemorySize, cudaMemcpyDeviceToHost );
//    cudaCheckError( errorCode );
	// XXX there is no need to copy back mass vector

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

//    for( size_t i = 0; i < spaceDimension * numOfParticles; ++ i ) {
//        printf( "x[ %zu ] = %g\n", i, x[i] );
//    }

	// XXX this shouldn't be needed because of the previous sync
	cudaDeviceSynchronize();

	// free device memory
	fprintf( stderr, "Freeing Device memory... " );
    errorCode = cudaFree( device_x );
    cudaCheckError( errorCode );

    errorCode = cudaFree( device_v );
    cudaCheckError( errorCode );

	errorCode = cudaFree( device_m );
	cudaCheckError( errorCode );
	fprintf( stderr, "done!\n" );


	// free host memory
	fprintf( stderr, "Freeing Host memory... " );
    free( x );
    free( v );
	free( m );
	fprintf( stderr, "done!\n" );

	/*
     * `cudaDeviceReset()` causes the driver to clean up all state. While
     * not mandatory in normal operation, it is good practice.  It is also
     * needed to ensure correct operation when the application is being
     * profiled. Calling `cudaDeviceReset()` causes all profile data to be
     * flushed before the application exits.
	 */
	fprintf( stderr, "Reset Device memory... " );
    cudaDeviceReset();
	fprintf( stderr, "done!\n" );

	fprintf( stderr, "\nGoodbye!\n" );
	return 0;
}
