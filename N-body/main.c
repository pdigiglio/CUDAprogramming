#include "integrator.h"
#include "cudaIntegrator.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include <omp.h>

#include <cuda_runtime.h>
// header located in /usr/local/cuda/samples/common/inc
#include <helper_cuda.h>

//const unsigned int numOfParticles = 1;

// TODO consider accepting these parameters as cmd-line args
const unsigned short int BLOCK_SIZE = 32;
const unsigned short int  GRID_SIZE = 1;

const size_t numOfStreams = BLOCK_SIZE;

/**
 * @brief Checks if there were errors in calling a CUDA function
 * 
 * To actually "enable" this function the program must be compiled with either
 * `-DDEBUG` or `-D_DEBUG` flags.
 *
 * @return the error code passed as argument
 */
    inline cudaError_t
cudaCheckError( cudaError_t errorCode ) {
//	errorCode = cudaGetLastError();

    // compile with -DDEBUG or -D_DEBUG option to enable this check
#if defined(DEBUG) || defined(_DEBUG) 
	if( errorCode != cudaSuccess ) {
		fprintf( stderr, "%s\n", cudaGetErrorString( errorCode ) );
//		exit( EXIT_FAILURE );
        assert( errorCode == cudaSuccess );
	}
#endif

    return errorCode;
}

/**
 * @brief Print GPU info and number of CPU cores
 *
 * The function has been taken from `0_Simple/cudaOpenMp/cudaOpenMP.cu`.
 */
void cudaPrintDeviceInfo( FILE *stream = stderr ) {
    // determine the number of CUDA capable GPUs
	int numGPUs = 0;
	cudaGetDeviceCount( &numGPUs );
    if ( numGPUs < 1 ) {
        fprintf( stream, "no CUDA capable devices were detected\n");
        exit( EXIT_FAILURE );
    }

    // display CPU and GPU configuration
    fprintf( stream, "number of host CPUs:\t%d\n", omp_get_num_procs());
    fprintf( stream, "number of CUDA devices:\t%d\n", numGPUs);

    for ( int i = 0; i < numGPUs; ++ i) {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        fprintf( stream, "   %d: %s\n", i, dprop.name);
    }

    fprintf( stream, "---------------------------\n");
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
    cudaCheckError( cudaEventCreate( &start ) );
    cudaCheckError( cudaEventCreate( &stop ) );

    /* record start (0 = default stream) */
    cudaEventRecord( start, 0 );

    // variables to control block and grid dimension
    dim3  dimBlock( BLOCK_SIZE /*, BLOCK_SIZE */ );
    dim3   dimGrid( GRID_SIZE,  GRID_SIZE );
//	trial <<< dimGrid, dimBlock >>> ();
//

	// --------------------------------------------------------------------------------------
	// create streams
	//
	fprintf( stderr, "Creating %zu streams... ", numOfStreams );
	cudaStream_t stream[ numOfStreams ];
	for ( size_t s = 0; s < numOfStreams; ++ s ) 
		cudaCheckError( cudaStreamCreate( &( stream[s] ) ) );
	fprintf( stderr, "done!\n" );
	// --------------------------------------------------------------------------------------

	for ( unsigned t = 0; t < 1; t += 1 ) {

		for ( unsigned int j = 0; j < 1; ++ j ) {
//			for( size_t s = 0; s < numOfStreams; ++ s )
				cudaLeapFrogVerlet<1,1,float> <<< dimGrid, dimBlock /*, 0, stream[s]*/ >>> ( device_x /* + s*/, device_v /*+ s */);
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

	// -------------------------------------------------------------------------
	// destroy streams
	//
	fprintf( stderr, "Destroying the streams... " );
	for( size_t s = 0; s < numOfStreams; ++ s ) {
		cudaCheckError( cudaStreamDestroy( stream[s] ) );
	}
	fprintf( stderr, "done!\n" );
	// -----------------------------------------------------------------------

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
