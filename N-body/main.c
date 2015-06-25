//#include "integrator.h"
#include "cudaIntegrator.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <errno.h>

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
const unsigned int BLOCK_SIZE     = 32 * 32;
const unsigned int GRID_SIZE      = 4;

/**
 * @brief Number of particles in the system.
 *
 * @attention This __must be a multiple of 4__ since in later functions a 4x manual loop 
 * unrolling is performed.
 */
const unsigned int numOfParticles = GRID_SIZE * BLOCK_SIZE; /* XXX this must be even! */

	int
main ( int argc, char *argv[] ) {

    // random seed for random generator
    srand( time( NULL ) );

	if ( argc > 1 )
		fprintf( stderr, "Too many arguments: program doesn't accept any!\n" );

	fprintf(stderr, "%s Starting...\n\n", argv[0]);;
    cudaPrintDeviceInfo();

    // allocate host memory
	double *x = NULL, *v = NULL, *m = NULL;
	initializeSystem < double, spaceDimension, numOfParticles > ( x, v, m );

	size_t xMemorySize = spaceDimension * numOfParticles * sizeof( x[0] );
	size_t vMemorySize = spaceDimension* numOfParticles * sizeof( v[0] );
	size_t mMemorySize = numOfParticles * sizeof( m[0] );

    // allocate device memory and copy host memory
	double *device_x = NULL, *device_v = NULL, *device_m = NULL;
	copyConfigurationToDevice < double, spaceDimension, numOfParticles > (
			x, &device_x, xMemorySize,
			v, &device_v, vMemorySize,
			m, &device_m, mMemorySize );

	// variable to control errors in CUDA calls
	cudaError_t errorCode = cudaSuccess;

    // event to get CUDA execution time
    cudaEvent_t start, stop;
    cudaCheckError( cudaEventCreate( &start ) );
    cudaCheckError( cudaEventCreate( &stop ) );

    // record start (0 = default stream)
    cudaEventRecord( start, 0 );

    // variables to control block and grid dimension
    dim3  dimBlock( BLOCK_SIZE );
    dim3   dimGrid( GRID_SIZE  );

    char outFileName[80];
    FILE *outFile = NULL;

	const unsigned int MaxNumberOfTimeSteps = 10000;
	const unsigned int TimeStepIncrement    = 5;
	for ( unsigned t = 0; t < MaxNumberOfTimeSteps; t += TimeStepIncrement ) {

		fprintf( stderr, "Evolving particles... [step %u of %u]\r", t , MaxNumberOfTimeSteps );

        sprintf( outFileName, "grav%06u.csv", t / TimeStepIncrement );
        outFile = fopen( outFileName, "w" );
        if( !outFile ) {
            fprintf ( stderr, "couldn't open file '%s'; %s\n",
                    outFileName, strerror(errno) );
            exit (EXIT_FAILURE);
        }

        printVectorAsCSV<numOfParticles,spaceDimension>( x, outFile );

        if( fclose( outFile ) == EOF ) {         /* close output file   */
            fprintf ( stderr, "couldn't close file '%s'; %s\n",
                    outFileName, strerror(errno) );
            exit (EXIT_FAILURE);
        }

		for ( unsigned int j = 0; j < TimeStepIncrement; ++ j ) {
			cudaUpdateSystemGlobalPositions<numOfParticles,spaceDimension,double> <<<dimGrid,dimBlock>>>( device_x, device_v );
			// implicit synchronization here!!
			cudaLeapFrogVerlet<numOfParticles,spaceDimension,double> <<< dimGrid, dimBlock, (spaceDimension + 1) * BLOCK_SIZE * sizeof( double ) >>> ( device_x, device_v, device_m );
		}

		// collect results
		errorCode = cudaMemcpy( x, device_x, xMemorySize, cudaMemcpyDeviceToHost );
		cudaCheckError( errorCode );
	}

	fprintf( stderr, "Evolving particles... done!                                 \n" );

    /* record stop on the same stream as start */
    cudaEventRecord( stop, 0 );
    /* wait till every thread is done */
    cudaEventSynchronize( stop );
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop );

    fprintf( stderr, "CUDA kernel execution time: %g ms\n", elapsedTime );

//    char time_file_name[80]; // = "timeDouble_32x32_2.txt";       /* output-file name    */
//    sprintf( time_file_name, "timeDouble_%u_%u.txt", BLOCK_SIZE, GRID_SIZE );
//    FILE *timeOutputFile  = fopen( time_file_name, "a" );   /* output-file pointer */
//
//    if ( timeOutputFile == NULL ) {
//            fprintf ( stderr, "couldn't open file '%s'; %s\n",
//                                time_file_name, strerror(errno) );
//                exit (EXIT_FAILURE);
//    }
//
//    fprintf( timeOutputFile, "%g\n", elapsedTime / 1000 );
//
//    if( fclose( timeOutputFile ) == EOF ) {         /* close output file   */
//            fprintf ( stderr, "couldn't close file '%s'; %s\n",
//                                time_file_name, strerror(errno) );
//                exit (EXIT_FAILURE);
//    }


    cudaCheckError( cudaEventDestroy( start ) );
    cudaCheckError( cudaEventDestroy( stop ) );

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
