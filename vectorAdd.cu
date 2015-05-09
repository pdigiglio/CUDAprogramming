/**
 * @file vectorAdd.cu
 */

#include <stdio.h>
#include <time.h>

#include <cuda_runtime.h>

#define VECTOR_SIZE 100000

    __global__ void
kernelVecAdd ( const double *a, const double *b, double *c, size_t size ) {
    /* get position of thread */
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

    /**
     * @attention CUDA doesn't support the specifier `%zu` for type `size_t`!
     */
//    printf( "Kernel launched with index %u (%lu)\n", i, (long unsigned) size );

    if ( i < size )
        c[i] = a[i] + b[i];
}
    int
main () {
    
    //--------------------------------------------------------------------------------------------------

    /* allocate host memory */
    double *hostA = (double *) malloc( VECTOR_SIZE * sizeof(double) );
    if( ! hostA ) {
        fprintf( stderr, "hostA: allocation failed\n" );
        exit( EXIT_FAILURE );
    }

    double *hostB = (double *) malloc( VECTOR_SIZE * sizeof(double) );
    if( ! hostB ) {
        fprintf( stderr, "hostB: allocation failed\n" );
        exit( EXIT_FAILURE );
    }


    double *hostC = (double *) malloc( VECTOR_SIZE * sizeof(double) );
    if( ! hostC ) {
        fprintf( stderr, "hostC: allocation failed\n" );
        exit( EXIT_FAILURE );
    }

    double *hostC_check = (double *) malloc( VECTOR_SIZE * sizeof(double) );
    if( ! hostC_check ) {
        fprintf( stderr, "hostC_check: allocation failed\n" );
        exit( EXIT_FAILURE );
    }

    /* initialize values */
    for ( unsigned int j = 0; j < VECTOR_SIZE; ++ j ) {
        hostA[j] = (double) rand() / RAND_MAX;
        hostB[j] = (double) rand() / RAND_MAX;
    }

    //-----------------------------------------------------------------------------------------------

    time_t cudaVecAddTime = clock();

    /* allocate device memory */

    cudaError_t errorCode = cudaSuccess;

    double *deviceA = NULL;
    errorCode = cudaMalloc( (void **)&deviceA, VECTOR_SIZE * sizeof(double) );
    if ( errorCode != cudaSuccess ) {
        fprintf( stderr, "deviceA: allocation failed (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }

    double *deviceB = NULL;
    errorCode = cudaMalloc( (void **)&deviceB, VECTOR_SIZE * sizeof(double) );
    if ( errorCode != cudaSuccess ) {
        fprintf( stderr, "deviceB: allocation failed (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }

    double *deviceC = NULL;
    errorCode = cudaMalloc( (void **)&deviceC, VECTOR_SIZE * sizeof(double) );
    if ( errorCode != cudaSuccess ) {
        fprintf( stderr, "deviceC: allocation failed (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }

    fprintf( stderr, "Copying host data into device data\n" );

    errorCode = cudaMemcpy( deviceA, hostA, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice );
    if( errorCode != cudaSuccess ) {
        fprintf( stderr, "copying hostA -> deviceA failed (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }

    errorCode = cudaMemcpy( deviceB, hostB, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice );
    if( errorCode != cudaSuccess ) {
        fprintf( stderr, "copying hostA -> deviceB failed (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }

    const unsigned int threadsPerBlock = 256;
    /**
     * @attention `VECTOR_SIZE / threadsPerBlock` would round the value to the closest integer from below thus resulting
     * in the tail of the vector not being processed.
     *
     * Possible solutions:
     * * `ceil()`;
     * * `( VECTOR_SIZE + threadsPerBlock - 1) / threadsPerBlock`: if I didn't put -1 the result would 
     * always be bigger than needed by one unit;
     */
    const unsigned int blocksPerGrid = ( VECTOR_SIZE + threadsPerBlock - 1 ) / threadsPerBlock;

    fprintf( stderr, "CUDA kernel launch with %u blocks of %u threads\n", blocksPerGrid, threadsPerBlock );

    kernelVecAdd <<< blocksPerGrid,threadsPerBlock>>> ( deviceA, deviceB, deviceC, VECTOR_SIZE );
    errorCode = cudaGetLastError();

    if( errorCode != cudaSuccess ) {
        fprintf( stderr, "failed to launch CUDA kernelVecAdd() (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }


    fprintf( stderr, "copying device result back\n" );
    errorCode = cudaMemcpy( hostC, deviceC, VECTOR_SIZE * sizeof( double ), cudaMemcpyDeviceToHost );
    if( errorCode != cudaSuccess ) {
        fprintf( stderr, "copying deviceC -> hostC failed (error: %s)\n", cudaGetErrorString( errorCode ) );
        exit( EXIT_FAILURE );
    }

    fprintf( stderr, "CUDA time: %g\n", ( (double) - cudaVecAddTime + clock() ) / CLOCKS_PER_SEC );

    time_t cpuVecAddTime = clock();
    for ( unsigned int j = 0; j < VECTOR_SIZE; j ++ ) {
        hostC_check[j] = hostA[j] + hostB[j];
    }
    fprintf( stderr, "CPU time: %g\n", ( (double) - cpuVecAddTime + clock() ) / CLOCKS_PER_SEC );

    fprintf( stderr, "chech for errors\n" );

    time_t checkTime = clock();
    for ( unsigned int j = 0; j < VECTOR_SIZE; j ++ ) {
        if ( hostC_check[j] != hostC[j] )
            fprintf( stderr, "error found at index %u\n", j );
    }
    fprintf( stderr, "CPU check time: %g\n", ( (double) - checkTime + clock() ) / CLOCKS_PER_SEC );

    return 0;
}
