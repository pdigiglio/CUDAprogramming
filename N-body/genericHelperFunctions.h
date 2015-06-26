/**
 *
 *
 *           @name  genericHelperFunctions.h
 *          @brief  Some helper function which will be used in main ()
 *
 *        @version  1.0
 *           @date  06/21/2015 (07:43:05 PM)
 *       @revision  none
 *       @compiler  gcc
 *
 *
 *         @author  P. Di Giglio (github.com/pdigiglio), <p.digiglio91@gmail.com>
 *        @company  
 *
 *          Example usage:
 *          @code
 *          @endcode
 *
 *
 */

#ifndef GENERICHELPERFUNCTIONS_H_
#define GENERICHELPERFUNCTIONS_H_

/**
 * @brief Checks if there were errors in calling a CUDA function
 * 
 * To actually "enable" this function the program must be compiled with either
 * `-DDEBUG` or `-D_DEBUG` flags.
 *
 * @return the error code passed as argument
 */

#include <stdio.h>
#include <assert.h>

#include <cuda_runtime.h>
// header located in /usr/local/cuda/samples/common/inc
#include <helper_cuda.h>

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
};

/**
 * @brief Print GPU info and number of CPU cores
 *
 * The function has been taken from `0_Simple/cudaOpenMp/cudaOpenMP.cu`.
 */
void cudaPrintDeviceInfo( FILE *stream = stderr );

/**
 * @brief Template to copy initial configuration from Host to Device
 */
template <typename T, unsigned short D, size_t numOfParticles >
inline void copyConfigurationToDevice (
		const T *x, T **device_x, size_t xMemorySize,
		const T *v, T **device_v, size_t vMemorySize,
		const T *m, T **device_m, size_t mMemorySize ) {

	fprintf( stderr, "Copying memory from Host to Device... "/*, numOfParticles*/ );

	cudaError_t errorCode = cudaSuccess;

    // allocate device memory
    errorCode = cudaMalloc( device_x, xMemorySize );
    cudaCheckError( errorCode );

    errorCode = cudaMalloc( device_v, vMemorySize );
    cudaCheckError( errorCode );

	errorCode = cudaMalloc( device_m, mMemorySize );
	cudaCheckError( errorCode );

    // copy memory from host to device
    errorCode = cudaMemcpy( *device_x, x, xMemorySize, cudaMemcpyHostToDevice );
    cudaCheckError( errorCode );
    errorCode = cudaMemcpy( *device_v, v, vMemorySize, cudaMemcpyHostToDevice );
    cudaCheckError( errorCode );
    errorCode = cudaMemcpy( *device_m, m, mMemorySize, cudaMemcpyHostToDevice );
    cudaCheckError( errorCode );

	fprintf( stderr, "done!\n" );
}

/**
 * @brief Prints a vector as a column (a row for each entry)
 */
template <size_t N, size_t D, typename T>
inline void printVectorAsColumn( const T *x, FILE *stream = stdout ) {
	for( size_t i = 0; i < N * D; ++ i ) {
		fprintf( stream, "%.6g\t", x[i] );
	}
}

/**
 * @brief Prints a vector as a three-column CSV with an header.
 *
 * Three columns are printed, separated by a comma and _no spaces_.
 * Also a header is printed (i.e. "x,y,z" is printed at the beginning
 * of the vector).
 */
template<size_t N, size_t D, typename T>
inline void printVectorAsCSV( const T *x, FILE *stream = stdout ) {
    // print header
    fprintf( stream, "x,y,z\n" );

    for ( size_t i = 0; i < D * N; i += 6 ) {
        fprintf( stream, "%.6g,%.6g,%.6g\n", x[i  ], x[i+1], x[i+2] );
        fprintf( stream, "%.6g,%.6g,%.6g\n", x[i+3], x[i+4], x[i+5] );
    }
}

/**
 * @brief Prints a vector as a row.
 */
template<size_t N, size_t D, typename T>
inline void printVectorAsRow( const T *x, unsigned int t, FILE *stream = stdout ) {
		printf( "%u\t", t );
		for( unsigned int i = 0; i < D * N; i += 6 ) {
			printf( "%.6g\t%.6g\t%.6g\t", x[i  ], x[i+1], x[i+2] );
			printf( "%.6g\t%.6g\t%.6g\t", x[i+3], x[i+4], x[i+5] );
		}
		printf( "\n" );
}

#endif /* GENERICHELPERFUNCTIONS_H_ */
