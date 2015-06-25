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


#endif /* GENERICHELPERFUNCTIONS_H_ */
