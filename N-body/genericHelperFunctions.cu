/**
 *
 *
 *           @name  genericHelperFunctions.cu
 *          @brief  Implementation file for header `genericHelperFunctions.h`
 *
 *        @version  1.0
 *           @date  06/21/2015 (07:45:30 PM)
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

#include <stdlib.h>
#include <stdio.h>

#include <omp.h>
#include <cuda_runtime.h>
// header located in /usr/local/cuda/samples/common/inc
#include <helper_cuda.h>

#include "genericHelperFunctions.h"

/**
 * @brief Print GPU info and number of CPU cores
 *
 * The function has been taken from `0_Simple/cudaOpenMp/cudaOpenMP.cu`.
 */
void cudaPrintDeviceInfo( FILE *stream ) {
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
