/**
 *
 *
 *           @name  cudaArrayOperationsHelper.h
 *
 *        @version  1.0
 *           @date  06/18/2015 (03:21:28 PM)
 *       @revision  none
 *       @compiler  gcc
 *
 *
 *         @author  P. Di Giglio (github.com/pdigiglio), <p.digiglio91@gmail.com>
 *        @company  
 *          @brief  
 *
 *          Example usage:
 *          @code
 *          @endcode
 *
 *
 */

#ifndef  CUDAARRAYOPERATIONSHELPER_INC
#define  CUDAARRAYOPERATIONSHELPER_INC

/**
 * @brief Fetch memory from global to shared.
 *
 * In principle that's just a memory copy, irrespective of whether the memory
 * is copyed from or to the gobal memory.
 */
template <size_t D,typename T>
__device__
inline void fetchFromGlobalMemory ( T *mySharedArray, const T *x ) {
	for( size_t d = 0; d < D; ++ d )
		mySharedArray[d] = x[d];
}

/**
 * @brief Helper function to copy memory.
 *
 * This is the same function as `fetchFromGlobalMemory()` but I call it differently
 * to keep the m logically separated.
 */
template <size_t D,typename T>
__device__
inline void writeToGlobalMemory ( T *x, const T *mySharedArray ) {
	fetchFromGlobalMemory<D>( x, mySharedArray );
}


#endif   /* ----- #ifndef CUDAARRAYOPERATIONSHELPER_INC  ----- */
