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
inline void fetchFromGlobalMemory ( T *__restrict__ mySharedArray, const T *__restrict__ x ) {
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

/**
 * @brief Helper function to set vector to zero.
 */
template <size_t D, typename T>
__device__
inline void setVectorToZero( T *x ) {
    for( size_t d = 0; d < D; ++ d )
        x[d] = (T) 0;
}

/*
 * @brief Specialized template for \f$D=3\f$.
 */
//template <size_t D, typename T>
//__device__
//void setVectorToZero( T *x ) {
//	x[0] = (T) 0;
//	x[1] = (T) 0;
//	x[2] = (T) 0;
//}

/**
 * @brief Stores distance among `x_i` and `x_j' into `x_ij`
 */
template <size_t D, typename T>
__device__ 
inline void distance ( const T *x_i, const T *x_j, T *x_ij ) {
    for( size_t d = 0; d < D; ++ d )
        x_ij[d] = x_i[d] - x_j[d];

//    x_ij[0] = x_i[0] - x_j[0];
//    x_ij[1] = x_i[1] - x_j[1];
//    x_ij[2] = x_i[2] - x_j[2];
};

#endif   /* ----- CUDAARRAYOPERATIONSHELPER_INC  ----- */
