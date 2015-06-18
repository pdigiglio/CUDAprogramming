#ifndef CUDAINTEGRATOR_H_
#define CUDAINTEGRATOR_H_

#include <stdio.h>
#include "cudaArrayOperationsHelper.h"
#include "evolutionParameters.h"

__global__ void trial ();



/**
 * @brief Force.
 *
 * @param x_ij Particles displacement vector
 * @param D Space dimension (i.e. size of `x` pointer)
 *
 * The force is a vector so to obtain the component of the force one should multiply
 * the result of this function by the distance of the particles along the component, i.e.
 *
 * \f[
 *  F_x(r) = G\frac{Mm}{r^3} x
 * \f]
 *
 * Moreover, the result _has still to be multiplied by the masses_.
 *
 */
template <unsigned short D, typename T>
__device__
inline T F ( const T *x_ij ) {

	/** Evaluate \f$\epsilon^2+r^2 = \epsilon^2+x^2 + y^2 + z^2\f$. */
	T tmp = EPS2;
	tmp += x_ij[0] * x_ij[0];
	tmp += x_ij[1] * x_ij[1];
	tmp += x_ij[2] * x_ij[2];

//	fprintf( stderr, "square distance: %g\n", tmp - EPS2 );
	

	/** @return \f$1/(\sqrt{r^2 + \epsilon^2})^3\f$. */
	return 1. / ( tmp * sqrtf( tmp ) );
};

/**
 * @brief Helper function to evolve positions by one steps.
 *
 * Performs the operation \f$x_j(t) = x_j(t-h) + v_j( t - h/2) \mathrm{d}t\f$
 * for each component \f$j\f$ running from 0 to `D` (template parameter).
 */
template <unsigned short D, typename T>
__device__
inline void leapFrogVerletUpdatePositions( T *x, const T *v ) {
	x[0] += v[0] * dt;
	x[1] += v[1] * dt;
	x[2] += v[2] * dt;
};

/**
 * @brief Helper function
 *
 * @param v velocity D-tuple to be updated
 * @param x displacement D-tuple "to be used as direction for the force"
 * @param scale is meant to be acceleration times time step, i.e. \f$a\,\mathrm{d}t\f$
 */
template <unsigned short D, typename T>
__device__
inline void leapFrogVerletUpdateVelocities ( T *v, const T *x, T scale ) {
	v[0] += scale * x[0];
	v[1] += scale * x[1];
	v[2] += scale * x[2];
};

template <size_t N, size_t D, typename T>
__global__ 
void cudaLeapFrogVerlet( T* x, T* v ) {
	__shared__ T    evolvingParticle[ D * 32 ];
//	__shared__ T surroungingParticle[ D * 32 ];

    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
	fetchFromGlobalMemory <D> ( evolvingParticle + i, x + i ); 

	evolvingParticle[i] ++;

	writeToGlobalMemory<D>( x + i, evolvingParticle + i );



//    printf( "Args [%u]: x %p y %p N %zu D %zu\n", threadID, (void*) x, (void*) v, N , D );
    // if you do like that then it may be that some other thread
    // writes in-between
//    printf( "block %u (of %u), %u\n", blockIdx.x, gridDim.x, threadIdx.x );
//    printf( "block %u (of %u), %u\n", blockIdx.y, gridDim.y, threadIdx.y );
//    printf( "block %u (of %u), %u\n", blockIdx.z, gridDim.z, threadIdx.z );

};


#endif /* CUDAINTEGRATOR_H_ */

