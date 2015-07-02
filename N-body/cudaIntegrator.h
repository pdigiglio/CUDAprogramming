/**
 * @file cudaIntegrator.h
 * @brief Implementation of Leapfrog Verlet integrator in CUDA
 */

#ifndef CUDAINTEGRATOR_H_
#define CUDAINTEGRATOR_H_

#include <stdio.h>
#include "cudaArrayOperationsHelper.h"
#include "evolutionParameters.h"

/**
 * @brief Evaluate part of the force.
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
inline T basicInteraction ( const T *x_ij ) {

	/** Evaluate \f$\epsilon^2+r^2 = \epsilon^2+x^2 + y^2 + z^2\f$. */
	T tmp = (T) EPS2;
    for( short unsigned d = 0; d < D; ++ d )
        tmp += x_ij[d] * x_ij[d];

	/** @return \f$1/(\sqrt{r^2 + \epsilon^2})^3\f$. */
//    return - 1.0f / ( tmp * sqrtf( tmp ) );
    return - fsqrtf( tmp * tmp * tmp );
};

/**
 * @brief Specialized template for `double`.
 *
 * The function `sqrt()` is used instead of `sqrtf()` and this
 * _terribly slows down the calculations_.
 *
 * @attention Partial specialization is not allowed for standard
 * C/C++ functions. Actually this is not a specialized `template<>`
 * but just a different function.
 *
 */
template <unsigned short D>
__device__
inline double basicInteraction ( const double * x_ij ) {
	double tmp = (double) EPS2;
    for( short unsigned d = 0; d < D; ++ d )
        tmp += x_ij[d] * x_ij[d];

    // no need for literals here: default is double
//    return - 1.0 / ( tmp * sqrt( tmp ) );
    return - rsqrt( tmp * tmp * tmp );
}

/**
 * @brief Helper function to evolve positions by one steps.
 *
 * Performs the operation \f$x_j(t) = x_j(t-h) + v_j( t - h/2) \mathrm{d}t\f$
 * for each component \f$j\f$ running from 0 to `D` (template parameter).
 */
template <unsigned short D, typename T>
__device__
inline void leapFrogVerletUpdatePositions( T *x, const T *v ) {
    for( short int d = 0; d < D; ++ d )
        x[d] += v[d] * dt;

//	x[0] += v[0] * dt;
//	x[1] += v[1] * dt;
//	x[2] += v[2] * dt;
};

/**
 * @brief Helper function
 *
 * The time step `dt` is defined as a macro so I don't pass it as a value.
 *
 * @attention This is the same function as `leapFrogVerletUpdateAccelerations()` but
 * I call it differently to keep them logically separated.
 *
 * @param v velocity D-tuple to be updated
 * @param a acceleration D-tuple 
 */
template <unsigned short D, typename T>
__device__
inline void leapFrogVerletUpdateVelocities ( T *v, const T *a ) {
	v[0] += dt * a[0];
	v[1] += dt * a[1];
	v[2] += dt * a[2];
};

template <unsigned short D, typename T>
__device__
inline void leapFrogVerletUpdateAccelerations ( T *a, const T *x, T magnitude ) {
	a[0] += magnitude * x[0];
	a[1] += magnitude * x[1];
	a[2] += magnitude * x[2];
}

/**
 * @brief Kernel to update positions in the system.
 *
 * I'll use the default stream for this kernel so that there is an implicit synchronization
 * after the call and I can go on in evaluating the new velocities.
 *
 * @attention I don't use any shared memory since it wouldn't help.
 */
template <size_t N, size_t D, typename T>
__global__
void cudaUpdateSystemGlobalPositions( T *x, const T *v ) {
	size_t i = blockDim.x * blockIdx.x + threadIdx.x;

	leapFrogVerletUpdatePositions<D>( x + D * i, v + D * i );
}

/**
 * @brief
 *
 * At the beginning, each thread fetches velocity vectors of particles corresponding
 * to its ID.
 *
 * Required amounth of shared memory is the following:
 *  * `D * BLOCK_SIZE` for `evolvingParticleAcceleration[]`;
 *  * `D * BLOCK_SIZE` for `evolvingParticlePosition[]`;
 *  * `D * BLOCK_SIZE` for `surroundingParticlePosition[]`;
 *  * `BLOCK_SIZE` for `surroundingParticlePosition[]`;
 * So the total shared memory per block is `( 3*D + 1 ) * BLOCK_SIZE * sizeof( T )`.
 *
 * @param x vector of particle positions
 * @param v vector of particle velocities
 * @param m vector of partile masses
 */
template <size_t N, size_t D, typename T>
__global__ 
void cudaLeapFrogVerlet( const T* x, T* v, const T *m ) {
	/**
	 * Shared memory is dynamically allocated at run-time and the size is taken from the
	 * third argument of the kernel <<< ... >>>  call.
	 */
	extern __shared__ T blockSharedMemory[];

	// this points at the end of `evolvingParticlePosition[]`
	T *const surroundingParticlePosition  = blockSharedMemory;
	// this points at the end of `surroundingParticlePosition[]`
	T *const surroundingParticleMass      = surroundingParticlePosition + D * blockDim.x;
    // only blockDim.x elements are left in blockSharedMemory[]

    // each thread allocates its position and acceleration vector
	T evolvingParticleAcceleration[ D ];
	T evolvingParticlePosition[ D ];

    // evaluate the thread ID
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    // fetch position of the i-th particle 
	fetchFromGlobalMemory <D> ( evolvingParticlePosition, x + D * i ); 

	// set accelerations to zero
	setVectorToZero <D> ( evolvingParticleAcceleration );

	// temporary variable to hold particle distances
	// it's not shared among threads but it's private
	T x_ij[D], accelerationModulus = (T) 0;

	// N is assumed to be multiple of blockDim.x
	const size_t numOfIterations = N / blockDim.x;
	for( size_t it = 0; it < numOfIterations; ++ it ) {
		// fetch positions of particle to evaluate the interactions with
		fetchFromGlobalMemory<D> ( surroundingParticlePosition + D * threadIdx.x, x + D * ( it * blockDim.x + threadIdx.x ) );
		// mass is scalar so only one component per thread needs to be fetched
		fetchFromGlobalMemory<1> ( surroundingParticleMass + threadIdx.x, m + it * blockDim.x + threadIdx.x );

		// sync so that all memory is properly loaded
		__syncthreads();

		for( size_t j = 0; j < blockDim.x; ++ j ) {
			// store particle distance into "buffer" `x_ij[]`
			distance<D>( evolvingParticlePosition, surroundingParticlePosition + D * j, x_ij );

			// update acceleration vector
			accelerationModulus = basicInteraction<D>( x_ij ) * surroundingParticleMass[j];

//            if( accelerationModulus == 0 ) {
//                if( x_ij[0] != 0 || x_ij[1] != 0 || x_ij[1] != 0 ) {
//                    for( short i = 0; i < D; ++ i )
//                        printf( " x_ij[%d] = %g\n", i, x_ij[i] );
//                }
//                return ;
//
//            }

			leapFrogVerletUpdateAccelerations<D> ( evolvingParticleAcceleration, x_ij, accelerationModulus );
		}
        __syncthreads();
	}

	leapFrogVerletUpdateVelocities<D>( v + D * i, evolvingParticleAcceleration );
};


template <size_t N, size_t D, typename T>
__global__
void cudaLeapFrogVerletNoShared( const T *x, T* v, const T *m ) {
    unsigned i = blockDim.x * blockIdx.x + threadIdx.x;

	// temporary variable to hold particle distances
	// it's not shared among threads but it's private
	T x_ij[D], accelerationModulus;
    T particleAcceleration[D];

    setVectorToZero<D>( particleAcceleration );

	for( size_t j = 0; j < N; ++ j ) {
        // store particle distance into "buffer" `x_ij[]`
        distance<D>( x + D * i, x + D * j, x_ij );

        // update acceleration vector
        accelerationModulus = basicInteraction<D>( x_ij ) * m[j];
        leapFrogVerletUpdateAccelerations<D> ( particleAcceleration , x_ij, accelerationModulus );
    }

	leapFrogVerletUpdateVelocities<D>( v + D * i, particleAcceleration );
};
#endif /* CUDAINTEGRATOR_H_ */

