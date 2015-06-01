/**
 * @file integrator.h
 * @brief Header for integrator methods.
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

extern const float functionCoeff[4];
extern const float evolutionCoeff[4];
/*
 * use literals for float
 */
#define dt .001

/*
 * @def EPS2
 * @brief This _small_ shifting will be use in F() to prevent division by \f$0\f$. 
 * @attention One should be careful in choosing the value of \f$\epsilon^2\f$, e.g. if the
 * type is `float` then \f$ 1/(\epsilon^2)^3\f$ _must_ be smaller than the maximum value
 * a `float` variable can hold.
 */
#define EPS2 .002

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
//__device__ __host__
template <unsigned short D, typename T>
inline T F ( const T *x_ij ) {

	/** Evaluate \f$\epsilon^2+r^2 = \epsilon^2+x^2 + y^2 + z^2\f$. */
	T tmp = EPS2;
	tmp += x_ij[0] * x_ij[0];
	tmp += x_ij[1] * x_ij[1];
	tmp += x_ij[2] * x_ij[2];

//	fprintf( stderr, "square distance: %g\n", tmp - EPS2 );
	

	/** @return \f$1/(\sqrt{r^2 + \epsilon^2})^3\f$. */
	return - 1. / ( tmp * sqrtf( tmp ) );
};

template <unsigned short D, typename T>
void distance( const T *x, const T *y, T *x_ij ) {
	x_ij[0] = x[0] - y[0];
	x_ij[1] = x[1] - y[1];
	x_ij[2] = x[2] - y[2];
};

//__device__ __host__
template <size_t N, unsigned short D, typename T>
void leapfrogVerlet ( T *x, T *v ) {

	// ---------------------------------------------------------------------------------
	/* XXX if N % 4 == 0 then thi loop can be unrolled by 4x */
	/* evolve particles' position */
	T *x_i = x;
	T *v_i = v;
	for ( size_t i = 0; i < N; ++ i ) {
		x_i[0] += v_i[0] * dt;
		x_i[1] += v_i[1] * dt;
		x_i[2] += v_i[2] * dt;

		x_i += D;
		v_i += D + 1;
	}
	// ---------------------------------------------------------------------------------

	/* auxiliary variable to hold distance among particles */
	T x_ij[D];
	/* auxiliary variables for the inner/outer-loop particle */
	T *x_j = NULL;
	T *v_j = NULL;

	x_i = x;
	v_i = v;
	for ( size_t i = 0; i < N; ++ i ) {
		/* (re-)initialize x_j */
		x_j = x;
		v_j = v;

		for( size_t j= 0; j < i; ++ j ) {

			/* assign the distance among particles */
			distance< D >( x_i, x_j, x_ij );
			T acceleration_i = F< D >( x_ij );

			T acceleration_j = v_i[3] * acceleration_i;
			acceleration_i *= v_j[3];

//			/* 
//			 * replace this loop with the following to (hopefully) achieve a better
//			 * memory access
//			 */
//			for ( size_t d = 0; d < D; ++ d ) {
//				v[d]     += acceleration * x_ij[d] * dt; 
//				v[d + D] -= acceleration * x_ij[d] * dt; 
//			}

			// ---------------------------------------------------------------------
			/*
			 * XXX multiplying the vector by dt and acceleration x_ij in Distance
			 * would save 6 flops
			 */
			/* update velocities */
			v_i[0] += acceleration_i * x_ij[0] * dt;
			v_i[1] += acceleration_i * x_ij[1] * dt;
			v_i[2] += acceleration_i * x_ij[2] * dt;
			

			v_j[0] -= acceleration_j * x_ij[0] * dt;
			v_j[1] -= acceleration_j * x_ij[1] * dt;
			v_j[2] -= acceleration_j * x_ij[2] * dt;
			// ---------------------------------------------------------------------
			

			/* go to next D-tuple of coordinates */
			x_j += D;
			v_j += D + 1;
		}

		/* go to next D-tuple of coordinates */
		x_i += D;
		v_i += D + 1;
	}
};

/* TODO divide into blocks */
//__device__ __host__
template <size_t N, size_t BLOCK_SIZE, unsigned short D, typename T>
void leapfrogVerletBlock ( T *x, T *v ) {

	// ---------------------------------------------------------------------------------
	/* XXX if N % 4 == 0 then thi loop can be unrolled by 4x */
	/* evolve particles' position */
	for ( size_t i = 0; i < N * D; ++ i ) {
		x[i] += v[i] * dt;
	}
	// ---------------------------------------------------------------------------------

	/* auxiliary variable to hold distance among particles */
	T x_ij[D];
	/* auxiliary variables for the inner/outer-loop particle */
	T *x_j = NULL, *x_i = x;
	T *v_j = NULL, *v_i = v;

	for ( size_t i = 0; i < N; ++ i ) {
		/* (re-)initialize x_j */
		x_j = x;
		v_j = v;

		for( size_t j = 0; j < N; ++ j ) {

			/* assign the distance among particles */
			distance< D >( x_i, x_j, x_ij );
			T acceleration = F< D >( x_ij );

			// ---------------------------------------------------------------------
			/*
			 * XXX multiplying the vector by dt and acceleration x_ij in Distance
			 * would save 6 flops
			 */
			/* update velocities */
			v_i[0] += acceleration * x_ij[0] * dt;
			v_i[1] += acceleration * x_ij[1] * dt;
			v_i[2] += acceleration * x_ij[2] * dt;
			

//			v_j[0] -= acceleration * x_ij[0] * dt;
//			v_j[1] -= acceleration * x_ij[1] * dt;
//			v_j[2] -= acceleration * x_ij[2] * dt;
			// ---------------------------------------------------------------------
			

			/* go to next D-tuple of coordinates */
			x_j += D;
			v_j += D;
		}

		/* go to next D-tuple of coordinates */
		x_i += D;
		v_i += D;
	}
};
//__device__ __host__
void rungeKutta ( float *x, float *v, size_t N );

/**
 * @brief Auxiliaty function for Runge-Kutta method.
 *
 * Since Newton's equation is a 2nd order differential
 * equation, I'll just need an auxiliary function.
 *
 * @param v Called `v` since velocity will be passed
 */
inline float auxiliaryF ( float x, size_t N = 1 );

#endif /* INTEGRATOR_H_ */
