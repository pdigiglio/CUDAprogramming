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
	return 1. / ( tmp * sqrtf( tmp ) );
};

template <unsigned short D, typename T>
void distance( const T *x, const T *y, T *x_ij ) {
	x_ij[0] = x[0] - y[0];
	x_ij[1] = x[1] - y[1];
	x_ij[2] = x[2] - y[2];
};

/**
 * @brief Scales second vector by \f$c\f$ and adds it up into the first
 */
template <size_t N, unsigned short D, typename T>
void incrementFirstVecBySecond( T *a, const T *b, T c ) {
	/* XXX loop unrolled */
	for( size_t i = 0; i < N * D; i += 2 ) {
		a[i  ]  += c * b[i  ];
		a[i+1]  += c * b[i+1];
	}
}

/**
 * @brief Adds two vectors and stores result into the first
 */
template <size_t N, unsigned short D, typename T>
void incrementFirstVecBySecond( T *a, const T *b ) {
	/* XXX loop unrolled */
	for( size_t i = 0; i < N * D; i += 2 ) {
		a[i  ]  += b[i  ];
		a[i+1]  += b[i+1];
	}
}

//__device__ __host__
template <size_t N, unsigned short D, typename T>
void leapfrogVerlet ( T *x, T *v, const T *m ) {

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

/**
 * @brief Helper function to evolve positions by one steps.
 *
 * Performs the operation \f$x_j(t) = x_j(t-h) + v_j( t - h/2) \mathrm{d}t\f$
 * for each component \f$j\f$ running from 0 to `D` (template parameter).
 */
template <unsigned short D, typename T>
inline void leapFrogVerletUpdatePositions( T *x, const T *v ) {
	x[0] += v[0] * dt;
	x[1] += v[1] * dt;
	x[2] += v[2] * dt;
}

/**
 * @brief Helper function
 *
 * @param v velocity D-tuple to be updated
 * @param x displacement D-tuple "to be used as direction for the force"
 * @param scale is meant to be acceleration times time step, i.e. \f$a\,\mathrm{d}t\f$
 */
template <unsigned short D, typename T>
inline void leapFrogVerletUpdateVelocities ( T *v, const T *x, T scale ) {
	v[0] += scale * x[0];
	v[1] += scale * x[1];
	v[2] += scale * x[2];
}

/**
 * @brief Blocked version of Leapfrog algorithm.
 *
 * Same as `leapfrogVerlet()` but work is split into blocks.
 *
 * @attention The size `N` of vectors _must_ be a multiple of `BLOCK_SIZE`.
 * 
 * @param x position vector
 * @param v velocity vector
 * @param m mass vector
 */
//__device__ __host__
template <size_t N, size_t BLOCK_SIZE, unsigned short D, typename T>
void leapfrogVerletBlock ( T *x, T *v, const T *m ) {

//	fprintf( stderr, "calling leapfrogVerletBlock() with BLOCK_SIZE = %zu\n", BLOCK_SIZE );

	// ---------------------------------------------------------------------------------
	/* XXX if N % 4 == 0 then thi loop can be unrolled by 4x */
	/* evolve particles' position */
	T *x_i = x;
	T *v_i = v;
	for ( size_t i = 0; i < N; ++ i ) {

		leapFrogVerletUpdatePositions<D>( x_i, v_i );

		x_i += D;
		v_i += D;
	}
	// ---------------------------------------------------------------------------------

	/* auxiliary variable to hold distance among particles */
	T x_ij[D];

	/* auxiliary variables for the inner/outer-loop particle */
	T *x_j = NULL;
	T *v_j = NULL;

	// >> N is integer multiple of BLOCK_SIZE !!
	// >> N / BLOCK_SIZE will be evaluated @ compile-time
	for ( size_t ib = 0; ib < N / BLOCK_SIZE; ++ ib ) {

		for ( size_t jb = 0; jb < N / BLOCK_SIZE; ++ jb ) {
			x_i = x + D * (ib * BLOCK_SIZE); // re-use previous variable
			v_i = v + D * (ib * BLOCK_SIZE); // re-use previous variable

			for ( size_t i = 0; i < BLOCK_SIZE; ++ i ) {
				/* move pointers at the beginning of the block */
				x_j = x + D * (jb * BLOCK_SIZE);
				v_j = v + D * (jb * BLOCK_SIZE);

				for( size_t j = 0; j < BLOCK_SIZE; ++ j ) {

//					fprintf( stdout, "%zu %zu %zu %zu\n", ib, jb, i, j );
					/* assign the distance among particles */
					distance< D >( x_i, x_j, x_ij );

					/**
					 * Acceleration on \f$i\f$-th particle due to \f$j\f$-th is 
					 * \f$m_j\vec{r}/r^3\f$.
					 */
					T acceleration = F< D >( x_ij ) * m[ jb * BLOCK_SIZE + j ];


					/* update velocities */
					leapFrogVerletUpdateVelocities<D>( v_j, x_ij, acceleration * dt );

					// update coordinates till the end of the block is reached
					x_j += D;
					v_j += D;
				}

				// go to next row
				x_i += D;
				v_i += D;
			}
//			fprintf( stdout, "\n" );
		}
	}
};

///*
// * @brief Acceleration of the i-th particle
// */
//template <size_t N, unsigned int D, typename T>
//T Accelearation ( const T *x, const T *v, size_t i ) {
//
//	T acc = (T) 0;
//
//	/* auxiliary variable to hold distance among particles */
//	T x_ij[D];
//	T *x_j = x, *x_i = x + i * D;
//	for( size_t j = 0; j < N; ++ j ) {
//		/* evaluate distance */
//		distance <D> ( x_i, x_j, x_ij );	
//		/* add force due to interaction among i-j */
//		acc += v[ j * (D + 1) + 3 ] * F <D> ( x_ij );
//
//		/* update j-th particle */
//		x_j += D;
//	}
//
//	return acc;
//
//}
//
///*
// * @brief Copy second vector into first.
// */
//
//template <size_t N, unsigned short D, typename T>
//void copySecondVecIntoFirst( T *a, const T *b ) {
//	/* XXX loop unrolled */
//	for( size_t i = 0; i < N * D; i += 2 ) {
//		a[i  ]  = b[i  ];
//		a[i+1]  = b[i+1];
//	}
//}
//
////
//template <size_t N, unsigned short D, typename T>
//
//void newK ( const T *x, T *k ) {
//	for ( size_t i = 0; i < N; ++ i ) {
//		for ( size_t i = 0; i < N; ++ i ) {
//			/* evaluate force */
//		}
//	}
//}
//
//template <size_t N, unsigned short D, typename T>
//void evaluateNewK( const T *x, T *k, T c ) {
//	T tmp[N];
//	copySecondVecIntoFirst( tmp, x );
//	incrementFirstVecBySecond( tmp, k, c );
//
//	newK( tmp, k );
//}
//
//extern float functionCoeff[4];
//extern float evolutionCoeff[4]*;
//
////__device__ __host__
//template <size_t N, unsigned short D, typename T>
//void rungeKutta ( T *x, T *v) {
//
//	/* temporary vector to hold \f$k_j\f$ values */
//	T k[N];
//
//	/* increment for position */
//	T positionIncrement[N];
//	setVectorToZero( positionIncrement );
//	
//	// TODO evaluate k1 and store it into k
//	
//	incrementFirstVecBySecond( positionIncrement, k, (T) dt / 6. );
//
//	// TODO evaluate k2 and store it into k
//
//	incrementFirstVecBySecond( positionIncrement, k, (T) dt / 3. );
//
//	// TODO evaluate k3 and store it into k
//
//	incrementFirstVecBySecond( positionIncrement, k, (T) dt / 3. );
//
//	// TODO evaluate k4 and store it into k
//
//	incrementFirstVecBySecond( positionIncrement, k, (T) dt / 6. );
//
//	// now update position vector
//	incrementFirstVecBySecond( x, positionIncrement );
//
////	/* temporary vector to hold values of position where to evaluate force */
////	T tmpPosition[N]
////	for ( unsigned short step; step < 4; ++ step ) {
////		setVectorToZero( k );
////
////		for( size_t i = 0; i < N; ++ i ) {
////			for( size_t j = 0; j < i; ++ j ) {
////
////			}
////		}
////	}
//
////	/* auxiliary tmp variables */
////	T tmpX = (T) 0., incrementX = (T) 0.;
////	T tmpV = (T) 0., incrementV = (T) 0.;
////
////	for ( size_t i = 0; i < N; ++ i ) {
////	
////		// reset temporary variables 
////		incrementX = (T) 0.;
////		incrementV = (T) 0.;
////		
////		tmpX = (T) 0.;
////		tmpV = (T) 0.;
////
////		// TODO: consider loop unrolling
////		//
////		// This can't be vectorized but branching can be
////		// reduced
////
////		// evaluate next value
////		for ( short j = 0; j < 4; ++ j ) {
////			tmpX = auxiliaryF( v[i] + functionCoeff[j] * tmpV );
////			tmpV =          F( x[i] + functionCoeff[j] * tmpX );
////
////			// save increment
////			incrementX += evolutionCoeff[j] * tmpX;
////			incrementV += evolutionCoeff[j] * tmpV;
////		}
////
////		x[i] += incrementX;
////		v[i] += incrementV;
////	}
//};

/**
 * @brief Auxiliaty function for Runge-Kutta method.
 *
 * Since Newton's equation is a 2nd order differential
 * equation, I'll just need an auxiliary function.
 *
 * @param v Called `v` since velocity will be passed
 */
//inline float auxiliaryF ( float x, size_t N = 1 );

#endif /* INTEGRATOR_H_ */
