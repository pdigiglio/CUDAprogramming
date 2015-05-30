#include "integrator.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/**
 * The position is intended to be at the step \f$j\f, the velocity
 * at step \f$j-1/2\f$. This function will evolve both by one time step.
 
//__device__  __host__
template <size_t N, typename T>
void leapfrogVerlet ( T *x, T *v ) {

	for ( size_t i = 0; i < N; ++ i ) {
		x[i] += v[i] * dt;
	}

	T acceleration = F<N>( x, N );
	for ( size_t i = 0; i < N; ++ i ) {
		v[i] += acceleration * x[i] * dt; 
	}

}
*/
const float functionCoeff[4]  = {
	1.f * dt, // dummy value: it's not really used
	.5f * dt,
	.5f * dt,
	1.f * dt
};
const float evolutionCoeff[4] = {
	1.f * dt / 6.f,
	2.f * dt / 6.f,
	2.f * dt / 6.f,
	1.f * dt / 6.f
};
	void
rungeKutta ( float *x, float *v, size_t N ) {

	/*
	float tmpX = 0.f, incrementX = 0.f;
	float tmpV = 0.f, incrementV = 0.f;

	for ( size_t i = 0; i < N; ++ i ) {
	
		// reset temporary variables 
		incrementX = 0.f;
		incrementV = 0.f;
		
		tmpX = 0.f;
		tmpV = 0.f;

		// TODO: consider loop unrolling
		//
		// This can't be vectorized but branching can be
		// reduced

		// evaluate next value
		for ( short j = 0; j < 4; ++ j ) {
			tmpX = auxiliaryF( v[i] + functionCoeff[j] * tmpV );
			tmpV =          F( x[i] + functionCoeff[j] * tmpX );

			// save increment
			incrementX += evolutionCoeff[j] * tmpX;
			incrementV += evolutionCoeff[j] * tmpV;
		}

		x[i] += incrementX;
		v[i] += incrementV;
	}
	*/
}
/*
	template < size_t N, typename T>
	inline T
F ( const float *x ) {

	/// Evaluate \f$r^2 = x^2 + y^2 + z^2\f$.
	T tmp = x[0] * x[0];
	tmp += x[1] * x[1];
	tmp += x[2] * x[2] + EPS2;
	

	/// @return \f$1/r^3 = 1 / /r^2 \sqrt(r^2)\f$.
	return - 1. / ( tmp * sqrtf( tmp ) );
//	return .5f;
}
*/

	inline float
auxiliaryF( float v, size_t N ) {
	return v;
}
