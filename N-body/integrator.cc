#include "integrator.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * The position is intended to be at the step \f$j\f$, the velocity
 * at step \f$j-1/2\f$. This function will evolve both by one time step.
 */
//__device__  __host__
void leapfrogVerlet ( float *x, float *v, size_t N ) {
	for ( size_t i = 0; i < N; ++ i ) {
		x[i] += v[i] * dt;
		v[i] += F( x[i], N ) * dt; 
	}
}

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

	float tmpX = 0.f, incrementX = 0.f;
	float tmpV = 0.f, incrementV = 0.f;

	/**
	 * @attention this loop and the inner one commute _only
	 * if_ there is one particle or the force comes from an
	 * external field.
	 *
	 * If it is not the case, to evaluate the \f$k_j\f$-values
	 * the displacement of _all_ the particles has to be taken
	 * into accout.
	 */
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
}

	inline float
F ( float x, size_t N ) {
	return .5f;
}

/**
 * @brief Auxiliary function for RK.
 *
 * It's just the identical function (for the speed, in this case).
 */
	inline float
auxiliaryF( float v, size_t N ) {
	return v;
}
