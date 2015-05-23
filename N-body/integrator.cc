#include "integrator.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * The position is intended to be at the step \f$j\f, the velocity
 * at step \f$j-1/2\f$. This function will evolve both by one time step.
 */
//__device__  __host__
void leapfrogVerlet ( float *x, float *v, size_t N ) {
	for ( size_t i = 0; i < N; ++ i ) {
		x[i] += v[i] * dt;
		v[i] += F( x[i], N ) * dt; 
	}
}

	void
rungeKutta ( float *x, float *v, size_t N ) {

	float tmpX = 0.f, incrementX = 0.f;
	float tmpV = 0.f, incrementV = 0.f;

	for ( size_t i = 0; i < N; ++ i ) {
	
		incrementX = 0.f;
		incrementV = 0.f;

		// k1
		tmpX = v[i];
		tmpV = F( x[i], N );

		// save increment
		incrementX += tmpX;
		incrementV += tmpV;

		// k2
		tmpX = v[i] + .5 * dt * tmpX;
		tmpV = F( x[i] + .5 * dt * tmpV );

		// save increment
		incrementX += 2 * tmpX;
		incrementV += 2 * tmpV;

		// k3
		tmpX = v[i] + .5 * dt * tmpX;
		tmpV = F( x[i] + .5 * dt * tmpV );

		// save increment
		incrementX += 2 * tmpX;
		incrementV += 2 * tmpV;

		// k4
		tmpX = v[i] + dt * tmpX;
		tmpV = F( x[i] + dt * tmpV );

		x[i] += ( incrementX + tmpX ) * dt / 6;
		v[i] += ( incrementV + tmpV ) * dt / 6;
	}
}

float F ( float x, size_t N ) {
	return .5f;
}
