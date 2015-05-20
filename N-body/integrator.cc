#include "integrator.h"

#include <stdio.h>
#include <stdlib.h>

/**
 * The position is intended to be at the step \f$j\f, the velocity
 * at step \f$j-1/2\f$. This function will evolve both by one time step.
 */
//__device__  __host__
void leapfrogVerlet ( float *x, float *v, size_t N ) {
	for ( int i = 0; i < N; ++ i ) {
		x[i] += v[i] * dt;
		v[i] += F( x, N ) * dt; 
	}
}

int rungeKutta () {
	return 1;
}

float F ( float *x, size_t N ) {
	return .5f;
}
