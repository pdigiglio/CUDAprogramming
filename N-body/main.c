#include "integrator.h"

#include <stdlib.h>
#include <stdio.h>

const unsigned int numOfParticles = 1;

	int
main () {

	float x[numOfParticles] = {};
	float v[numOfParticles] = {};

	for ( unsigned t = 0; t < 1000000; ++ t ) {
//		leapfrogVerlet( x, v, numOfParticles );
		rungeKutta( x, v, numOfParticles );
		printf( "%u %f %f\n", t, x[0], v[0] );
	}
	return 0;
}
