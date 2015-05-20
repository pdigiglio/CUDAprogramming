#include "integrator.h"

#include <stdlib.h>
#include <stdio.h>

const unsigned int numOfParticles = 1;

	int
main () {

	float x[numOfParticles] = {};
	float v[numOfParticles] = {};

	for ( unsigned t = 0; t < 10; ++ t ) {
		leapfrogVerlet( x, v, numOfParticles );
		printf( "%f %f\n", x[0], v[0] );
	}
	return 0;
}
