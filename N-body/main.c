/**
 * @file main.c
 * @brief
 */

#include "integrator.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

const unsigned int spaceDimension = 3;
const unsigned int numOfParticles = 4; /* XXX this must be even! */

	int
main ( int argc, char *argv[] ) {
	
	fprintf(stderr, "%s Starting...\n\n", argv[0]);

	/**
	 * Define number of elements of position array.
	 */
	const size_t xEntries = spaceDimension * numOfParticles;
	double x[ xEntries ] = {
		1., 0., 0.,
		-1., 0., 0.,
		0., -1., 0.,
		0., 1., 0.,
	};

	/**
	 * Define number of elements of velocity array.
	 *
	 * @attention If one wants to use a kind of 4-velocity with the
	 * particle mass as 4-th component, then `vEntries` and `xEntries`
	 * differ.
	 */
	const size_t vEntries    = xEntries;
	double v[ vEntries ] = {
		0., 0., 0.,
		0., 0., 0.,
		0., 0., 0.,
		0., 0., 0.,
	};
	
	/**
	 * Define mass vector.
	 */
	double m[ numOfParticles ] = { 1., 1., 1., 1. };

//	const size_t xEntries    = spaceDimension * numOfParticles;
//	const size_t xMemorySize = xEntries * sizeof( double );
//	double *x = (double *) malloc( xMemorySize );
//	if( ! x ) {
//		fprintf( stderr, " > x allocation failed\n" );
//		return 1;
//	}
//
//	/**
//	 * Initialize particles in a symmetric configuration, such that
//	 * center of mass position is af \f$(0,0,0)\f$. 
//	 */
//	for( size_t j = 0; j < xEntries; j += 6 ) {
//		x[j  ] = ( (double) rand() / RAND_MAX ) - .5;
//		x[j+1] = ( (double) rand() / RAND_MAX ) - .5;
//		x[j+2] = ( (double) rand() / RAND_MAX ) - .5;
//
//		x[j+3] = - x[j  ];
//		x[j+4] = - x[j+1];
//		x[j+5] = - x[j+2];
//	}
//
//	/* the 4-th component of velocity will carry the particle mass */
//	const size_t vEntries    = ( spaceDimension + 1 ) * numOfParticles;
//	const size_t vMemorySize = xMemorySize + numOfParticles * sizeof( double );
//	double *v = (double *) malloc( vMemorySize );
//	if( ! v ) {
//		fprintf( stderr, " > v allocation failed\n" );
//		return 1;
//	}
//
//	/**
//	 * Initialize particles such that the center of mass speed is zero (i.e.
//	 * our frame is the center of mass one).
//	 */
//	const double vScale = .0;
//	for( size_t j = 0; j < vEntries; j += 8 ) {
//		v[j  ] = vScale * ( ( (double) rand() / RAND_MAX ) - .5 );
//		v[j+1] = vScale * ( ( (double) rand() / RAND_MAX ) - .5 );
//		v[j+2] = vScale * ( ( (double) rand() / RAND_MAX ) - .5 );
//		v[j+3] = (double) 1; // mass
//
//		/* make sure center of mass is at rest */
//		v[j+4] = - v[j];
//		v[j+5] = - v[j+1];
//		v[j+6] = - v[j+2];
//		v[j+7] = (double) 1; // mass
//	}

	const unsigned int MaxNumberOfTimeSteps = 10000;
	const unsigned int TimeStepIncrement    = 10;
	for ( unsigned int t = 0; t < MaxNumberOfTimeSteps; t += TimeStepIncrement ) {

//		fprintf( stderr, " > step %u of %u\n", t , MaxNumberOfTimeSteps );
//		printf( "%u ", t ); 

		printf( "%u\t", t );
		for( unsigned int i = 0; i < numOfParticles * spaceDimension;  i += 6 ) {
			printf( "%.6g\t%.6g\t%.6g\t", x[i ], x[i+1], x[i+2] );
			printf( "%.6g\t%.6g\t%.6g\t", x[i+3], x[i+4], x[i+5] );
		}

		printf( "\n" );

		for ( unsigned int j = 0; j < TimeStepIncrement; ++ j ) {
			leapfrogVerletBlock < numOfParticles, 2, spaceDimension > ( x, v, m );
	//		rungeKutta( x, v, numOfParticles );
		}

//		fprintf( stderr, " again > step %u of %u\n", t , MaxNumberOfTimeSteps );
	}

	/*
     * `cudaDeviceReset()` causes the driver to clean up all state. While
     * not mandatory in normal operation, it is good practice.  It is also
     * needed to ensure correct operation when the application is being
     * profiled. Calling `cudaDeviceReset()` causes all profile data to be
     * flushed before the application exits.
	 */
//    cudaDeviceReset();

	return 0;
}
