/**
 * @file integrator.h
 * @brief Header for integrator methods.
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <stdio.h>
#include <stdlib.h>

/*
 * use literals for float
 */
#define dt .001f

//__device__ __host__
void rungeKutta ( float *x, float *v, size_t N );

//__device__ __host__
void leapfrogVerlet ( float *x, float *v, size_t N );

/**
 * @brief Force.
 *
 * @param x Particles positions
 * @param N Number of particles (i.e. size of `x` pointer)
 */
//__device__ __host__
float F ( float x, size_t N = 1 );

#endif /* INTEGRATOR_H_ */
