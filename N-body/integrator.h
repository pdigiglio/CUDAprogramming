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
int rungeKutta ( void );

//__device__ __host__
void leapfrogVerlet ( float *x, float *v, size_t N );

/**
 * @brief Force.
 *
 * @param x Particles positions
 * @param N Number of particles (i.e. size of `x` pointer)
 */
//__device__ __host__
float F ( float *x, size_t N );

#endif /* INTEGRATOR_H_ */
