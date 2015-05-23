/**
 * @file integrator.h
 * @brief Header for integrator methods.
 */

#ifndef INTEGRATOR_H_
#define INTEGRATOR_H_

#include <stdio.h>
#include <stdlib.h>

extern const float functionCoeff[4];
extern const float evolutionCoeff[4];
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
inline float F ( float x, size_t N = 1 );

/**
 * @brief Auxiliaty function for Runge-Kutta method.
 *
 * Since Newton's equation is a 2nd order differential
 * equation, I'll just need an auxiliary function.
 *
 * @param v Called `v` since velocity will be passed
 */
inline float auxiliaryF ( float x, size_t N = 1 );

#endif /* INTEGRATOR_H_ */
