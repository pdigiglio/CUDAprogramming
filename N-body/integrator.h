
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
void leapfrogVerlet ( float *, float *, size_t );

/**
 * @brief Force.
 *
 * @param x Particles positions
 * @param N Number of particles (i.e. size of `x` pointer)
 */
//__device__ __host__
float F ( float *, size_t );

#endif /* INTEGRATOR_H_ */
