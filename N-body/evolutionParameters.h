/**
 *
 *
 *           @name  evolutionParameters.h
 *          @brief  Parameters like time step and distance correction \f$\epsilon^2\f$.
 *
 *        @version  1.0
 *           @date  06/18/2015 (03:37:42 PM)
 *       @revision  none
 *       @compiler  gcc
 *
 *
 *         @author  P. Di Giglio (github.com/pdigiglio), <p.digiglio91@gmail.com>
 *        @company  
 *
 *          Example usage:
 *          @code
 *          @endcode
 *
 *
 */

#ifndef  EVOLUTIONPARAMETERS_H_
#define  EVOLUTIONPARAMETERS_H_

#define dt .0001

/*
 * @def EPS2
 * @brief This _small_ shifting will be use in F() to prevent division by \f$0\f$. 
 * @attention One should be careful in choosing the value of \f$\epsilon^2\f$, e.g. if the
 * type is `float` then \f$ 1/(\epsilon^2)^3\f$ _must_ be smaller than the maximum value
 * a `float` variable can hold.
 */
#define EPS2 .0001


#endif   /* ----- #ifndef EVOLUTIONPARAMETERS_H_  ----- */
