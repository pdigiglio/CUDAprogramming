/**
 *
 *
 *           @name  systemInitializationHelper.h
 *          @brief  Template helper functions to initialize the system
 *
 *        @version  1.0
 *           @date  06/16/2015 (11:11:09 AM)
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

/**
 * @brief Helper function to allocate memory for single pointer.
 *
 * @param x reference to a pointer, otherwise copy by value prevents pointer itself to be changed
 */
template <typename T>
void allocatePointer( T* &x, size_t xEntries, const char name[] = "pointer" ) {
	x = (T *) malloc( xEntries * sizeof( T ) );
	if ( ! x ) {
		fprintf( stderr, "%s allocation failed\n", name );
		exit( EXIT_FAILURE );
	}
//	else {
//		fprintf( stderr, "%s [%p] properly allocated!", name, (void *) x );
//	}
}

/**
 * @brief Initializes vectors.
 * 
 * It takes two contiguous particles and initializes randomly the first one with each
 * component taking values in the interval \f$[-a,+a)\f$ (being \f$a>0\f$).
 * The second one is initialized int such a way that \f$m_1x_1 + m_2 x_2 = 0\f$.
 * This is simply achieved by choosing \f$x_2 = -x_1m_1/m_2\f$ and, since the number of
 * particles is even, this ensures that
 *  * center of mass is at rest;
 *  * center of mass frame is such that its origin is \f$(0,0,0)\f$.
 *
 * @param x vector to initialize
 * @param m mass(es) corresponding to the two particles
 * @param scale the value of \f$a\f$
 */
template <typename T, short int D>
void initialize( T *x, const T *m, T scale = 1. ) {
	// random for first entry
	x[0] = (T) ( scale * ( (T) rand() / RAND_MAX - .5 ) );
	x[1] = (T) ( scale * ( (T) rand() / RAND_MAX - .5 ) );
	x[2] = (T) ( scale * ( (T) rand() / RAND_MAX - .5 ) );

	// not random for the second entry
	x[3] = - ( m[0] / m[1] ) * x[0];
	x[4] = - ( m[0] / m[1] ) * x[1];
	x[5] = - ( m[0] / m[1] ) * x[2];
}

/**
 * @brief Helper function to initialize positions, velocities and masses.
 */
template <typename T, unsigned short D, size_t numOfParticles>
void initializeSystem ( T* &x, T* &v, T* &m ) {
	/**
	 * Allocate memory via `allocatePointer()`.
	 */
	fprintf( stderr, "Initializing system with %zu particles... ", numOfParticles );
	allocatePointer<T>( x, D * numOfParticles, "x" );
	allocatePointer<T>( v, D * numOfParticles, "v" );
	allocatePointer<T>( m, numOfParticles, "m" );

	/**
	 * Initialize the system such that the center of mass is at \f$(0,0,0)\f$ and it's
	 * at rest.
	 *
	 * @attention `numOfParticles` has to be _even!_
	 */
	for( size_t i = 0; i < numOfParticles; ++ i ) {
//		printf( "%zu %p\n", i, (void *) m );
		m[i] = ( (T) rand() ) / RAND_MAX;
	}

//	fprintf( stderr, "mass initialized\n" );

	/**
	 * In this case `xEntries` and `vEntries` are equal so I can merge the loops in
	 * one loop.
	 */
	for( size_t i = 0; i < numOfParticles; i += 2 ) {
		initialize< T, D >( x + D * i, m + i );
		initialize< T, D >( v + D * i, m + i );
	}

	fprintf( stderr, "done!\n" );
}
