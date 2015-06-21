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
inline void allocatePointer( T* &x, size_t xEntries, const char name[] = "pointer" ) {
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
inline void initializeRandom( T *x, const T *m, T scale = 1. ) {
	// random for first entry
	x[0] = (T) ( scale * 2 * ( (T) rand() / RAND_MAX - .5 ) );
	x[1] = (T) ( scale * 2 * ( (T) rand() / RAND_MAX - .5 ) );
	x[2] = (T) ( scale * 2 * ( (T) rand() / RAND_MAX - .5 ) );

	// not random for the second entry
	// XXX masses are pair-wise equal
	x[3] = - /* ( m[0] / m[1] ) * */ x[0];
	x[4] = - /* ( m[0] / m[1] ) * */ x[1];
	x[5] = - /* ( m[0] / m[1] ) * */ x[2];
}

/**
 * @param phi angle in radiants
 * @param radius
 */
template <typename T, short int D> //, long double tollerance, long double radius>
inline void initializeRing ( T *x, const T *m, T phi, T tollerance = .2, T radius = 1. ) {
	radius += tollerance * 2 * ( (T) rand() / RAND_MAX - .5 );

	x[0] = radius * cos( phi );
	x[1] = radius * sin( phi );
	x[2] = (T) 0;


	// masses are always pair-wise couppled
	x[3] = - /* ( m[0] / m[1] ) * */ x[0];
	x[4] = - /* ( m[0] / m[1] ) * */ x[1];
	x[5] = (T) 0;
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
	for( size_t i = 0; i < numOfParticles; i += 2 ) {
//		printf( "%zu %p\n", i, (void *) m );

		// shift masses to prevent 0 value
		m[i  ] = ( (T) rand() ) / RAND_MAX + .000001;
		m[i+1] = m[i];
	}

//	fprintf( stderr, "mass initialized\n" );

	/**
	 * In this case `xEntries` and `vEntries` are equal so I can merge the loops in
	 * one loop.
	 */

	// half of angle step since index is incremented by 2 every time
	const T phiStep = (T) M_PI / numOfParticles;
	for( size_t i = 0; i < numOfParticles; i += 2 ) {
		initializeRing   < T, D > ( x + D * i, m + i, i * phiStep );
		initializeRandom < T, D > ( v + D * i, m + i );
	}

	fprintf( stderr, "done!\n" );
}
