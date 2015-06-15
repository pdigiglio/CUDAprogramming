/**
 *
 *
 *           @name  createTimeEvolution.c
 *
 *        @version  1.0
 *           @date  06/01/2015 (03:21:46 PM)
 *       @revision  none
 *       @compiler  gcc
 *
 *
 *         @author  P. Di Giglio (github.com/pdigiglio), <p.digiglio91@gmail.com>
 *        @company  
 *          @brief  
 *
 *          Example usage:
 *          @code
 *          @endcode
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>

#include <string.h>
#include <ctype.h>
#include <errno.h>

unsigned long int detectPartcleNumber( FILE *input, bool print = false ) {
	unsigned long int wordCount = 0;
	/* load first line of file in memory */
	char tmp = fgetc( input );
	while ( tmp != '\n' ){


//		putchar( tmp );
		/* skip all spaces at the beginning */
		while( isspace(tmp) && tmp != '\n' )
			tmp = fgetc( input );

		/* takes into account trailing spaces at the end of the line */
		if( tmp == '\n' )
			break;

		/* update word counter */
		wordCount ++;

		/* a word has been found so I run through it */
		while( !(  isspace( tmp ) ) ) {
			tmp = fgetc( input );
//			putchar( tmp );
		}

//		if( tmp == '\n' )
//			break;
	}

	if ( print ) {
		printf( "%lu particle detected\n", ( wordCount - 1 ) / 3 );
		exit(1);
	}

	/* first column is time */
	return ( wordCount - 1 ) / 3;

}

int main ( int argc, char *argv[] ) {


	char inputFileName[] = "grav.txt";                  /* input-file name    */
	FILE *input= fopen( inputFileName, "r" );

	if ( ! input ) {
		fprintf ( stderr, "couldn't open file '%s'; %s\n",
				inputFileName, strerror(errno) );
		exit (EXIT_FAILURE);
	}

	const unsigned long int numOfParticles = detectPartcleNumber( input /*, true */ );
//	fprintf( stderr, "par: %lu\n", numOfParticles );
//	return 0;

	const char *rootFileName = "grav";
	char	outputFileName[100] = ""; /* output-file name    */
	FILE	*output = stderr;										/* output-file pointer */

	int n = 1;
	int dummy;
	long double centerOfMassPosition[3] = { 0, 0, 0 };
	long double dataInput[3];

	/* loop over lines in the input file */
	while ( !feof( input )  ) {
		sprintf( outputFileName, "%s.%06d.csv", rootFileName, n );
		output = fopen( outputFileName, "w" );
		if ( output == NULL ) {
			fprintf ( stderr, "couldn't open file '%s'; %s\n",
					outputFileName, strerror(errno) );
			exit (EXIT_FAILURE);
		}

		fprintf( stderr, "[%lu] opening %s\n", numOfParticles, outputFileName );

		fscanf( input, "%d\t", &dummy );

		for ( unsigned int d = 0; d < 3; ++ d ) 
			centerOfMassPosition[d] = 0.;

		/* loop over particles */
		fprintf( output, "x,y,z\n" );
		for( unsigned int j = 0; j < numOfParticles; ++ j ) {

			for ( unsigned int d = 0; d < 3; ++ d ) {
				fscanf( input, "%Lf\t", & dataInput[d] );
				centerOfMassPosition[d] += dataInput[d];
			}

			fprintf( output, "%.6Lg,%.6Lg,%.6Lg,\n", dataInput[0], dataInput[1], dataInput[2] );
		}
		fscanf( input, "\n" );

		fprintf( stderr, "center of mass position: ");
		fprintf( stdout, "%Lg, %Lg, %Lg\n",
				centerOfMassPosition[0], centerOfMassPosition[1], centerOfMassPosition[2] );
	

		if( fclose(output) == EOF ) {			/* close output file   */
			fprintf ( stderr, "couldn't close file '%s'; %s\n",
					outputFileName, strerror(errno) );
			exit (EXIT_FAILURE);
		}
		++ n;
	}

	if( fclose( input ) == EOF ) {			/* close input file   */
		fprintf ( stderr, "couldn't close file '%s'; %s\n",
				inputFileName, strerror(errno) );
		exit (EXIT_FAILURE);
	}

}
