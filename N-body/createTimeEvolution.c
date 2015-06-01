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
#include <errno.h>

int main ( int argc, char *argv[] ) {

	char inputFileName[] = "grav.txt";                  /* input-file name    */
	FILE *input          = fopen( inputFileName, "r" );	/* input-file pointer */
	if ( input == NULL ) {
		fprintf ( stderr, "couldn't open file '%s'; %s\n",
				inputFileName, strerror(errno) );
		exit (EXIT_FAILURE);
	}

	const char *rootFileName = "grav";
	char	outputFileName[100] = ""; /* output-file name    */
	FILE	*output = stderr;										/* output-file pointer */

	int n = 1;
	int dummy;
	long double centerOfMassPosition[3] = { 0, 0, 0 };
	long double dataInput[3];

	while ( !feof( input )  ) {
		sprintf( outputFileName, "%s.%06d.csv", rootFileName, n );
		output = fopen( outputFileName, "w" );
		if ( output == NULL ) {
			fprintf ( stderr, "couldn't open file '%s'; %s\n",
					outputFileName, strerror(errno) );
			exit (EXIT_FAILURE);
		}

		fprintf( stderr, "opening %s\n", outputFileName );

		fscanf( input, "%d\t", &dummy );

		for ( unsigned int d = 0; d < 3; ++ d ) 
			centerOfMassPosition[d] = 0.;

		for( unsigned int j = 0; j < 4; ++ j ) {

			for ( unsigned int d = 0; d < 3; ++ d ) {
				fscanf( input, "%Lf\t", & dataInput[d] );
				centerOfMassPosition[d] += dataInput[d];
			}

			fprintf( output, "%.6Lg,%.6Lg,%.6Lg\n", dataInput[0], dataInput[1], dataInput[2] );
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



	if( fclose(input) == EOF ) {			/* close input file   */
		fprintf ( stderr, "couldn't close file '%s'; %s\n",
				inputFileName, strerror(errno) );
		exit (EXIT_FAILURE);
	}

}
