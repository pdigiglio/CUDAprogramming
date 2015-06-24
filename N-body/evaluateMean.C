#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

int round( long double val, long double err, FILE * stream) {

    /* dichiaro una variabile per l'esponente */
    short int exp = (short) log10( fabs( err ) );

    /* controllo che l'approssimazione sia corretta */
    long double tmp = err / powl( 10, exp );
    while ( !( tmp >= 1. && tmp < 10. ) ) {
        if ( tmp <= 1 ) exp --;
        else exp ++;

//        printf( "%hd\n", exp );
        tmp = err / powl( 10, exp );
    }

//    printf( "%Lg, %Lg, %hd\n", err, tmp, exp );

    /* controllo le cifre decimali da tenere */
    if ( tmp < 3. )
        tmp = err / powl( 10, -- exp );

//    printf( "%hd\n", exp);

    fprintf( stream, "%.16Lg\t%.16Lg",
            roundl( val / powl(10., exp) ) * powl(10., exp),
            roundl( tmp ) * powl(10., exp)
           );

    return exp;
}       /* -----  end of function round  ----- */

int main ( int argc, char *argv[] ) {
//int evaluateMean ( int argc, char *argv[] ) {

    if( argc != 3 ) {
        fprintf( stderr,
                "Wrong number of arguments!\n"
                "Usage:\n"
                "./evaluateMean <file-name> <(int)num-of-particles>\n" );

        return 1;
    }

    FILE *const inputFile = fopen( argv[1], "r" );
    if ( inputFile == NULL ) {
//        fprintf( stderr, "couldn't open file '%s': %s\n",
//                argv[1], strerror( errno ) );

        return 2;
    }

    long double tmp, mean = 0.;
    unsigned int i = 0;
    while( ! feof( inputFile ) ) {
        fscanf( inputFile, "%Lg\n", &tmp );

        mean += tmp;
        i ++;
    }

    mean /= i;

    // go to the beginning of file
    rewind( inputFile );

    long double stDev = 0.;
    while( ! feof( inputFile ) ) {
        fscanf( inputFile, "%Lg\n", &tmp );

        stDev += tmp * tmp - mean * mean;
    }

    stDev /= i - 1;

    fprintf( stdout, "%u\t", atoi( argv[2]) );
    round( mean, sqrtl( stDev / i ), stdout );
    fprintf( stdout, "\n" );

    if( fclose( inputFile ) == EOF ) { 
//        fprintf( stderr, "couldn't close file '%s': %s\n",
//                argv[1], strerror( errno ) );

        return 3;
    }

    return 0;
}
