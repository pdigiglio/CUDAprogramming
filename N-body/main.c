#include "integrator.h"

// Always include `glew.h` bedore `gl.h` and `glfw.h`
#include <GL/glew.h>

// openGL libraries
#include <GL/glut.h>
#include <GLFW/glfw3.h>


#include <stdlib.h>
#include <stdio.h>

const unsigned int numOfParticles = 1;

	int
main ( int argc, char *argv[] ) {
	
	printf("%s Starting...\n\n", argv[0]);

	// initialize GLFW
	if( !glfwInit() ) {
		fprintf( stderr, "Failed to initialize GLFW\n" );
		exit( EXIT_FAILURE );
	}

	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	
	GLFWwindow *window = glfwCreateWindow( 1024, 786, argv[0], NULL, NULL );
	if( !window ) {
		fprintf( stderr,  "Failed to opend GLFW window!\n" );
		glfwTerminate();
		exit( EXIT_FAILURE );
	}

	glfwMakeContextCurrent( window );
	glewExperimental = true; // ?
	if( glewInit() != GLEW_OK ) {
		fprintf( stderr, "Failed to initialize GLEW!\n" );
		exit( EXIT_FAILURE );
	}

	// ensure we can caputure the ESC key when pressed
	glfwSetInputMode( window, GLFW_STICKY_KEYS, GL_TRUE );

	// Dark blue background
	glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

	do {
		glfwSwapBuffers( window );
		glfwPollEvents();
	} // check if ESC key was pressed or window was closed
	while ( glfwGetKey( window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
			glfwWindowShouldClose( window ) == 0 );

	glfwTerminate();

	fprintf( stderr, "... bye from %s\n", argv[0] );

	// uncomment this just to test the openGL part
	return 0;

	/* taken from 0_Simple/cudaOpenMp/cudaOpenMP.cu
    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
	
	cudaGetDeviceCount(&num_gpus);

    if (num_gpus < 1)
    {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++)
    {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");

	*/

	float x[numOfParticles] = {};
	float v[numOfParticles] = {};

	for ( unsigned t = 0; t < 1000000; ++ t ) {
//		leapfrogVerlet( x, v, numOfParticles );
		rungeKutta( x, v, numOfParticles );
		printf( "%u %.6g %.6g\n", t, x[0], v[0] );
	}

	/*
     * `cudaDeviceReset()` causes the driver to clean up all state. While
     * not mandatory in normal operation, it is good practice.  It is also
     * needed to ensure correct operation when the application is being
     * profiled. Calling `cudaDeviceReset()` causes all profile data to be
     * flushed before the application exits.
	 */
//    cudaDeviceReset();

	return 0;
}
