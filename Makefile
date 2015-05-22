main=vectorAdd

FLAGS=-m64 -gencode arch=compute_30,code=sm_30

$(main): %: %.cu Makefile
	nvcc $(FLAGS) $< -o $@
