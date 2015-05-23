main=vectorAdd

$(main): %: %.cu Makefile
	nvcc $< -o $@
	
# -gencode arch=compute_10,code=sm_10
