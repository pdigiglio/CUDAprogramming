main=vectorAdd

$(main): %: %.cu
	nvcc $< -o $@ -gencode arch=compute_30,code=sm_30
