ifeq ($(COMPILE_CUDA_WITH_CLANG),1)
	CUDACXX=clang++-18
	CUDAFLAGS=--cuda-gpu-arch="sm_86" -lcudart -lrt -pthread -ldl -L/usr/local/cuda/lib64
else
	CUDACXX=nvcc
	CUDAFLAGS=
endif

all:
	$(CUDACXX) $(CUDAFLAGS) \
		-O3 \
		-DDATA_TYPE=double -DPOLYBENCH_TIME \
		${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe
