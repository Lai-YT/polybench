/**
 * syrk.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

/* Problem size */
#define N 1024
#define M 1024

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for alpha and beta (same as values in PolyBench 2.0) */
#define alpha 12435
#define beta 4546

/* Can switch DATA_TYPE between float and double */
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif


void init_arrays(DATA_TYPE* A, DATA_TYPE* C)
{
	int i, j;
	
	for (i = 0; i < N; i++)
    	{
		for (j = 0; j < M; j++)
		{
			A[i*M + j] = ((DATA_TYPE) i*j) / N;
		}
		
		for (j = 0; j < N; j++)
		{
			C[i*M + j] = ((DATA_TYPE) i*j + 2) / N;
		}
	}
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	cudaSetDevice( GPU_DEVICE );
	
	return;
}


__global__ void syrk_kernel(DATA_TYPE ALPHA, DATA_TYPE BETA, DATA_TYPE *a, DATA_TYPE *c)
{
	/*  C := alpha*A*A' + beta*C */
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < N) && (j < N))
	{
		c[i * N + j] *= beta;
		int k;		
		for(k=0; k< M; k++)
		{
			c[i * N + j] += alpha * a[i * M + k] * a[j * M + k];
		}
	}
}


void syrkCuda(DATA_TYPE* A, DATA_TYPE* C)
{
	DATA_TYPE* A_gpu;
	DATA_TYPE* C_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M);
	cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * N * N);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice);
	cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)(ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X))), (size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_Y)));
	syrk_kernel<<<grid,block>>>(alpha, beta, A_gpu,C_gpu);
	cudaThreadSynchronize();

	cudaMemcpy(C, C_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(C_gpu);
}


int main()
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* C;

	A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	init_arrays(A, C);
	
	GPU_argv_init();	

	t_start = rtclock();
	syrkCuda(A, C);
	t_end = rtclock();

#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lfs\n", t_end - t_start);
#endif

	free(A);
	free(C);

	return 0;
}

