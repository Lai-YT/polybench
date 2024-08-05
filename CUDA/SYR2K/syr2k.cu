/**
 * syr2k.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

/* Problem size */
#define N 2048
#define M 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 12435
#define BETA 4546

/* Can switch DATA_TYPE between float and double */
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif

#define cudaCheckReturn(ret) \
	do { \
		cudaError_t cudaCheckReturn_e = (ret); \
		if (cudaCheckReturn_e != cudaSuccess) { \
			fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaCheckReturn_e)); \
			fflush(stderr); \
		} \
		assert(cudaCheckReturn_e == cudaSuccess); \
	} while(0)

#define cudaCheckKernel() \
	do { \
		cudaCheckReturn(cudaGetLastError()); \
	} while(0)

void init_arrays(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;
  
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			C[i*N + j] = ((DATA_TYPE) i*j + 2) / N;
		}

		for (j = 0; j < M; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
			B[i*N + j] = ((DATA_TYPE) i*j + 1) / N;
		}
	}
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaCheckReturn(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE));
	cudaCheckReturn(cudaSetDevice(GPU_DEVICE));
}


__global__ void syr2k_kernel(DATA_TYPE *a, DATA_TYPE *b, DATA_TYPE *c)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < N) && (j < N))
	{
		c[i * N + j] *= BETA;
		
		int k;
		for(k = 0; k < M; k++)
		{
			c[i * N + j] += ALPHA * a[i * M + k] * b[j * M + k] + ALPHA * b[i * M + k] * a[j * M + k];
		}
	}
}


void syr2kCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;

	cudaCheckReturn(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * N * M));
	cudaCheckReturn(cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * N * M));
	cudaCheckReturn(cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * N * N));
	cudaCheckReturn(cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * N * M, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice));
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_X) ), (size_t)(ceil( ((float)N) / ((float)DIM_THREAD_BLOCK_Y) )));
	
	syr2k_kernel<<<grid,block>>>(A_gpu,B_gpu,C_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());
	
	cudaCheckReturn(cudaMemcpy(C, C_gpu, sizeof(DATA_TYPE) * N * N, cudaMemcpyDeviceToHost));

	cudaCheckReturn(cudaFree(A_gpu));
	cudaCheckReturn(cudaFree(B_gpu));
	cudaCheckReturn(cudaFree(C_gpu));
}


int main()
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;

	A = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(N*M*sizeof(DATA_TYPE));

	init_arrays(A, B, C);
    
	GPU_argv_init();
	
	t_start = rtclock();
	syr2kCuda(A, B, C);
	t_end = rtclock();
#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lf\n", t_end - t_start);
#endif

	free(A);
	free(B);
	free(C);

  	return 0;
}

