/**
 * mvt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define N 4096

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

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

void init_array(DATA_TYPE* A, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y1, DATA_TYPE* y2)
{
	int i, j;

	for (i = 0; i < N; i++)
	{
		x1[i] = ((DATA_TYPE) i) / N;
		x2[i] = ((DATA_TYPE) i + 1) / N;
		y1[i] = ((DATA_TYPE) i + 3) / N;
		y2[i] = ((DATA_TYPE) i + 4) / N;
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) i*j) / N;
		}
	}
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaCheckReturn(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE));
	cudaCheckReturn(cudaSetDevice(GPU_DEVICE));
}


__global__ void mvt_kernel1(DATA_TYPE *a, DATA_TYPE *x1, DATA_TYPE *y_1)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x1[i] += a[i * N + j] * y_1[j];
		}
	}
}


__global__ void mvt_kernel2(DATA_TYPE *a, DATA_TYPE *x2, DATA_TYPE *y_2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < N)
	{
		int j;
		for(j=0; j < N; j++)
		{
			x2[i] += a[j * N + i] * y_2[j];	
		}
	}
}

void mvtCuda(DATA_TYPE* a, DATA_TYPE* x1, DATA_TYPE* x2, DATA_TYPE* y_1, DATA_TYPE* y_2)
{
	DATA_TYPE* a_gpu;
	DATA_TYPE* x1_gpu;
	DATA_TYPE* x2_gpu;
	DATA_TYPE* y_1_gpu;
	DATA_TYPE* y_2_gpu;

	cudaCheckReturn(cudaMalloc((void **)&a_gpu, sizeof(DATA_TYPE) * N * N));
	cudaCheckReturn(cudaMalloc((void **)&x1_gpu, sizeof(DATA_TYPE) * N));
	cudaCheckReturn(cudaMalloc((void **)&x2_gpu, sizeof(DATA_TYPE) * N));
	cudaCheckReturn(cudaMalloc((void **)&y_1_gpu, sizeof(DATA_TYPE) * N));
	cudaCheckReturn(cudaMalloc((void **)&y_2_gpu, sizeof(DATA_TYPE) * N));
	cudaCheckReturn(cudaMemcpy(a_gpu, a, sizeof(DATA_TYPE) * N * N, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(x1_gpu, x1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(x2_gpu, x2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(y_1_gpu, y_1, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(y_2_gpu, y_2, sizeof(DATA_TYPE) * N, cudaMemcpyHostToDevice));
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil((float)N/ ((float)DIM_THREAD_BLOCK_X)), 1);
	
	mvt_kernel1<<<grid,block>>>(a_gpu,x1_gpu,y_1_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());
	mvt_kernel2<<<grid,block>>>(a_gpu,x2_gpu,y_2_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());

	cudaCheckReturn(cudaMemcpy(x1, x1_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost));
	cudaCheckReturn(cudaMemcpy(x2, x2_gpu, sizeof(DATA_TYPE) * N, cudaMemcpyDeviceToHost));    
	
	cudaCheckReturn(cudaFree(a_gpu));
	cudaCheckReturn(cudaFree(x1_gpu));
	cudaCheckReturn(cudaFree(x2_gpu));
	cudaCheckReturn(cudaFree(y_1_gpu));
	cudaCheckReturn(cudaFree(y_2_gpu));
}


int main()
{
	double t_start, t_end;

	DATA_TYPE* a;
	DATA_TYPE* x1;
	DATA_TYPE* x2;
	DATA_TYPE* y_1;
	DATA_TYPE* y_2;

	a = (DATA_TYPE*)malloc(N*N*sizeof(DATA_TYPE));
	x1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	x2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_1 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));
	y_2 = (DATA_TYPE*)malloc(N*sizeof(DATA_TYPE));

	init_array(a, x1, x2, y_1, y_2);
	
	GPU_argv_init();

	t_start = rtclock();
	mvtCuda(a, x1, x2, y_1, y_2);
	t_end = rtclock();
#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lf\n", t_end - t_start);
#endif
	
	free(a);
	free(x1);
	free(x2);
	free(y_1);
	free(y_2);

  	return 0;
}

