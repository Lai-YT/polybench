/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

/* Problem size. */
#define NX 4096
#define NY 4096

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


void init_array(DATA_TYPE *x, DATA_TYPE *A)
{
	int i, j;

	for (i = 0; i < NX; i++)
	{
		x[i] = i * M_PI;
		for (j = 0; j < NY; j++)
		{
			A[i*NY + j] = ((DATA_TYPE) i*(j)) / NX;
		}
	}
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaCheckReturn(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE));
	cudaCheckReturn(cudaSetDevice(GPU_DEVICE));
}

__global__ void atax_kernel1(DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < NX)
	{
		int j;
		for(j=0; j < NY; j++)
		{
			tmp[i] += A[i * NY + j] * x[j];
		}
	}
}

__global__ void atax_kernel2(DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < NY)
	{
		int i;
		for(i=0; i < NX; i++)
		{
			y[j] += A[i * NY + j] * tmp[i];
		}
	}
}

void ataxGpu(DATA_TYPE* A, DATA_TYPE* x, DATA_TYPE* y, DATA_TYPE* tmp)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *x_gpu;
	DATA_TYPE *y_gpu;
	DATA_TYPE *tmp_gpu;

	cudaCheckReturn(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NX * NY));
	cudaCheckReturn(cudaMalloc((void **)&x_gpu, sizeof(DATA_TYPE) * NY));
	cudaCheckReturn(cudaMalloc((void **)&y_gpu, sizeof(DATA_TYPE) * NY));
	cudaCheckReturn(cudaMalloc((void **)&tmp_gpu, sizeof(DATA_TYPE) * NX));
	
	cudaCheckReturn(cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NX * NY, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(x_gpu, x, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(y_gpu, y, sizeof(DATA_TYPE) * NY, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(tmp_gpu, tmp, sizeof(DATA_TYPE) * NX, cudaMemcpyHostToDevice));
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil( ((float)NX) / ((float)block.x) )), 1);
	dim3 grid2((size_t)(ceil( ((float)NY) / ((float)block.x) )), 1);

	atax_kernel1<<< grid1, block >>>(A_gpu,x_gpu,tmp_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());
	atax_kernel2<<< grid2, block >>>(A_gpu,y_gpu,tmp_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());
	
	cudaCheckReturn(cudaMemcpy(tmp, tmp_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost));
	cudaCheckReturn(cudaMemcpy(y, y_gpu, sizeof(DATA_TYPE) * NX, cudaMemcpyDeviceToHost));

	cudaCheckReturn(cudaFree(A_gpu));
	cudaCheckReturn(cudaFree(x_gpu));
	cudaCheckReturn(cudaFree(y_gpu));
	cudaCheckReturn(cudaFree(tmp_gpu));
}

int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* x;
	DATA_TYPE* y;
	DATA_TYPE* tmp;

	A = (DATA_TYPE*)malloc(NX*NY*sizeof(DATA_TYPE));
	x = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	y = (DATA_TYPE*)malloc(NY*sizeof(DATA_TYPE));
	tmp = (DATA_TYPE*)malloc(NX*sizeof(DATA_TYPE));

	init_array(x, A);

	GPU_argv_init();
	
	t_start = rtclock();
	ataxGpu(A, x, y, tmp);
	t_end = rtclock();

#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lf\n", t_end - t_start);
#endif

	free(A);
	free(x);
	free(y);
	free(tmp);

  	return 0;
}

