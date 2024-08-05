/**
 * covariance.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#define M 2048
#define N 2048

/* Thread block dimensions for kernel 1*/
#define DIM_THREAD_BLOCK_KERNEL_1_X 256
#define DIM_THREAD_BLOCK_KERNEL_1_Y 1

/* Thread block dimensions for kernel 2*/
#define DIM_THREAD_BLOCK_KERNEL_2_X 32
#define DIM_THREAD_BLOCK_KERNEL_2_Y 8

/* Thread block dimensions for kernel 3*/
#define DIM_THREAD_BLOCK_KERNEL_3_X 256
#define DIM_THREAD_BLOCK_KERNEL_3_Y 1

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#define FLOAT_N 3214212.01
#define EPS 0.005

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

void init_arrays(DATA_TYPE* data)
{
	int i, j;

	for (i = 1; i < (M+1); i++)
	{
		for (j = 1; j < (N+1); j++)
		{
			data[i*(N+1) + j] = ((DATA_TYPE) i*j) / M;
		}
	}
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaCheckReturn(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE));
	cudaCheckReturn(cudaSetDevice(GPU_DEVICE));
	
	return;
}

__global__ void mean_kernel(DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;

	if ((j >= 1) && (j < (M+1)))
	{
		mean[j] = 0.0;

		int i;
		for(i = 1; i < (N+1); i++)
		{
			mean[j] += data[i * (M+1) + j];
		}
		mean[j] /= (DATA_TYPE)FLOAT_N;
	}
}

__global__ void reduce_kernel(DATA_TYPE *mean, DATA_TYPE *data)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		
	if ((i >= 1) && (i < (N+1)) && (j >= 1) && (j < (M+1)))
	{
		data[i * (M+1) + j] -= mean[j];	
	}
}

__global__ void covar_kernel(DATA_TYPE *symmat, DATA_TYPE *data)
{
	int j1 = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int i, j2;

	if ((j1 >= 1) && (j1 < (M+1)))
	{
		for (j2 = j1; j2 < (M+1); j2++)
		{		
			symmat[j1*(M+1) + j2] = 0.0;
			for(i = 1; i < (N+1); i++)
			{
				symmat[j1 * (M+1) + j2] += data[i *(M+1) + j1] * data[i *(M+1) + j2];
			}
			symmat[j2 * (M+1) + j1] = symmat[j1 * (M+1) + j2];
		}
	}
}

void covarianceCuda(DATA_TYPE* data, DATA_TYPE* symmat, DATA_TYPE* mean)
{
	DATA_TYPE *data_gpu;
	DATA_TYPE *mean_gpu;
	DATA_TYPE *symmat_gpu;

	cudaCheckReturn(cudaMalloc((void **)&data_gpu, sizeof(DATA_TYPE) * (M+1) * (N+1)));
	cudaCheckReturn(cudaMalloc((void **)&symmat_gpu, sizeof(DATA_TYPE) * (M+1) * (M+1)));
	cudaCheckReturn(cudaMalloc((void **)&mean_gpu, sizeof(DATA_TYPE) * (M+1)));
	cudaCheckReturn(cudaMemcpy(data_gpu, data, sizeof(DATA_TYPE) * (M+1) * (N+1), cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(symmat_gpu, symmat, sizeof(DATA_TYPE) * (M+1) * (M+1), cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(mean_gpu, mean, sizeof(DATA_TYPE) * (M+1), cudaMemcpyHostToDevice));
	
	dim3 block1(DIM_THREAD_BLOCK_KERNEL_1_X, DIM_THREAD_BLOCK_KERNEL_1_Y);
	dim3 grid1((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_1_X)), 1);
	
	dim3 block2(DIM_THREAD_BLOCK_KERNEL_2_X, DIM_THREAD_BLOCK_KERNEL_2_Y);
	dim3 grid2((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)), (size_t)(ceil((float)N) / ((float)DIM_THREAD_BLOCK_KERNEL_2_X)));
	
	dim3 block3(DIM_THREAD_BLOCK_KERNEL_3_X, DIM_THREAD_BLOCK_KERNEL_3_Y);
	dim3 grid3((size_t)(ceil((float)M) / ((float)DIM_THREAD_BLOCK_KERNEL_3_X)), 1);
	
	mean_kernel<<<grid1, block1>>>(mean_gpu, data_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());
	
	reduce_kernel<<<grid2, block2>>>(mean_gpu, data_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());

	covar_kernel<<<grid3, block3>>>(symmat_gpu, data_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());

	cudaCheckReturn(cudaMemcpy(data, data_gpu, sizeof(DATA_TYPE) * (M+1) * (N+1), cudaMemcpyDeviceToHost));
	cudaCheckReturn(cudaMemcpy(mean, mean_gpu, sizeof(DATA_TYPE) * (M+1), cudaMemcpyDeviceToHost));
	cudaCheckReturn(cudaMemcpy(symmat, symmat_gpu, sizeof(DATA_TYPE) * (M+1) * (N+1), cudaMemcpyDeviceToHost));

	cudaCheckReturn(cudaFree(data_gpu));
	cudaCheckReturn(cudaFree(symmat_gpu));
	cudaCheckReturn(cudaFree(mean_gpu));
}

int main()
{
	double t_start, t_end;

	DATA_TYPE* data;
	DATA_TYPE* symmat;
	DATA_TYPE* mean;

	data = (DATA_TYPE*)malloc((M+1)*(N+1)*sizeof(DATA_TYPE));
	symmat = (DATA_TYPE*)malloc((M+1)*(M+1)*sizeof(DATA_TYPE));
	mean = (DATA_TYPE*)malloc((M+1)*sizeof(DATA_TYPE));

	init_arrays(data);

	GPU_argv_init();

	t_start = rtclock();
	covarianceCuda(data, symmat, mean);
	t_end = rtclock();
#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lf\n", t_end - t_start);
#endif

	free(data);
	free(symmat);
	free(mean);

  	return 0;
}

