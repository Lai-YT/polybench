/**
 * gramschmidt.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#include "../../common/polybenchUtilFuncts.h"

#define GPU_DEVICE 0

/* Problem size */
#define M 2048
#define N 2048

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 256
#define DIM_THREAD_BLOCK_Y 1

/* Can switch DATA_TYPE between float and double */
#ifndef DATA_TYPE
#define DATA_TYPE float
#endif


void init_array(DATA_TYPE* A)
{
	int i, j;

	for (i = 0; i < M; i++)
	{
		for (j = 0; j < N; j++)
		{
			A[i*N + j] = ((DATA_TYPE) (i+1)*(j+1)) / (M+1);
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


__global__ void gramschmidt_kernel1(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid==0)
	{
		DATA_TYPE nrm = 0.0;
		int i;
		for (i = 0; i < M; i++)
		{
			nrm += a[i * N + k] * a[i * N + k];
		}
      		r[k * N + k] = sqrt(nrm);
	}
}


__global__ void gramschmidt_kernel2(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i < M)
	{	
		q[i * N + k] = a[i * N + k] / r[k * N + k];
	}
}


__global__ void gramschmidt_kernel3(DATA_TYPE *a, DATA_TYPE *r, DATA_TYPE *q, int k)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if ((j > k) && (j < N))
	{
		r[k*N + j] = 0.0;

		int i;
		for (i = 0; i < M; i++)
		{
			r[k*N + j] += q[i*N + k] * a[i*N + j];
		}
		
		for (i = 0; i < M; i++)
		{
			a[i*N + j] -= q[i*N + k] * r[k*N + j];
		}
	}
}


void gramschmidtCuda(DATA_TYPE* A, DATA_TYPE* R, DATA_TYPE* Q)
{
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 gridKernel1(1, 1);
	dim3 gridKernel2((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	dim3 gridKernel3((size_t)ceil(((float)N) / ((float)DIM_THREAD_BLOCK_X)), 1);
	
	DATA_TYPE *A_gpu;
	DATA_TYPE *R_gpu;
	DATA_TYPE *Q_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&R_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMalloc((void **)&Q_gpu, sizeof(DATA_TYPE) * M * N);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * M * N, cudaMemcpyHostToDevice);
	
	int k;
	for (k = 0; k < N; k++)
	{
		gramschmidt_kernel1<<<gridKernel1,block>>>(A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
		gramschmidt_kernel2<<<gridKernel2,block>>>(A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
		gramschmidt_kernel3<<<gridKernel3,block>>>(A_gpu, R_gpu, Q_gpu, k);
		cudaThreadSynchronize();
	}
	
	cudaMemcpy(A, A_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);    
	cudaMemcpy(R, R_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);
	cudaMemcpy(Q, Q_gpu, sizeof(DATA_TYPE) * M * N, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(R_gpu);
	cudaFree(Q_gpu);
}


int main(int argc, char *argv[])
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* R;
	DATA_TYPE* Q;
	
	A = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));
	R = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	Q = (DATA_TYPE*)malloc(M*N*sizeof(DATA_TYPE));  
	
	init_array(A);
	
	GPU_argv_init();
	
	t_start = rtclock();
	gramschmidtCuda(A, R, Q);
	t_end = rtclock();

#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lfs\n", t_end - t_start);
#endif
	
	free(A);
	free(R);
	free(Q);  

    	return 0;
}

