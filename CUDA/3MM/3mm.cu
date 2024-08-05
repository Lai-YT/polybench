/**
 * 3mm.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
# define NI 512
# define NJ 512
# define NK 512
# define NL 512
# define NM 512

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

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

void init_array(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D)
{
	int i, j;

	for (i = 0; i < NI; i++)
	{
		for (j = 0; j < NK; j++)
		{
			A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}
  
	for (i = 0; i < NK; i++)
	{
		for (j = 0; j < NJ; j++)
		{
			B[i*NJ + j] = ((DATA_TYPE) i*(j+1)) / NJ;
		}
	}
  
	for (i = 0; i < NJ; i++)
	{
		for (j = 0; j < NM; j++)
		{
			C[i*NM + j] = ((DATA_TYPE) i*(j+3)) / NL;
		}
	}
  
	for (i = 0; i < NM; i++)
	{
		for (j = 0; j < NL; j++)
		{
			D[i*NL + j] = ((DATA_TYPE) i*(j+2)) / NK;
		}
	}
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaCheckReturn(cudaGetDeviceProperties(&deviceProp, GPU_DEVICE));
	cudaCheckReturn(cudaSetDevice(GPU_DEVICE));
}

__global__ void mm3_kernel1(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *E)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{
		int k;
		for(k=0; k < NK; k++)
		{
			E[i * NJ + j] += A[i * NK + k] * B[k * NJ + j];
		}
	}
}

	
__global__ void mm3_kernel2(DATA_TYPE *C, DATA_TYPE *D, DATA_TYPE *F)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NJ) && (j < NL))
	{
		int k;
		for(k=0; k < NM; k++)
		{
			F[i * NL + j] += C[i * NM + k] * D[k * NL +j];
		}
	}
}

	
__global__ void mm3_kernel3(DATA_TYPE *E, DATA_TYPE *F, DATA_TYPE *G)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NL))
	{
		int k;
		for(k=0; k < NJ; k++)
		{
			G[i * NL + j] += E[i * NJ + k] * F[k * NL + j];
		}
	}
}

void mm3Cuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C, DATA_TYPE* D, DATA_TYPE* E, DATA_TYPE* F, 
		DATA_TYPE* G)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;
	DATA_TYPE *C_gpu;
	DATA_TYPE *D_gpu;
	DATA_TYPE *E_gpu;
	DATA_TYPE *F_gpu;
	DATA_TYPE *G_gpu;

	cudaCheckReturn(cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * NI * NK));
	cudaCheckReturn(cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * NK * NJ));
	cudaCheckReturn(cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * NJ * NM));
	cudaCheckReturn(cudaMalloc((void **)&D_gpu, sizeof(DATA_TYPE) * NM * NL));
	cudaCheckReturn(cudaMalloc((void **)&E_gpu, sizeof(DATA_TYPE) * NI * NJ));
	cudaCheckReturn(cudaMalloc((void **)&F_gpu, sizeof(DATA_TYPE) * NJ * NL));
	cudaCheckReturn(cudaMalloc((void **)&G_gpu, sizeof(DATA_TYPE) * NI * NL));

	cudaCheckReturn(cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * NI * NK, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * NK * NJ, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * NJ * NM, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(D_gpu, D, sizeof(DATA_TYPE) * NM * NL, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(E_gpu, E, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(F_gpu, F, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyHostToDevice));
	cudaCheckReturn(cudaMemcpy(G_gpu, G, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyHostToDevice));

	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid1((size_t)(ceil(((float)NJ) / ((float)DIM_THREAD_BLOCK_X))), (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))));
	dim3 grid2((size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))), (size_t)(ceil((float)NJ / ((float)DIM_THREAD_BLOCK_Y))));
	dim3 grid3((size_t)(ceil(((float)NL) / ((float)DIM_THREAD_BLOCK_X))), (size_t)(ceil((float)NI / ((float)DIM_THREAD_BLOCK_Y))));

	mm3_kernel1<<<grid1,block>>>(A_gpu, B_gpu, E_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());

	mm3_kernel2<<<grid2,block>>>(C_gpu, D_gpu, F_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());

	mm3_kernel3<<<grid3,block>>>(E_gpu, F_gpu, G_gpu);
	cudaCheckKernel();
	cudaCheckReturn(cudaDeviceSynchronize());

	cudaCheckReturn(cudaMemcpy(E, E_gpu, sizeof(DATA_TYPE) * NI * NJ, cudaMemcpyDeviceToHost));
	cudaCheckReturn(cudaMemcpy(F, F_gpu, sizeof(DATA_TYPE) * NJ * NL, cudaMemcpyDeviceToHost));
	cudaCheckReturn(cudaMemcpy(G, G_gpu, sizeof(DATA_TYPE) * NI * NL, cudaMemcpyDeviceToHost));

	cudaCheckReturn(cudaFree(A_gpu));
	cudaCheckReturn(cudaFree(B_gpu));
	cudaCheckReturn(cudaFree(C_gpu));
	cudaCheckReturn(cudaFree(D_gpu));
	cudaCheckReturn(cudaFree(E_gpu));
	cudaCheckReturn(cudaFree(F_gpu));
	cudaCheckReturn(cudaFree(G_gpu));
}

int main(int argc, char** argv)
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;
	DATA_TYPE* C;
	DATA_TYPE* D;
	DATA_TYPE* E;
	DATA_TYPE* F;
	DATA_TYPE* G;

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));
	C = (DATA_TYPE*)malloc(NJ*NM*sizeof(DATA_TYPE));
	D = (DATA_TYPE*)malloc(NM*NL*sizeof(DATA_TYPE));
	E = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE));
	F = (DATA_TYPE*)malloc(NJ*NL*sizeof(DATA_TYPE));
	G = (DATA_TYPE*)malloc(NI*NL*sizeof(DATA_TYPE));

	init_array(A, B, C, D);

	GPU_argv_init();

	t_start = rtclock();
	mm3Cuda(A, B, C, D, E, F, G);
	t_end = rtclock();

#ifdef POLYBENCH_TIME
	fprintf(stdout, "%0.6lf\n", t_end - t_start);
#endif

	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(F);
	free(G);

	return 0;
}

