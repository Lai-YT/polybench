#include <assert.h>
#include <stdio.h>
#include "2mm_kernel.hu"
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Default problem size. */
#ifndef NI
# define NI 512
#endif
#ifndef NJ
# define NJ 512
#endif
#ifndef NK
# define NK 512
#endif
#ifndef NL
# define NL 512
#endif


/* Default data type is double (dgemm). */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE alpha1;
DATA_TYPE beta1;
DATA_TYPE alpha2;
DATA_TYPE beta2;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE C[NI][NJ];
DATA_TYPE A[NI][NK];
DATA_TYPE B[NK][NJ];
DATA_TYPE D[NJ][NL];
DATA_TYPE E[NI][NL];
#else
DATA_TYPE** C = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
DATA_TYPE** A = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(NK * sizeof(DATA_TYPE*));
DATA_TYPE** D = (DATA_TYPE**)malloc(NJ * sizeof(DATA_TYPE*));
DATA_TYPE** E = (DATA_TYPE**)malloc(NI * sizeof(DATA_TYPE*));
{
  int i;
  for (i = 0; i < NI; ++i)
    {
      C[i] = (DATA_TYPE*)malloc(NJ * sizeof(DATA_TYPE));
      A[i] = (DATA_TYPE*)malloc(NK * sizeof(DATA_TYPE));
      E[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
    }
  for (i = 0; i < NK; ++i)
    B[i] = (DATA_TYPE*)malloc(NJ * sizeof(DATA_TYPE));
  for (i = 0; i < NJ; ++i)
    D[i] = (DATA_TYPE*)malloc(NL * sizeof(DATA_TYPE));
}
#endif

static inline
void init_array()
{
  int i, j;

  alpha1 = 32412;
  beta1 = 2123;
  alpha2 = 132412;
  beta2 = 92123;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i][j] = ((DATA_TYPE) i*j)/NI;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i][j] = ((DATA_TYPE) i*j + 1)/NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NJ; i++)
    for (j = 0; j < NL; j++)
      D[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NL; j++)
      E[i][j] = ((DATA_TYPE) i*j + 2)/NJ;
}

/* Define the live-out variables. Code is not executed unless
   POLYBENCH_DUMP_ARRAYS is defined. */
static inline
void print_array(int argc, char** argv)
{
  int i, j;
#ifndef POLYBENCH_DUMP_ARRAYS
  if (argc > 42 && ! strcmp(argv[0], ""))
#endif
    {
      for (i = 0; i < NI; i++) {
	for (j = 0; j < NL; j++) {
	  fprintf(stderr, "%0.2lf ", E[i][j]);
	  if ((i * NI + j) % 80 == 20) fprintf(stderr, "\n");
	}
	fprintf(stderr, "\n");
      }
    }
}


int main(int argc, char** argv)
{
  int i, j, k;
  int ni = NI;
  int nj = NJ;
  int nk = NK;
  int nl = NL;

  /* Initialize array. */
  init_array();

#if defined(__CUDACC__) && defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  /* Start timer. */
  polybench_start_instruments;

#if defined(__CUDACC__) && !defined(POLYBENCH_TIME_NO_CUDA_INIT_CTX)
  cudaSetDevice(CUDA_DEVICE);
#endif

  #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
  if ((ni >= 1 && nj >= 1) || (ni >= 1 && nl >= 1)) {
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

    float (*dev_A)[2048];
    float (*dev_B)[2048];
    float (*dev_C)[2048];
    float (*dev_D)[2048];
    float (*dev_E)[2048];

    cudaCheckReturn(cudaMalloc((void **) &dev_A, (ppcg_min(2048, ni)) * (2048) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_B, (ppcg_min(2048, nk)) * (2048) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_C, (ppcg_min(2048, ni)) * (2048) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_D, (ppcg_min(2048, nj)) * (2048) * sizeof(float)));
    cudaCheckReturn(cudaMalloc((void **) &dev_E, (ppcg_min(2048, ni)) * (2048) * sizeof(float)));

    if (nj >= 1 && nk >= 1) {
      cudaCheckReturn(cudaMemcpy(dev_A, A, (ppcg_min(2048, ni)) * (2048) * sizeof(float), cudaMemcpyHostToDevice));
      cudaCheckReturn(cudaMemcpy(dev_B, B, (ppcg_min(2048, nk)) * (2048) * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (nj >= 1) {
      cudaCheckReturn(cudaMemcpy(dev_C, C, (ppcg_min(2048, ni)) * (2048) * sizeof(float), cudaMemcpyHostToDevice));
      if (nl >= 1) {
        cudaCheckReturn(cudaMemcpy(dev_D, D, (ppcg_min(2048, nj)) * (2048) * sizeof(float), cudaMemcpyHostToDevice));
      }
    }
    if (nl >= 1) {
      cudaCheckReturn(cudaMemcpy(dev_E, E, (ppcg_min(2048, ni)) * (2048) * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (nj >= 1) {
      {
        dim3 k0_dimBlock(16, 32);
        dim3 k0_dimGrid(ppcg_min(256, (nj + 31) / 32), ppcg_min(256, (ni + 31) / 32));
        kernel0 <<<k0_dimGrid, k0_dimBlock>>> (dev_A, dev_B, dev_C, ni, nl, nj, nk);
        cudaCheckKernel();
      }

    }
    if (nl >= 1) {
      {
        dim3 k1_dimBlock(16, 32);
        dim3 k1_dimGrid(ppcg_min(256, (nl + 31) / 32), ppcg_min(256, (ni + 31) / 32));
        kernel1 <<<k1_dimGrid, k1_dimBlock>>> (dev_C, dev_D, dev_E, ni, nl, nj, nk);
        cudaCheckKernel();
      }

    }
    if (nj >= 1) {
      cudaCheckReturn(cudaMemcpy(C, dev_C, (ppcg_min(2048, ni)) * (2048) * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (nl >= 1) {
      cudaCheckReturn(cudaMemcpy(E, dev_E, (ppcg_min(2048, ni)) * (2048) * sizeof(float), cudaMemcpyDeviceToHost));
    }
    cudaCheckReturn(cudaFree(dev_A));
    cudaCheckReturn(cudaFree(dev_B));
    cudaCheckReturn(cudaFree(dev_C));
    cudaCheckReturn(cudaFree(dev_D));
    cudaCheckReturn(cudaFree(dev_E));
  }

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
