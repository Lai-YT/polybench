#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Default problem size. */
#ifndef POLYBENCH_N
# define POLYBENCH_N 4000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE alpha;
DATA_TYPE beta;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE B[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE x[POLYBENCH_N];
DATA_TYPE y[POLYBENCH_N];
DATA_TYPE tmp[POLYBENCH_N];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE* x = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* y = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* tmp = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < POLYBENCH_N; ++i)
    {
      A[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
      B[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
    }
}
#endif

static inline
void init_array()
{
  int i, j;

  alpha = 43532;
  beta = 12313;
  for (i = 0; i < POLYBENCH_N; i++)
    {
      x[i] = ((DATA_TYPE) i) / POLYBENCH_N;
      for (j = 0; j < POLYBENCH_N; j++)
	A[i][j] = ((DATA_TYPE) i*j) / POLYBENCH_N;
    }
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
      for (i = 0; i < POLYBENCH_N; i++) {
	fprintf(stderr, "%0.2lf ", y[i]);
	if (i%80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int i, j;
  int n = POLYBENCH_N;

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

#pragma scop
// #pragma live-out y

  for (i = 0; i < n; i++)
    {
      tmp[i] = 0;
      y[i] = 0;
      for (j = 0; j < n; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
    }

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
