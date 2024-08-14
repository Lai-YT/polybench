#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Default problem size. */
#ifndef TSTEPS
# define TSTEPS 10
#endif
#ifndef POLYBENCH_N
# define POLYBENCH_N 1024
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE X[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE A[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE B[POLYBENCH_N][POLYBENCH_N];
#else
DATA_TYPE** X = (DATA_TYPE**)malloc(MAXGRID * sizeof(DATA_TYPE*));
DATA_TYPE** A = (DATA_TYPE**)malloc(MAXGRID * sizeof(DATA_TYPE*));
DATA_TYPE** B = (DATA_TYPE**)malloc(MAXGRID * sizeof(DATA_TYPE*));
{
  int i;
  for (i = 0; i < POLYBENCH_N; ++i)
    {
      X[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
      A[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
      B[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
    }
}
#endif

static inline
void init_array()
{
  int i, j;

  for (i = 0; i < POLYBENCH_N; i++)
    for (j = 0; j < POLYBENCH_N; j++)
      {
	X[i][j] = ((DATA_TYPE) i*(j+1) + 1) / POLYBENCH_N;
	A[i][j] = ((DATA_TYPE) (i-1)*(j+4) + 2) / POLYBENCH_N;
	B[i][j] = ((DATA_TYPE) (i+3)*(j+7) + 3) / POLYBENCH_N;
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
      for (i = 0; i < POLYBENCH_N; i++)
	for (j = 0; j < POLYBENCH_N; j++) {
	  fprintf(stderr, DATA_PRINTF_MODIFIER, A[i][j]);
	  if ((i * POLYBENCH_N + j) % 80 == 20) fprintf(stderr, "\n");
	}
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int t, i1, i2;
  int n = POLYBENCH_N;
  int tsteps = TSTEPS;

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
// #pragma live-out X

  for (t = 0; t < tsteps; t++)
    {
      for (i1 = 0; i1 < n; i1++)
	for (i2 = 1; i2 < n; i2++)
	  {
	    X[i1][i2] = X[i1][i2] - X[i1][i2-1] * A[i1][i2] / B[i1][i2-1];
	    B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1][i2-1];
	  }

      for (i1 = 0; i1 < n; i1++)
	X[i1][n-1] = X[i1][n-1] / B[i1][n-1];

      for (i1 = 0; i1 < n; i1++)
	for (i2 = 0; i2 < n-2; i2++)
	  X[i1][n-i2-2] = (X[i1][n-2-i2] - X[i1][n-2-i2-1] * A[i1][n-i2-3]) / B[i1][n-3-i2];

      for (i1 = 1; i1 < n; i1++)
	for (i2 = 0; i2 < n; i2++) {
	  X[i1][i2] = X[i1][i2] - X[i1-1][i2] * A[i1][i2] / B[i1-1][i2];
	  B[i1][i2] = B[i1][i2] - A[i1][i2] * A[i1][i2] / B[i1-1][i2];
	}

      for (i2 = 0; i2 < n; i2++)
	X[n-1][i2] = X[n-1][i2] / B[n-1][i2];

      for (i1 = 0; i1 < n-2; i1++)
	for (i2 = 0; i2 < n; i2++)
	  X[n-2-i1][i2] = (X[n-2-i1][i2] - X[n-i1-3][i2] * A[n-3-i1][i2]) / B[n-2-i1][i2];
    }

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
