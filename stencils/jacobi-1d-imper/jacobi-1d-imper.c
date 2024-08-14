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
# define TSTEPS 10000
#endif
#ifndef POLYBENCH_N
# define POLYBENCH_N 4096
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
DATA_TYPE A[POLYBENCH_N];
DATA_TYPE B[POLYBENCH_N];
#else
DATA_TYPE* A = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* B = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
#endif

static inline
void init_array()
{
  int i, j;

  for (i = 0; i < POLYBENCH_N; i++)
    {
      A[i] = ((DATA_TYPE) 4 * i + 10) / POLYBENCH_N;
      B[i] = ((DATA_TYPE) 7 * i + 11) / POLYBENCH_N;
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
	fprintf(stderr, DATA_PRINTF_MODIFIER, A[i]);
	if (i % 80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int t, i, j;
  int tsteps = TSTEPS;
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
// #pragma live-out A

  for (t = 0; t < tsteps; t++)
    {
      for (i = 2; i < n - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);

      for (j = 2; j < n - 1; j++)
	A[j] = B[j];
    }

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
