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
DATA_TYPE w;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE a[POLYBENCH_N+1][POLYBENCH_N+1];
DATA_TYPE x[POLYBENCH_N+1];
DATA_TYPE y[POLYBENCH_N+1];
DATA_TYPE b[POLYBENCH_N+1];
#else
DATA_TYPE** a = (DATA_TYPE**)malloc((POLYBENCH_N + 1) * sizeof(DATA_TYPE*));
DATA_TYPE* x = (DATA_TYPE*)malloc((POLYBENCH_N + 1) * sizeof(DATA_TYPE));
DATA_TYPE* y = (DATA_TYPE*)malloc((POLYBENCH_N + 1) * sizeof(DATA_TYPE));
DATA_TYPE* b = (DATA_TYPE*)malloc((POLYBENCH_N + 1) * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i <= POLYBENCH_N; ++i)
    a[i] = (DATA_TYPE*)malloc((POLYBENCH_N + 1) * sizeof(DATA_TYPE));
}
#endif

static inline
void init_array()
{
  int i, j;

  for (i = 0; i <= POLYBENCH_N; i++)
    {
      x[i] = ((DATA_TYPE) i + 1) / POLYBENCH_N;
      b[i] = ((DATA_TYPE) i + 2) / POLYBENCH_N;
      for (j = 0; j <= POLYBENCH_N; j++)
	a[i][j] = ((DATA_TYPE) i*j + 1) / POLYBENCH_N;
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
      for (i = 0; i <= POLYBENCH_N; i++) {
	fprintf(stderr, DATA_PRINTF_MODIFIER, x[i]);
	if (i % 80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int i, j, k;
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
// #pragma live-out x

  b[0] = 1.0;
  for (i = 0; i < n; i++)
    {
      for (j = i+1; j <= n; j++)
        {
	  w = a[j][i];
	  for (k = 0; k < i; k++)
	    w = w- a[j][k] * a[k][i];
	  a[j][i] = w / a[i][i];
        }
      for (j = i+1; j <= n; j++)
        {
	  w = a[i+1][j];
	  for (k = 0; k <= i; k++)
	    w = w  - a[i+1][k] * a[k][j];
	  a[i+1][j] = w;
        }
    }
  y[0] = b[0];
  for (i = 1; i <= n; i++)
    {
      w = b[i];
      for (j = 0; j < i; j++)
	w = w - a[i][j] * y[j];
      y[i] = w;
    }
  x[n] = y[n] / a[n][n];
  for (i = 0; i <= n - 1; i++)
    {
      w = y[n - 1 - (i)];
      for (j = n - i; j <= n; j++)
	w = w - a[n - 1 - (i)][j] * x[j];
      x[n - 1 - (i)] = w / a[n - 1 - (i)][n - 1-(i)];
    }

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
