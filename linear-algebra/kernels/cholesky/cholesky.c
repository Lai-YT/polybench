#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

#ifndef POLYBENCH_N
# define POLYBENCH_N 1024
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE x;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE a[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE p[POLYBENCH_N];
#else
DATA_TYPE** a = (DATA_TYPE**) malloc((POLYBENCH_N) * sizeof(DATA_TYPE*));
DATA_TYPE* p = (DATA_TYPE*) malloc ((POLYBENCH_N) * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < nx; ++i)
    a[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
}
#endif


static inline
void init_array()
{
    int i, j;

    for (i = 0; i < POLYBENCH_N; i++)
      {
	p[i] = M_PI * i;
        for (j = 0; j < POLYBENCH_N; j++)
	  a[i][j] = M_PI * i + 2 * j;
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
	  for (j = 0; j < POLYBENCH_N; j++) {
	    fprintf(stderr, "%0.2lf ", a[i][j]);
	    if ((i * POLYBENCH_N + j) % 80 == 20) fprintf(stderr, "\n");
	  }
	  fprintf(stderr, "\n");
	}
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
// #pragma live-out a

for (i = 0; i < n; ++i)
  {
    x = a[i][i];
    for (j = 0; j <= i - 1; ++j)
      x = x - a[i][j] * a[i][j];
    p[i] = 1.0 / sqrt (x);
    for (j = i + 1; j < n; ++j)
      {
	x = a[i][j];
	for (k = 0; k <= i - 1; ++k)
	  x = x - a[j][k] * a[i][k];
	a[j][i] = x * p[i];
      }
  }

#pragma endscop

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    print_array(argc, argv);

    return 0;
}
