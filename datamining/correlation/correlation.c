#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"

#if defined(__CUDACC__) && !defined(CUDA_DEVICE)
#define CUDA_DEVICE 0
#endif

/* Default problem size. */
#ifndef M
# define M 500
#endif
#ifndef POLYBENCH_N
# define POLYBENCH_N 500
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%0.2lf "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
DATA_TYPE float_n = 321414134.01;
DATA_TYPE eps = 0.005;
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE data[M + 1][POLYBENCH_N + 1];
DATA_TYPE symmat[M + 1][M + 1];
DATA_TYPE stddev[M + 1];
DATA_TYPE mean[M + 1];
#else
DATA_TYPE** data = (DATA_TYPE**)malloc((M + 1) * sizeof(DATA_TYPE*));
DATA_TYPE** symmat = (DATA_TYPE**)malloc((M + 1) * sizeof(DATA_TYPE*));
DATA_TYPE* stddev = (DATA_TYPE*)malloc((M + 1) * sizeof(DATA_TYPE));
DATA_TYPE* mean = (DATA_TYPE*)malloc((M + 1) * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i <= M; ++i)
    {
      data[i] = (DATA_TYPE*)malloc((POLYBENCH_N + 1) * sizeof(DATA_TYPE));
      symmat[i] = (DATA_TYPE*)malloc((M + 1) * sizeof(DATA_TYPE));
    }
}
#endif

static inline
void init_array()
{
  int i, j;

  for (i = 0; i <= M; i++)
    for (j = 0; j <= POLYBENCH_N; j++)
      data[i][j] = ((DATA_TYPE) i*j) / (M+1);
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
      for (i = 0; i <= M; i++)
	for (j = 0; j <= M; j++) {
	  fprintf(stderr, DATA_PRINTF_MODIFIER, symmat[i][j]);
	  if ((i * M + j) % 80 == 20) fprintf(stderr, "\n");
	}
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int i, j, j1, j2;
  int m = M;
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

#define sqrt_of_array_cell(x,j) sqrt(x[j])

#pragma scop
// #pragma live-out symmat

  /* Determine mean of column vectors of input data matrix */
  for (j = 1; j <= m; j++)
    {
      mean[j] = 0.0;
      for (i = 1; i <= n; i++)
	mean[j] += data[i][j];
      mean[j] /= float_n;
    }

/* Determine standard deviations of column vectors of data matrix. */
  for (j = 1; j <= m; j++)
    {
      stddev[j] = 0.0;
      for (i = 1; i <= n; i++)
	stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
      stddev[j] /= float_n;
      stddev[j] = sqrt_of_array_cell(stddev, j);
      /* The following in an inelegant but usual way to handle
	 near-zero std. dev. values, which below would cause a zero-
	 divide. */
      stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }

 /* Center and reduce the column vectors. */
  for (i = 1; i <= n; i++)
    for (j = 1; j <= m; j++)
      {
	data[i][j] -= mean[j];
	data[i][j] /= sqrt(float_n) * stddev[j];
      }

/* Calculate the m * m correlation matrix. */
  for (j1 = 1; j1 <= m-1; j1++)
    {
      symmat[j1][j1] = 1.0;
      for (j2 = j1+1; j2 <= m; j2++)
	{
	  symmat[j1][j2] = 0.0;
	  for (i = 1; i <= n; i++)
	    symmat[j1][j2] += (data[i][j1] * data[i][j2]);
	  symmat[j2][j1] = symmat[j1][j2];
	}
    }
 symmat[m][m] = 1.0;

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
