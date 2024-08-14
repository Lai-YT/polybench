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
# define M 1920
#endif
#ifndef POLYBENCH_N
# define POLYBENCH_N 1080
#endif
#ifndef T
# define T 1920
#endif

/* Default data type is int. */
#ifndef DATA_TYPE
# define DATA_TYPE int
#endif
#ifndef DATA_PRINTF_MODIFIER
# define DATA_PRINTF_MODIFIER "%d "
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE tot[4];
DATA_TYPE Gauss[4];
DATA_TYPE g_tmp_image[POLYBENCH_N][M];
DATA_TYPE g_acc1[POLYBENCH_N][M][4];
DATA_TYPE g_acc2[POLYBENCH_N][M][4];
DATA_TYPE in_image[POLYBENCH_N][M]; //input
DATA_TYPE gauss_image[POLYBENCH_N][M]; //output
#else
DATA_TYPE** g_tmp_image = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE** in_image = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE** gauss_image = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE*** g_acc1 = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE**));
DATA_TYPE*** g_acc2 = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE**));
{
  int i, j;
  for (i = 0; i < POLYBENCH_N; ++i)
    {
      g_tmp_image[i] = (DATA_TYPE*)malloc(M * sizeof(DATA_TYPE));
      in_image[i] = (DATA_TYPE*)malloc(M * sizeof(DATA_TYPE));
      gauss_image[i] = (DATA_TYPE*)malloc(M * sizeof(DATA_TYPE));
      g_acc1[i] = (DATA_TYPE**)malloc(M * sizeof(DATA_TYPE*));
      g_acc2[i] = (DATA_TYPE**)malloc(M * sizeof(DATA_TYPE*));
      for (j = 0; j < M; ++j)
	{
	  g_acc1[i][j] = (DATA_TYPE*)malloc(4 * sizeof(DATA_TYPE));
	  g_acc2[i][j] = (DATA_TYPE*)malloc(4 * sizeof(DATA_TYPE));
	}
    }
}
#endif

static inline
void init_array()
{
  int i, j;

  for (i = 0; i < POLYBENCH_N; i++)
    for (j = 0; j < M; j++)
      in_image[i][j] = ((DATA_TYPE) i*j) / M;
  for (i = 0; i < 4; i++)
      Gauss[i] = i;
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
	for (j = 0; j < M; j++) {
	  fprintf(stderr, DATA_PRINTF_MODIFIER, gauss_image[i][j]);
	  if ((i * POLYBENCH_N + j) % 80 == 20) fprintf(stderr, "\n");
	}
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int x, y, k;
  int t = T;
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

#pragma scop
// #pragma live-out gauss_image

    tot[0] = 0;
    for (k = t-1; k <= 1 + t; k++)
      tot[k + 2 - t] = tot[k + 1 - t] + Gauss[k - t + 1];
    for (k = t - 1; k <= 1 + t; k++)
      tot[k + 2 - t] = tot[k + 1 - t] + Gauss[k - t + 1];
    for (x = 1; x < n-2; x++)
      {
        for (y = 0; y < m; y++)
	  {
            g_acc1[x][y][0]=0;
            for (k = t - 1; k <= 1 + t; k++)
                g_acc1[x][y][k + 2 - t] = g_acc1[x][y][k + 1 - t] +
		  in_image[x + k - t][y] * Gauss[k - t + 1];
            g_tmp_image[x][y] = g_acc1[x][y][3] / tot[3];
	  }
      }
    for (x = 1; x < n-1; x++)
      {
        for (y = 1; y < m-1; y++)
	  {
            g_acc2[x][y][0]=0;
            for (k = t-1; k <= 1 + t; k++)
                g_acc2[x][y][k + 2 - t] =
		  g_acc2[x][y][k + 1 - t] + g_tmp_image[x][y + k - t] *
		  Gauss[k - t + 1];
            gauss_image[x][y] = g_acc2[x][y][3] / tot[3];
	  }
      }

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
