#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef POLYBENCH_N
# define POLYBENCH_N 4000
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
DATA_TYPE y[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE sum[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE beta[POLYBENCH_N];
DATA_TYPE alpha[POLYBENCH_N];
DATA_TYPE r[POLYBENCH_N]; //input
DATA_TYPE out[POLYBENCH_N]; //output
#else
DATA_TYPE** y = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE** sum = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE* beta = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* alpha = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* r = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* out = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < POLYBENCH_N; ++i)
    {
      y[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
      sum[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
    }
}
#endif

static inline
void init_array()
{
  int i;

  for (i = 0; i < POLYBENCH_N; i++)
    r[i] = i * M_PI;
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
	fprintf(stderr, DATA_PRINTF_MODIFIER, r[i]);
	if (i%80 == 20) fprintf(stderr, "\n");
      }
      fprintf(stderr, "\n");
    }
}


int main(int argc, char** argv)
{
  int i, k;
  int n = POLYBENCH_N;

  /* Initialize array. */
  init_array();

  /* Start timer. */
  polybench_start_instruments;

#pragma scop
// #pragma live-out out

  y[0][0] = r[0];
  beta[0] = 1;
  alpha[0] = r[0];
  for (k = 1; k < n; k++)
    {
      beta[k] = beta[k-1] - alpha[k-1] * alpha[k-1] * beta[k-1];
      sum[0][k] = r[k];
      for (i = 0; i <= k - 1; i++)
	sum[i+1][k] = sum[i][k] + r[k-(i)-1] * y[i][k-1];
      alpha[k] = -sum[k][k] * beta[k];
      for (i = 0; i <= k-1; i++)
	y[i][k] = y[i][k-1] + alpha[k] * y[k-(i)-1][k-1];
      y[k][k] = alpha[k];
    }
  for (i = 0; i < n; i++)
    out[i] = y[i][POLYBENCH_N-1];

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
