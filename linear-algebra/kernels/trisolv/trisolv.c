#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#include "instrument.h"


/* Default problem size. */
#ifndef POLYBENCH_N
# define POLYBENCH_N 4000
#endif

/* Default data type is double. */
#ifndef DATA_TYPE
# define DATA_TYPE double
#endif

/* Array declaration. Enable malloc if POLYBENCH_TEST_MALLOC. */
#ifndef POLYBENCH_TEST_MALLOC
DATA_TYPE A[POLYBENCH_N][POLYBENCH_N];
DATA_TYPE x[POLYBENCH_N];
DATA_TYPE c[POLYBENCH_N];
#else
DATA_TYPE** A = (DATA_TYPE**)malloc(POLYBENCH_N * sizeof(DATA_TYPE*));
DATA_TYPE* x = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
DATA_TYPE* c = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
{
  int i;
  for (i = 0; i < POLYBENCH_N; ++i)
    A[i] = (DATA_TYPE*)malloc(POLYBENCH_N * sizeof(DATA_TYPE));
}
#endif

static inline
void init_array()
{
  int i, j;

  for (i = 0; i < POLYBENCH_N; i++)
    {
      c[i] = ((DATA_TYPE) i) / POLYBENCH_N;
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
	fprintf(stderr, "%0.2lf ", x[i]);
	if ((2 * i) % 80 == 20) fprintf(stderr, "\n");
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

  /* Start timer. */
  polybench_start_instruments;

#pragma scop
// #pragma live-out x

  for (i = 0; i < n; i++)
    {
      x[i] = c[i];
      for (j = 0; j <= i - 1; j++)
        x[i] = x[i] - A[i][j] * x[j];
      x[i] = x[i] / A[i][i];
    }

#pragma endscop

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  print_array(argc, argv);

  return 0;
}
