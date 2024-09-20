#include "2mm_kernel.hu"
__global__ void kernel0(float (*A)[2048], float (*B)[2048], float (*C)[2048], int ni, int nl, int nj, int nk)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    float private_C[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < ni; c0 += 8192) {
      for (int c1 = 32 * b1; c1 < nj; c1 += 8192) {
        for (int c2 = 0; c2 < nk; c2 += 32) {
          if (ni >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2047 && c0 == 32 * b0 && c2 <= 2016) {
            for (int c4 = t1; c4 <= 31; c4 += 16) {
              // shared_A[t0][c4] = A[(32 * b0 + t0) * 2048 + (c2 + c4)];
              shared_A[t0][c4] = A[32 * b0 + t0][c2 + c4];
            }
          }
          if (b1 <= 63 && c1 == 32 * b1 && nk >= t0 + c2 + 1 && t0 + c2 <= 2047) {
            for (int c4 = t1; c4 <= 31; c4 += 16) {
              // shared_B[t0][c4] = B[(t0 + c2) * 2048 + (32 * b1 + c4)];
              shared_B[t0][c4] = B[t0 + c2][32 * b1 + c4];
            }
          }
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1 && c2 == 0) {
            private_C[0][0] = 0;
            if (nj >= t1 + c1 + 17) {
              private_C[0][1] = 0;
            }
          }
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
            for (int c3 = 0; c3 <= ppcg_min(31, nk - c2 - 1); c3 += 1) {
              private_C[0][0] += (shared_A[t0][c3] * shared_B[c3][t1]);
              if (nj >= t1 + c1 + 17) {
                private_C[0][1] += (shared_A[t0][c3] * shared_B[c3][t1 + 16]);
              }
            }
          }
          __syncthreads();
        }
        if (nk <= 0) {
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nj >= t1 + c1 + 1) {
            private_C[0][0] = 0;
            if (nj >= t1 + c1 + 17) {
              private_C[0][1] = 0;
            }
          }
          __syncthreads();
        }
        if (b1 <= 63 && ni >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2047 && nj >= 32 * b1 + t1 + 1 && c0 == 32 * b0 && c1 == 32 * b1) {
          // C[(32 * b0 + t0) * 2048 + (32 * b1 + t1)] = private_C[0][0];
          C[32 * b0 + t0][32 * b1 + t1] = private_C[0][0];
          if (nj >= 32 * b1 + t1 + 17) {
            // C[(32 * b0 + t0) * 2048 + (32 * b1 + t1 + 16)] = private_C[0][1];
            C[32 * b0 + t0][32 * b1 + t1 + 16] = private_C[0][1];
          }
        }
        __syncthreads();
      }
    }
}
__global__ void kernel1(float (*C)[2048], float (*D)[2048], float (*E)[2048], int ni, int nl, int nj, int nk)
{
    int b0 = blockIdx.y, b1 = blockIdx.x;
    int t0 = threadIdx.y, t1 = threadIdx.x;
    __shared__ float shared_C[32][32];
    __shared__ float shared_D[32][32];
    float private_E[1][2];

    #define ppcg_min(x,y)    ({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); _x < _y ? _x : _y; })
    for (int c0 = 32 * b0; c0 < ni; c0 += 8192) {
      for (int c1 = 32 * b1; c1 < nl; c1 += 8192) {
        for (int c2 = 0; c2 < nj; c2 += 32) {
          if (ni >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2047 && c0 == 32 * b0 && c2 <= 2016) {
            for (int c4 = t1; c4 <= 31; c4 += 16) {
              // shared_C[t0][c4] = C[(32 * b0 + t0) * 2048 + (c2 + c4)];
              shared_C[t0][c4] = C[32 * b0 + t0][c2 + c4];
            }
          }
          if (b1 <= 63 && c1 == 32 * b1 && nj >= t0 + c2 + 1 && t0 + c2 <= 2047) {
            for (int c4 = t1; c4 <= 31; c4 += 16) {
              // shared_D[t0][c4] = D[(t0 + c2) * 2048 + (32 * b1 + c4)];
              shared_D[t0][c4] = D[t0 + c2][32 * b1 + c4];
            }
          }
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1 && c2 == 0) {
            private_E[0][0] = 0;
            if (nl >= t1 + c1 + 17) {
              private_E[0][1] = 0;
            }
          }
          if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
            for (int c3 = 0; c3 <= ppcg_min(31, nj - c2 - 1); c3 += 1) {
              private_E[0][0] += (shared_C[t0][c3] * shared_D[c3][t1]);
              if (nl >= t1 + c1 + 17) {
                private_E[0][1] += (shared_C[t0][c3] * shared_D[c3][t1 + 16]);
              }
            }
          }
          __syncthreads();
        }
        if (nj <= 0) {
          __syncthreads();
          if (ni >= t0 + c0 + 1 && nl >= t1 + c1 + 1) {
            private_E[0][0] = 0;
            if (nl >= t1 + c1 + 17) {
              private_E[0][1] = 0;
            }
          }
          __syncthreads();
        }
        if (b1 <= 63 && ni >= 32 * b0 + t0 + 1 && 32 * b0 + t0 <= 2047 && nl >= 32 * b1 + t1 + 1 && c0 == 32 * b0 && c1 == 32 * b1) {
          // E[(32 * b0 + t0) * 2048 + (32 * b1 + t1)] = private_E[0][0];
          E[32 * b0 + t0][32 * b1 + t1] = private_E[0][0];
          if (nl >= 32 * b1 + t1 + 17) {
            // E[(32 * b0 + t0) * 2048 + (32 * b1 + t1 + 16)] = private_E[0][1];
            E[32 * b0 + t0][32 * b1 + t1 + 16] = private_E[0][1];
          }
        }
        __syncthreads();
      }
    }
}
