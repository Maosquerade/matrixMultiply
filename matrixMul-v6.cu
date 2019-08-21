#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cublas_v2.h>

__device__ void d_rank8x8( float* C, const float* A, const float* B )
{
  float a[8], b;
  a[0] = A[0];
  a[1] = A[16];
  a[2] = A[32];
  a[3] = A[48];
  a[4] = A[64];
  a[5] = A[80];
  a[6] = A[96];
  a[7] = A[112];

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    b = B[i * 16];
    C[i * 8 + 0] += a[0] * b;
    C[i * 8 + 1] += a[1] * b;
    C[i * 8 + 2] += a[2] * b;
    C[i * 8 + 3] += a[3] * b;
    C[i * 8 + 4] += a[4] * b;
    C[i * 8 + 5] += a[5] * b;
    C[i * 8 + 6] += a[6] * b; 
    C[i * 8 + 7] += a[7] * b;
  }
}

#define mRank_8x8(i0, i1, i2, i3) { \
  c[0] += a[i0].x * b[i0].x; \
  c[1] += a[i0].y * b[i0].x; \
  c[2] += a[i1].x * b[i0].x; \
  c[3] += a[i1].y * b[i0].x; \
  c[4] += a[i2].x * b[i0].x; \
  c[5] += a[i2].y * b[i0].x; \
  c[6] += a[i3].x * b[i0].x; \
  c[7] += a[i3].y * b[i0].x; \
  c[8] += a[i0].x * b[i0].y; \
  c[9] += a[i0].y * b[i0].y; \
  c[10] += a[i1].x * b[i0].y; \
  c[11] += a[i1].y * b[i0].y; \
  c[12] += a[i2].x * b[i0].y; \
  c[13] += a[i2].y * b[i0].y; \
  c[14] += a[i3].x * b[i0].y; \
  c[15] += a[i3].y * b[i0].y; \
  c[16] += a[i0].x * b[i1].x; \
  c[17] += a[i0].y * b[i1].x; \
  c[18] += a[i1].x * b[i1].x; \
  c[19] += a[i1].y * b[i1].x; \
  c[20] += a[i2].x * b[i1].x; \
  c[21] += a[i2].y * b[i1].x; \
  c[22] += a[i3].x * b[i1].x; \
  c[23] += a[i3].y * b[i1].x; \
  c[24] += a[i0].x * b[i1].y; \
  c[25] += a[i0].y * b[i1].y; \
  c[26] += a[i1].x * b[i1].y; \
  c[27] += a[i1].y * b[i1].y; \
  c[28] += a[i2].x * b[i1].y; \
  c[29] += a[i2].y * b[i1].y; \
  c[30] += a[i3].x * b[i1].y; \
  c[31] += a[i3].y * b[i1].y; \
  c[32] += a[i0].x * b[i2].x; \
  c[33] += a[i0].y * b[i2].x; \
  c[34] += a[i1].x * b[i2].x; \
  c[35] += a[i1].y * b[i2].x; \
  c[36] += a[i2].x * b[i2].x; \
  c[37] += a[i2].y * b[i2].x; \
  c[38] += a[i3].x * b[i2].x; \
  c[39] += a[i3].y * b[i2].x; \
  c[40] += a[i0].x * b[i2].y; \
  c[41] += a[i0].y * b[i2].y; \
  c[42] += a[i1].x * b[i2].y; \
  c[43] += a[i1].y * b[i2].y; \
  c[44] += a[i2].x * b[i2].y; \
  c[45] += a[i2].y * b[i2].y; \
  c[46] += a[i3].x * b[i2].y; \
  c[47] += a[i3].y * b[i2].y; \
  c[48] += a[i0].x * b[i3].x; \
  c[49] += a[i0].y * b[i3].x; \
  c[50] += a[i1].x * b[i3].x; \
  c[51] += a[i1].y * b[i3].x; \
  c[52] += a[i2].x * b[i3].x; \
  c[53] += a[i2].y * b[i3].x; \
  c[54] += a[i3].x * b[i3].x; \
  c[55] += a[i3].y * b[i3].x; \
  c[56] += a[i0].x * b[i3].y; \
  c[57] += a[i0].y * b[i3].y; \
  c[58] += a[i1].x * b[i3].y; \
  c[59] += a[i1].y * b[i3].y; \
  c[60] += a[i2].x * b[i3].y; \
  c[61] += a[i2].y * b[i3].y; \
  c[62] += a[i3].x * b[i3].y; \
  c[63] += a[i3].y * b[i3].y; \
}

#define mFetchSmem(i0, i1, i2, i3, k) { \
  a[i0] = s_a[k * 64 + 0 + u]; \
  a[i1] = s_a[k * 64 + 16 + u]; \
  a[i2] = s_a[k * 64 + 32 + u]; \
  a[i3] = s_a[k * 64 + 48 + u]; \
  b[i0] = s_b[k * 64 + 0 + v]; \
  b[i1] = s_b[k * 64 + 16 + v]; \
  b[i2] = s_b[k * 64 + 32 + v]; \
  b[i3] = s_b[k * 64 + 48 + v]; \
}

__global__ void matrixMul( float* d_A, float* __restrict__ d_B, float* d_C, int n ) {
  __shared__ float2 s_a[512];
  __shared__ float2 s_b[512];
  float c[64] = {0.f};
  float2 a[4];
  float2 b[4];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int x = threadIdx.x;
  int tidx = x % 32;
  int tidy = x / 32;
  int u = x / 16;
  int v = x % 16;
  
  d_A += (by * 128 + tidx) + tidy * n;
  d_B += tidy * n + bx * 128 + tidx;
  d_C += (by * 128 + u * 2) * n + bx * 128 + v * 2;
  // 这里 A 是转置过的，因此也取 A 的行的数据入 shared memory
  
  for (int i = 0; i < n; i += 8) {
    ((float*)s_a)[tidy * 128 + tidx] = d_A[0];
    ((float*)s_a)[tidy * 128 + tidx + 32] = d_A[32];
    ((float*)s_a)[tidy * 128 + tidx + 64] = d_A[64];
    ((float*)s_a)[tidy * 128 + tidx + 96] = d_A[96];
 
    ((float*)s_b)[tidy * 128 + tidx] = d_B[0];
    ((float*)s_b)[tidy * 128 + tidx + 32] = d_B[32];
    ((float*)s_b)[tidy * 128 + tidx + 64] = d_B[64];
    ((float*)s_b)[tidy * 128 + tidx + 96] = d_B[96];

    __syncthreads();
  
  //值得注意的是这里取 shared memory 和计算并不是一一对应的，这也是 shared memory 很有意思的地方，这样可以分别进行存储的优化
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      mFetchSmem(0, 1, 2, 3, k);
      mRank_8x8(0, 1, 2, 3);
    }

    __syncthreads();
    d_A += 8 * n;
    d_B += 8 * n;
  }

  #pragma unroll
  for (int j = 0; j < 8; j += 2) {
    d_C[0] = c[j]; 
    d_C[1] = c[j + 8];
    d_C[32] = c[j + 16];
    d_C[33] = c[j + 24];
    d_C[64] = c[j + 32];
    d_C[65] = c[j + 40];
    d_C[96] = c[j + 48];
    d_C[97] = c[j + 56];
    d_C += n;
    d_C[0] = c[j + 1];
    d_C[1] = c[j + 9];
    d_C[32] = c[j + 17];
    d_C[33] = c[j + 25];
    d_C[64] = c[j + 33];
    d_C[65] = c[j + 41];
    d_C[96] = c[j + 49];
    d_C[97] = c[j + 57];
    d_C += 31 * n;
  }

}

__host__ float matrixCPU(float *A, float *B, float *C, int N) {
  float sum;
  float wucha = 0.0;
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
          sum = 0.0;
          for (int k = 0; k < N; k++) {
              sum += A[i * N + k] * B[k * N + j];
          }
          wucha += (sum - C[i * N + j]) * (sum - C[i * N + j]);
          if (i == 0 && j == 0) printf("%f %f\n", sum, C[0]);
      }
  }
  return wucha;
}

__host__ void matrixT(float *A, int N) {
  float temp;
  for (int i = 0; i < N; i++) {
      for (int j = 0; j < i; j++) {
          temp = A[i * N + j];
          A[i * N + j] = A[j * N + i];
          A[j * N + i] = temp;
      }
  }
}

int main() {
  float *matrixA, *matrixB, *matrixC;
  float *dev_A, *dev_B, *dev_C;
  int N = 1024;
  int i, j;
  float milliseconds;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  matrixA = (float*)malloc(N*N*sizeof(float));
  matrixB = (float*)malloc(N*N*sizeof(float));
  matrixC = (float*)malloc(N*N*sizeof(float));

  cudaMalloc(&dev_A, N * N * sizeof(float));
  cudaMalloc(&dev_B, N * N * sizeof(float));
  cudaMalloc(&dev_C, N * N * sizeof(float));

  for (i = 0; i < N; i++) {
      for (j = 0; j < N; j++) {
          matrixA[i * N + j] = j % 3 ? 1.0 : 0.5;
          matrixB[i * N + j] = j % 3 ? 1.0 : 0.5;
          // matrixA[i * N + j] = 1.0;
          // matrixB[i * N + j] = 1.0;
      }
  }

  matrixT(matrixA, N);
  cudaMemcpy(dev_A, matrixA, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, matrixB, N * N * sizeof(float), cudaMemcpyHostToDevice);
  
  dim3 blocks(N / 128, N / 128);
  dim3 threads(256);

  cudaEventRecord(start);
  matrixMul<<<blocks, threads>>>(dev_A, dev_B, dev_C, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(matrixC, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("The time used is %fms\n", milliseconds);

  // 用 cublas 计算结果 //////////////////////////////////////////////////
  cublasSetVector(N * N, sizeof(float), matrixA, 1, dev_A, 1);
  cublasSetVector(N * N, sizeof(float), matrixB, 1, dev_B, 1);
  cudaThreadSynchronize();

  cublasHandle_t handle;
  cublasCreate(&handle);

  float a = 1, b = 0;
  cudaEventRecord(start);
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, N, N, &a, dev_A, N, dev_B, N, &b, dev_C, N);
  cudaThreadSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("The time used is %fms\n", milliseconds);
  /////////////////////////////////////////////////////////////////////
  matrixT(matrixA, N);
  printf("wucha is %f\n", matrixCPU(matrixA, matrixB, matrixC, N)); // 计算误差
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}