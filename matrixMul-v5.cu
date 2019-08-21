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

__global__ void matrixMul( float* d_A, float* d_B, float* d_C, int n ) {
  __shared__ float s_a[1024];
  __shared__ float s_b[1024];
  float c[64] = {0.f};
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int x = threadIdx.x;
  int tidx = x % 32;
  int tidy = x / 32;
  int u = x / 16;
  int v = x % 16;
  float *s_ap = &s_a[u];
  float *s_bp = &s_b[v];
  
  d_A += (by * 128 + tidx) + tidy * n;
  d_B += tidy * n + bx * 128 + tidx;
  d_C += (by * 128 + u) * n + bx * 128 + v;
  // 这里 A 是转置过的，因此也取 A 的行的数据入 shared memory
  
  for (int i = 0; i < n; i += 8) {
    s_a[tidy * 128 + tidx] = d_A[0];
    s_a[tidy * 128 + tidx + 32] = d_A[32];
    s_a[tidy * 128 + tidx + 64] = d_A[64];
    s_a[tidy * 128 + tidx + 96] = d_A[96];
 
    s_b[tidy * 128 + tidx] = d_B[0];
    s_b[tidy * 128 + tidx + 32] = d_B[32];
    s_b[tidy * 128 + tidx + 64] = d_B[64];
    s_b[tidy * 128 + tidx + 96] = d_B[96];

    __syncthreads();
  
  //值得注意的是这里取 shared memory 和计算并不是一一对应的，这也是 shared memory 很有意思的地方，这样可以分别进行存储的优化
    #pragma unroll
    for (int k = 0; k < 8; k++) {
      d_rank8x8(c, &s_ap[k * 128], &s_bp[k * 128]);
    }

    __syncthreads();
    d_A += 8 * n;
    d_B += 8 * n;
  }

  #pragma unroll
  for (int j = 0; j < 8; j += 1) {
    d_C[0] = c[j]; 
    d_C[16] = c[j + 8];
    d_C[32] = c[j + 16];
    d_C[48] = c[j + 24];
    d_C[64] = c[j + 32];
    d_C[80] = c[j + 40];
    d_C[96] = c[j + 48];
    d_C[112] = c[j + 56];
    d_C += 16 * n;
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
  int N = 4096;
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
  // printf("wucha is %f\n", matrixCPU(matrixA, matrixB, matrixC, N)); // 计算误差
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_C);
  free(matrixA);
  free(matrixB);
  free(matrixC);

  return 0;
}