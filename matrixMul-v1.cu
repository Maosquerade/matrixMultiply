#include <stdio.h>
#include <cuda_runtime.h>

#define blockSize 32

__global__ void matrixMul (float *A, float *B, float *C, int N) {
    float sum = 0.0;
    extern __shared__ float shared_A[];
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int column = threadIdx.x;
    int i, j;
    // 矩阵分块
    
    for (i = tid; i < N; i += blockDim.x) {
        shared_A[i] = A[row * N + i];
    }
    __syncthreads();

    for (i = tid; i < N; i += blockDim.x) {
        sum = 0.0;
        for (j = 0; j < N; j += 8) {
            sum += shared_A[j] * B[j * N + i];
            sum += shared_A[j + 1] * B[(j + 1)* N + i];
            sum += shared_A[j + 2] * B[(j + 2)* N + i];
            sum += shared_A[j + 3] * B[(j + 3)* N + i];
            sum += shared_A[j + 4] * B[(j + 4)* N + i];
            sum += shared_A[j + 5] * B[(j + 5)* N + i];
            sum += shared_A[j + 6] * B[(j + 6)* N + i];
            sum += shared_A[j + 7] * B[(j + 7)* N + i];
        }
        C[row * N + i] = sum;
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
        }
    }
    return wucha;
}

int main() {
    float *matrixA, *matrixB, *matrixC;
    float *dev_A, *dev_B, *dev_C;
    int N = 1024;
    int i, j;
    float milliseconds;
    int dimGrids, dimBlocks, sharedSize;
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
            matrixA[i * N + j] = 1.0;
            matrixB[i * N + j] = 1.0;
        }
    }

    if (N <= 1024) {
        dimBlocks = N;
        dimGrids = N;
    } else {
        dimGrids = N * N / 1024;
        dimBlocks = 1024;
    }
    sharedSize = N * sizeof(float); 

    cudaMemcpy(dev_A, matrixA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, matrixB, N * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blocks(N / blockSize, N / blockSize);
    dim3 threads(blockSize, blockSize);

    cudaEventRecord(start);
    matrixMul<<<dimGrids, dimBlocks, sharedSize>>>(dev_A, dev_B, dev_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(matrixC, dev_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("C[0,0] is %f\n", matrixC[2]);
    printf("The time used is %fms\n", milliseconds);
    // printf("wucha is %f\n", matrixCPU(matrixA, matrixB, matrixC, N));
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    free(matrixA);
    free(matrixB);
    free(matrixC);

}
