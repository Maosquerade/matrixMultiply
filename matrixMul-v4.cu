#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <cublas_v2.h>

#define blockSize 32

__device__ void update1(float *a, float b, float *c) {
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        c[i] += a[i * 4] * b;
    }
}

__global__ void matrixMul(float *a, float *b, float *c, int n) {
    __shared__ float sa[16][65]; // A shared memory array

    float cr[16] = {0.f}; // C register
    int nDiv64 = n / 64;
    int sRow = threadIdx.y;
    int sRow4 = sRow * 4;
    int sRow4Plus32 = sRow4 + 32;
    // int sRow4Plus64 = sRow4 + 64;
    // int sRow4Plus96 = sRow4 + 96;
    int sCol = threadIdx.x;
    // int sCol2 = sCol * 2;
    // int sCol2Plus1 = sCol2 + 1;
    // int sCol2Plus2 = sCol2 + 2;
    // int sCol2Plus3 = sCol2 + 3;
    int tid = sRow * 16 + sCol;
    int aNext = (16 * blockIdx.y + sRow) * n + sCol * 4; // 这里怪怪的, 一个 block 读 32*32 的 a 的子矩阵，因为每行读两个所以乘2
    int bNext = 128 * blockIdx.x + tid; // 对同一个 thread 只算同一列的结果 ，所以需要的 B 也是同一列的
    // int sRowPlus16 = sRow + 16;
    // int sRowPlus8 = sRow + 8;
    // int sRowPlus24 = sRow + 24;
    int nTimes2 = n * 2;
    int nTimes3 = n * 3;
    int nTimes4 = n * 4;
    int nTimes8 = n * 8;
    // int nTimes16 = n * 16;
    // int nTimes24 = n * 24;

    a += aNext;
    b += bNext;

    int i,j;
    // float4 *temp = (float4 *)a;

    // if (blockIdx.x == 7 && blockIdx.y == 31 && threadIdx.x == 0 && threadIdx.y == 0) printf("I'm here.\n");
    for (i = 0; i < nDiv64; i++) {
        // temp = *(float2 *)a;
        // sa[sCol2][sRow] = a[0];
        // sa[sCol2Plus1][sRow] = a[1];
        // sa[sCol2][sRowPlus8] = a[nTimes8];
        // sa[sCol2Plus1][sRowPlus8] = a[nTimes8 + 1];
        // temp = *(float2 *)(a + nTimes8);
        // sa[sCol2][sRowPlus16] = a[nTimes16];
        // sa[sCol2Plus1][sRowPlus16] = a[nTimes16 + 1];
        // sa[sCol2][sRowPlus24] = a[nTimes24];
        // sa[sCol2Plus1][sRowPlus24] = a[nTimes24 + 1];
        // *((float4 *)(&sa[sCol][sRow4])) = temp[0];
        // *((float4 *)(&sa[sCol][sRow4 + 32])) = temp[nTimes2];

        // *((float4 *)(&sa[sCol][sRow4])) = temp[0];
        // *((float4 *)(&sa[sCol][sRow4Plus32])) = temp[nTimes2];
        sa[sCol][sRow4] = a[0];
        sa[sCol][sRow4 + 1] = a[1];
        sa[sCol][sRow4 + 2] = a[2];
        sa[sCol][sRow4 + 3] = a[3];
        sa[sCol][sRow4Plus32] = a[nTimes8];
        sa[sCol][sRow4Plus32 + 1] = a[nTimes8 + 1];
        sa[sCol][sRow4Plus32 + 2] = a[nTimes8 + 2];
        sa[sCol][sRow4Plus32 + 3] = a[nTimes8 + 3];
        // sa[sCol][sRow4Plus64] = a[nTimes16];
        // sa[sCol][sRow4Plus64 + 1] = a[nTimes16 + 1];
        // sa[sCol][sRow4Plus64 + 2] = a[nTimes16 + 2];
        // sa[sCol][sRow4Plus64 + 3] = a[nTimes16 + 3];
        // sa[sCol][sRow4Plus96] = a[nTimes24];
        // sa[sCol][sRow4Plus96 + 1] = a[nTimes24 + 1];
        // sa[sCol][sRow4Plus96 + 2] = a[nTimes24 + 2];
        // sa[sCol][sRow4Plus96 + 3] = a[nTimes24 + 3];
        __syncthreads();
        float br0 = b[0];
        float br1 = b[n];
        float br2 = b[nTimes2];
        float br3 = b[nTimes3];

        b += nTimes4;
        #pragma unroll
        for (j = 0; j < 15; j++) {
            update1(&sa[j][0], br0, cr); br0 = b[0];
            update1(&sa[j][1], br1, cr); br1 = b[n];
            update1(&sa[j][2], br2, cr); br2 = b[nTimes2];
            update1(&sa[j][3], br3, cr); br3 = b[nTimes3];
            b += nTimes4;
        }
        update1(&sa[15][0], br0, cr); 
        update1(&sa[15][1], br1, cr); 
        update1(&sa[15][2], br2, cr); 
        update1(&sa[15][3], br3, cr); 
        // for (j = 0; j < 32; j++) {
        //     float br = b[0];
        //     b += n;
        //     update1(&sa[j][0], br, cr);
        // }
        a += 64;
        // a += 32;
        __syncthreads();
    }

    int cNext = 16 * blockIdx.y * n + 128 * blockIdx.x + tid;
    c += cNext;
    for (int k = 0; k < 16; k++) {
        c[0] = cr[k];
        c += n;
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

__host__ void matrixT(float *A, int N) {
    float temp;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
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
            matrixA[i * N + j] = j % 2 ? 1.0 : 0.5;
            matrixB[i * N + j] = j % 2 ? 0.5 : 1.0;
            // matrixA[i * N + j] = 1.0;
            // matrixB[i * N + j] = 1.0;
        }
    }

    cudaMemcpy(dev_A, matrixA, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, matrixB, N * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blocks(N / 128, N / 16);
    dim3 threads(16, 8);

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
    printf("wucha is %f\n", matrixCPU(matrixA, matrixB, matrixC, N)); // 计算误差
    cudaFree(dev_A);
    cudaFree(dev_B);
    cudaFree(dev_C);
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}