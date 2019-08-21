#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define GPUClockRate 823500

template<unsigned int numThreads>
__global__ void reduction(int *answer, const int *in, const int N) {
    extern __shared__ int sPartials[];
    
    const int tid = threadIdx.x;
    int sum = 0;

    // 将该线程应该计算的单元求和
    for (int i = tid + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
        sum += in[i];
    }

    sPartials[tid] = sum;
    __syncthreads();

    // 这里通过展开循环优化性能，减少需要判断的分支次数，并用 __syncthreads() 同步
    if (numThreads >= 1024) { 
        if (tid < 512) { sPartials[tid] += sPartials[tid + 512]; } 
        __syncthreads();
    }
    if (numThreads >= 512) { 
        if (tid < 256) { sPartials[tid] += sPartials[tid + 256]; } 
        __syncthreads();
    }
    if (numThreads >= 256) { 
        if (tid < 128) { sPartials[tid] += sPartials[tid + 128]; } 
        __syncthreads();
    }
    if (numThreads >= 128) { 
        if (tid < 64) { sPartials[tid] += sPartials[tid + 64]; } 
        __syncthreads();
    }

    // 由于一个线程束 warp 支持 32 线程，而在一个线程束内部采用 SIMD 结构，所有线程自动同步执行，而不需要通过 __syncthreads() 同步，可以节省大量时间
    if (tid < 32) {
        volatile int *Partials = sPartials;

        Partials[tid] += Partials[tid + 32];
        Partials[tid] += Partials[tid + 16];
        Partials[tid] += Partials[tid + 8];
        Partials[tid] += Partials[tid + 4];
        Partials[tid] += Partials[tid + 2];
        Partials[tid] += Partials[tid + 1];
        
        if (tid == 0) {
            answer[blockIdx.x] = Partials[0];
        } // 将数据结果写入 answer
    }
}

int main(){
    int *dev_number, *number;
    int *dev_answer, *dev_answer1, answer;
    int N = 536870912, sharedMemSize;
    int dimGrids, dimBlocks;
    cudaEvent_t start, stop; // 用于检测程序运行时间

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dimBlocks = 1024;
    dimGrids = N / 1024 <= 65536 ? N / 1024 : 65536;
    if ( dimGrids == 0 ) dimGrids = 1;

    // 分配数据内存并给数据赋值
    number = (int *)malloc(sizeof(int)*N);
    for (int i = 0; i < N; i += 1) {
        number[i] = i % 4;
    }

    // 给 device 变量分配内存
    cudaMalloc(&dev_number, N*sizeof(int));
    cudaMalloc(&dev_answer, sizeof(int));
    cudaMalloc(&dev_answer, sizeof(int)*dimGrids);
    cudaMalloc(&dev_answer1, sizeof(int));
    sharedMemSize = dimBlocks * sizeof(int);
    
    // 将 host 数据复制到 device 上
    cudaMemcpy(dev_number, number, N*sizeof(int), cudaMemcpyHostToDevice);
    
    float milliseconds, avr_time = 0.0;
    for (int j = 0; j < 100; j++) {
        cudaEventRecord(start);
        reduction<1024><<<dimGrids, dimBlocks, sharedMemSize>>>(dev_answer, dev_number, N);
        reduction<1024><<<1, dimBlocks, sharedMemSize>>>(dev_answer1, dev_answer, dimGrids);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        avr_time += milliseconds;
    }

    // 把结果传回 host 上
    cudaMemcpy(&answer, dev_answer1, sizeof(int), cudaMemcpyDeviceToHost);
    printf("the sum is %d\n", answer);

    cudaEventSynchronize(stop);

    printf("The time used is %fms\n", avr_time / 100);
    cudaFree(dev_number);
    cudaFree(dev_answer);
    free(number);
    return 0;
}