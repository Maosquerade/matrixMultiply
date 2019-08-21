#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test(const int *in, int *answer) {
    const int tid = threadIdx.x;

    if (tid == 0) printf("hello!\n");
    int sum = in[tid];
    if (tid == 0) printf("sum[0] is %d\n", sum);
    *answer = sum;
}

int main() {
    int *h_data, *d_data;
    int N = 64;
    int *h_answer, *d_answer;

    h_data = (int *)malloc(N*sizeof(int));
    h_answer = (int *)malloc(sizeof(int));
    cudaMalloc(&d_data, N*sizeof(int));
    cudaMalloc(&d_answer, sizeof(int));

    for (int i = 0; i < N; i += 1) {
        h_data[i] = N - i;
    }
    cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);
    test<<<1, N>>>(h_data, d_answer);

    cudaMemcpy(h_answer, d_answer, sizeof(int), cudaMemcpyDeviceToHost);
    printf("The answer is %d\n", *h_answer);
}