#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add(int *c) {
    *c *= 2;
    printf("Hello, World!\n");
}

int main(void) {
    int c = 2;
    int *dev_c;
    //cudaMalloc()
    cudaMalloc(&dev_c, sizeof(int));
    cudaMemcpy(dev_c, &c, sizeof(int), cudaMemcpyHostToDevice);
    //核函数执行
    add<<<1,1>>>(dev_c);
    //cudaMemcpy()
    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2 * 2 = %d\n", c);
    cudaFree(dev_c);

    return 0;
}
