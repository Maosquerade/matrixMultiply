#include <stdio.h>
#include <cuda_runtime.h>
__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}
int main( void ) {
    int c;
    int *dev_c;
    //cudaMalloc()
    cudaMalloc( (void**)&dev_c, sizeof(int) );
    //核函数执行
    add<<<1,1>>>( 2, 7, dev_c );   
    //cudaMemcpy()
    cudaMemcpy( &c, dev_c, sizeof(int),cudaMemcpyDeviceToHost ) ;
    printf( "2 + 7 = %d\n", c );
    //cudaFree()
    cudaFree( dev_c );
 
    return 0;
}