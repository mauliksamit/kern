#include <cstdio>
__global__ void k(int* o) { *o = 42; }
int main() {
    int* d; int h = 0;
    cudaMalloc(&d, 4);
    k<<<1,1>>>(d);
    cudaMemcpy(&h, d, 4, cudaMemcpyDeviceToHost);
    printf("%d\n", h);
    cudaFree(d);
}