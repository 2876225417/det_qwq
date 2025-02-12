#include <cuda_runtime.h>
#include <iostream>
// 测试cuda是否可用
// 高负载的核函数：无尽循环
__global__ void highLoadKernel(int* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        // 产生高负载：一个长时间执行的循环
        long long sum = 0;
        for (long long i = 0; i < 1000000000; ++i) {
            sum += (i * i) % 3;  // 执行一些冗长的操作
        }

        result[idx] = sum;
    }
}

// 高负载函数调用
extern "C" void highLoad(int* result, int N) {

    int* dev_result;
    size_t size = N * sizeof(int);

    // 分配 GPU 内存
    cudaMalloc((void**)&dev_result, size);

    // 将数据从主机复制到设备
    cudaMemcpy(dev_result, result, size, cudaMemcpyHostToDevice);

    // 启动核函数，多个线程并行执行
    int block_size = 128;
    int num_blocks = (N + block_size - 1) / block_size;

    // 调用核函数
    highLoadKernel<<<num_blocks, block_size>>>(dev_result, N);

    // 等待 GPU 完成计算
    cudaDeviceSynchronize();

    // 将计算结果从设备复制回主机
    cudaMemcpy(result, dev_result, size, cudaMemcpyDeviceToHost);

    // 清理设备内存
    cudaFree(dev_result);
}
