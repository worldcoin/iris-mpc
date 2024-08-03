#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

// Basic CPU impl
void simpleCpuGemm(int m, int n, int k, const int8_t *A, const int8_t *B, int32_t *C)
{
    for (int row = 0; row < m; ++row)
    {
        for (int col = 0; col < n; ++col)
        {
            int32_t sum = 0;
            for (int i = 0; i < k; ++i)
            {
                sum += A[i + row * k] * B[i + col * k];
            }
            C[row + col * m] = sum;
        }
    }
}

int main()
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    int m = 99, n = 32, k = 12800;
    std::srand(std::time(0));

    std::vector<int8_t> h_A(m * k);
    std::vector<int8_t> h_B(k * n);
    std::vector<int8_t> h_B_transposed(n * k);
    std::vector<int32_t> h_C(m * n);
    std::vector<int32_t> h_C_cpu(m * n);
    std::vector<int32_t> h_C_cpu_transposed(m * n);

    for (int i = 0; i < m * k; ++i)
    {
        h_A[i] = static_cast<int8_t>(std::rand() % 127);
    }
    for (int i = 0; i < k * n; ++i)
    {
        h_B[i] = static_cast<int8_t>(std::rand() % 127);
    }

    int8_t *A, *B;
    int32_t *C;
    cudaMalloc(&A, m * k * sizeof(int8_t));
    cudaMalloc(&B, k * n * sizeof(int8_t));
    cudaMalloc(&C, m * n * sizeof(int32_t));

    cudaMemcpy(A, h_A.data(), m * k * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(B, h_B.data(), k * n * sizeof(int8_t), cudaMemcpyHostToDevice);

    const int32_t alpha = 1;
    const int32_t beta = 0;

    // Matmul using cublasGemmEx
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                 m, n, k,
                 &alpha,
                 A, CUDA_R_8I, k,
                 B, CUDA_R_8I, k,
                 &beta,
                 C, CUDA_R_32I, m,
                 CUBLAS_COMPUTE_32I_PEDANTIC, CUBLAS_GEMM_DEFAULT);

    cudaMemcpy(h_C.data(), C, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // CPU sanity check
    simpleCpuGemm(m, n, k, h_A.data(), h_B.data(), h_C_cpu.data());

    // Compare the results 
    int diffs = 0;
    for (int i = 0; i < m * n; ++i)
    {
        if (h_C_cpu[i] != h_C[i])
        {
            std::cout << h_C_cpu[i] << " " << h_C[i] << " " << i << "\n";
            diffs++;
        }
    }
    std::cout << "The results " << (diffs == 0 ? "MATCH" : "DO NOT MATCH") << ": " << diffs << " out of " << n * m << " values differ\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cublasDestroy(handle);

    return 0;
}
