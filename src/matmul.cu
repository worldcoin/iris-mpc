extern "C" __global__ void matmul_f16(int* c,  unsigned short* output, unsigned int* a0Sums, unsigned int* a1Sums, int* b0Sums, int* b1Sums, size_t numRows, size_t numElements, size_t numCols, long long p, long long lCoeff) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements) {
        unsigned int a0s = a0Sums[idx % numRows];
        unsigned int a1s = a1Sums[idx % numRows];

        // Correct the sum to unsigned
        int b0s = b0Sums[idx / numRows] + numCols * 128;
        int b1s = b1Sums[idx / numRows] + numCols * 128;

        // Correct the intermediate results to unsigned
        long long c00 = c[idx] + ((a0s + b0s) << 7) - (numCols * 16384);
        long long c01 = c[idx + numElements] + ((a0s + b1s) << 7) - (numCols * 16384);
        long long c10 = c[idx + numElements * 2] + ((a1s + b0s) << 7) - (numCols * 16384);
        long long c11 = c[idx + numElements * 3] + ((a1s + b1s) << 7) - (numCols * 16384);

        output[idx] = (((c00 + ((c01 + c10) << 8) + (c11 << 16))) * lCoeff) % p;
    }
}