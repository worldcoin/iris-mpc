#define P 65519

extern "C" __global__ void matmul(int *c, unsigned short *output, unsigned int *a0Sums, unsigned int *a1Sums, int *b0Sums, int *b1Sums, size_t numRows, size_t numElements, size_t numCols, long long lCoeff, unsigned short *rngMasks0, unsigned short *rngMasks1)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
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
        unsigned short result = (((c00 + ((c01 + c10) << 8) + (c11 << 16))) * lCoeff) % P;

        output[idx] = ((unsigned int)P + (unsigned int)result) % (unsigned int)P;
    }
}

extern "C" __global__ void reconstructAndCompare(unsigned short *codes_result1, unsigned short *codes_result2, unsigned short *codes_result3, unsigned short *masks_result1, unsigned short *masks_result2, unsigned short *masks_result3, bool *output, double match_ratio, size_t numElements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        short nom = ((unsigned int)codes_result1[idx] + (unsigned int)codes_result2[idx] + (unsigned int)codes_result3[idx]) % (unsigned int)P;
        short den = ((unsigned int)masks_result1[idx] + (unsigned int)masks_result2[idx] + (unsigned int)masks_result3[idx]) % (unsigned int)P;
        output[idx] = nom > (1 - 2 * match_ratio) * den;
    }
}

extern "C" __global__ void reconstructDebug(unsigned short *codes_result1, unsigned short *codes_result2, unsigned short *codes_result3, unsigned short *masks_result1, unsigned short *masks_result2, unsigned short *masks_result3, double *output1, unsigned short *output2, unsigned short *output3, size_t numElements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned short nom = ((unsigned int)codes_result1[idx] + (unsigned int)codes_result2[idx] + (unsigned int)codes_result3[idx]) % (unsigned int)P;
        unsigned short den = ((unsigned int)masks_result1[idx] + (unsigned int)masks_result2[idx] + (unsigned int)masks_result3[idx]) % (unsigned int)P;
        output2[idx] = nom;
        nom = ((unsigned int)nom + 32759) % (unsigned int)P;
        output1[idx] = 0.5 - (double)nom / (2.0 * (double)den) + (32759.0/(2 * (double)den));
        output3[idx] = den;
    }
}