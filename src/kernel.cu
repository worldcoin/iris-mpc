#define P 65519
#define UINT_MAX 0xffffffff
#define ROTATIONS 15
#define IRIS_CODE_LENGTH 12800

extern "C" __global__ void matmul(int *c, unsigned short *output, int *a0Sums, int *a1Sums, int *b0Sums, int *b1Sums, size_t numRows, size_t numElements, size_t numCols, long long lCoeff, unsigned short *rngMasks0, unsigned short *rngMasks1)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        // Correct the sums to unsigned
        int a0s = a0Sums[idx % numRows] + numCols * 128;
        int a1s = a1Sums[idx % numRows] + numCols * 128;
        int b0s = b0Sums[idx / numRows] + numCols * 128;
        int b1s = b1Sums[idx / numRows] + numCols * 128;

        // Correct the intermediate results to unsigned
        long long c00 = c[idx + numElements * 0] + ((a0s + b0s) << 7) - (numCols * 16384);
        long long c01 = c[idx + numElements * 1] + ((a0s + b1s) << 7) - (numCols * 16384);
        long long c10 = c[idx + numElements * 2] + ((a1s + b0s) << 7) - (numCols * 16384);
        long long c11 = c[idx + numElements * 3] + ((a1s + b1s) << 7) - (numCols * 16384);
        unsigned short result = (((c00 + ((c01 + c10) << 8) + (c11 << 16))) * lCoeff) % P;

        output[idx] = ((unsigned int)P + (unsigned int)result + (unsigned int)rngMasks0[idx] - (unsigned int)rngMasks1[idx]) % (unsigned int)P;
    }
}

extern "C" __global__ void reconstructAndCompare(unsigned short *codes_result1, unsigned short *codes_result2, unsigned short *codes_result3, unsigned short *masks_result1, unsigned short *masks_result2, unsigned short *masks_result3, unsigned int *output, double match_ratio, size_t dbLength, size_t queryLength)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dbLength * queryLength)
    {
        short nom = ((unsigned int)codes_result1[idx] + (unsigned int)codes_result2[idx] + (unsigned int)codes_result3[idx]) % (unsigned int)P;
        short den = ((unsigned int)masks_result1[idx] + (unsigned int)masks_result2[idx] + (unsigned int)masks_result3[idx]) % (unsigned int)P;
        if ((nom > (1.0 - 2.0 * match_ratio) * den) && (output[idx / dbLength] > idx % dbLength))
        {
            // return db element with smallest index
            output[idx / dbLength] = idx % dbLength;
        }
    }
}

extern "C" __global__ void dedupQuery(unsigned int* matchResultsSelf, unsigned int* matchResults, unsigned char** queries, unsigned char** queriesNew, unsigned char** queriesSum, unsigned char** queriesSumNew, unsigned int rowCounter, size_t queryLength)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < queryLength)
    {
        bool compSelf = false;
        if (idx == ROTATIONS || (idx - 1) % ROTATIONS == 0)
        {
            compSelf = true;
        }

        if (matchResults[idx] == UINT_MAX || (compSelf && matchResultsSelf[idx] == UINT_MAX))
        {
            int row = atomicAdd(rowCounter, 1) - 1;
            queriesSumNew[0][row] = queriesSum[0][row];
            queriesSumNew[1][row] = queriesSum[1][row];
            for (int i=0;i<IRIS_CODE_LENGTH;i++)
            {
                queriesNew[0][idx * IRIS_CODE_LENGTH + i] = queries[0][idx * IRIS_CODE_LENGTH + i];
                queriesNew[1][idx * IRIS_CODE_LENGTH + i] = queries[1][idx * IRIS_CODE_LENGTH + i];
            }
        }
    }
}
