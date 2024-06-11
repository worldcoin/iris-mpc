#define UINT_MAX 0xffffffff
#define ROTATIONS 15
#define IRIS_CODE_LENGTH 12800

// TODO remove and merge into gemm call, custom kernel is not needed with field
extern "C" __global__ void matmul(int *c, unsigned short *output, int *a0Sums, int *a1Sums, int *b0Sums, int *b1Sums, size_t numRows, size_t numElements, size_t numCols, unsigned short *rngMasks0, unsigned short *rngMasks1)
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
        unsigned int c00 = c[idx + numElements * 0] + ((a0s + b0s) << 7) - (numCols * 16384);
        unsigned int c01 = c[idx + numElements * 1] + ((a0s + b1s) << 7) - (numCols * 16384);
        unsigned int c10 = c[idx + numElements * 2] + ((a1s + b0s) << 7) - (numCols * 16384);
        unsigned int c11 = c[idx + numElements * 3] + ((a1s + b1s) << 7) - (numCols * 16384);
        unsigned short result = ((c00 + ((c01 + c10) << 8) + (c11 << 16)));

        output[idx] = (unsigned int)result;
    }
}

extern "C" __global__ void openResults(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned int *output, size_t dbLength, size_t queryLength)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dbLength * queryLength / 64)
    {
        unsigned long long result = result1[idx] ^ result2[idx] ^ result3[idx];
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;
            bool match = (result & (1ULL << i));
            if (match)
            {
                // return db element with smallest index
                atomicMin(&output[queryIdx], dbIdx);
            }
        }
    }
}

extern "C" __global__ void mergeResults(unsigned int *matchResultsSelf, unsigned int *matchResults, unsigned int *finalResults, size_t queryLength)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < queryLength)
    {
        bool match = false;

        // Check if there is a match in the db
        for (int r = 0; r <= ROTATIONS * 2; r++)
        {
            int oldIdx = idx * (2 * ROTATIONS + 1) + r;
            if (matchResults[oldIdx] != UINT_MAX)
            {
                finalResults[idx] = matchResults[oldIdx];
                match = true;
            }
        }

        if (match)
            return;

        // Check if there is a match in the query itelf
        // We only need to check a single query, since we don't want to rotate double
        int oldIdx = idx * (2 * ROTATIONS + 1) + ROTATIONS;
        if (matchResultsSelf[oldIdx] != UINT_MAX && oldIdx != matchResultsSelf[oldIdx])
        {
            finalResults[idx] = matchResultsSelf[oldIdx];
            return;
        }

        finalResults[idx] = UINT_MAX;
    }
}
