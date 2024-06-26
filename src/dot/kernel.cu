#define UINT_MAX 0xffffffff
#define ROTATIONS 15
#define IRIS_CODE_LENGTH 12800

extern "C" __global__ void matmul_correct_and_reduce(int *c, unsigned short *output, int *a0Sums, int *a1Sums, int *b0Sums, int *b1Sums, size_t numRows, size_t numElements, unsigned short *rngMasks0, unsigned short *rngMasks1)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        int s0 = a0Sums[idx % numRows] + b0Sums[idx / numRows];
        int s1 = a1Sums[idx % numRows] + b1Sums[idx / numRows];
        output[idx] = c[idx] + (s0 << 7) + ((s0 + s1) << 15) + rngMasks0[idx] - rngMasks1[idx];
    }
}

extern "C" __global__ void openResults(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned int *output, size_t dbLength, size_t queryLength)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (dbLength * queryLength + 63) / 64)
    {
        unsigned long long result = result1[idx] ^ result2[idx] ^ result3[idx];
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;
            bool match = (result & (1ULL << i));

            // Check if we are out of bounds for the query or db
            if (queryIdx >= queryLength || dbIdx >= dbLength)
                return;

            // return db element with smallest index
            if (match)
                atomicMin(&output[queryIdx], dbIdx);
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

        // Check if there is a match in the query itelf
        // We only need to check a single query, since we don't want to rotate double
        int oldIdx = idx * (2 * ROTATIONS + 1) + ROTATIONS;
        if (matchResultsSelf[oldIdx] != UINT_MAX && oldIdx != matchResultsSelf[oldIdx])
        {
            finalResults[idx] = UINT_MAX - 1; // Set to UINT_MAX - 1 to indicate that the match is in the query itself
            return;
        }

        if (!match)
            finalResults[idx] = UINT_MAX;
    }
}