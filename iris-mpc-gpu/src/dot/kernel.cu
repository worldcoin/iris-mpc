#define UINT_MAX 0xffffffff
#define ROTATIONS 15
#define ALL_ROTATIONS (2 * ROTATIONS + 1)
#define IRIS_CODE_LENGTH 12800
#define U8 unsigned char

extern "C" __global__ void xor_assign_u8(U8 *lhs, U8 *rhs, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        lhs[i] ^= rhs[i];
    }
}

extern "C" __global__ void matmul_correct_and_reduce(int *c, unsigned short *output, int *a0Sums, int *a1Sums, int *b0Sums, int *b1Sums, size_t dbLength, size_t numElements, size_t offset, unsigned short *rngMasks0, unsigned short *rngMasks1)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned int queryIdx = idx / dbLength;
        unsigned int dbIdx = idx % dbLength;
        int s0 = a0Sums[offset + dbIdx] + b0Sums[queryIdx];
        int s1 = a1Sums[offset + dbIdx] + b1Sums[queryIdx];
        output[idx] = c[idx] + (s0 << 7) + ((s0 + s1) << 15) + rngMasks0[idx] - rngMasks1[idx];
    }
}

extern "C" __global__ void openResults(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned long long *output, size_t dbLength, size_t queryLength, size_t offset, size_t numElements, size_t realDbLen)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned long long result = result1[idx] ^ result2[idx] ^ result3[idx];
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;
            bool match = (result & (1ULL << i));

            // Check if we are out of bounds for the query or db
            if (queryIdx >= queryLength || dbIdx >= realDbLen || !match)
            {
                continue;
            }

            unsigned int outputIdx = (idx * 64 + i) + offset * queryLength / ROTATIONS;
            atomicOr(&output[outputIdx / 64], (1ULL << (outputIdx % 64)));
        }
    }
}

extern "C" __global__ void mergeDbResults(unsigned long long *matchResultsLeft, unsigned long long *matchResultsRight, unsigned int *finalResults, size_t dbLength, size_t numElements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;
            bool matchLeft = (matchResultsLeft[idx] & (1ULL << i));
            bool matchRight = (matchResultsRight[idx] & (1ULL << i));

            finalResults[queryIdx] = UINT_MAX;

            if (matchLeft && matchRight)
            {
                atomicMin(&finalResults[queryIdx], dbIdx);
            }
        }
    }
}

extern "C" __global__ void mergeBatchResults(unsigned long long *matchResultsSelfLeft, unsigned long long *matchResultsSelfRight, unsigned int *finalResults, size_t dbLength, size_t numElements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        // Join together the results from both eyes
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;

            if ((queryIdx - ROTATIONS - 1) % ALL_ROTATIONS != 0)
            {
                continue;
            }

            if (queryIdx == dbIdx)
            {
                continue;
            }

            bool matchLeft = (matchResultsSelfLeft[idx] & (1ULL << i));
            bool matchRight = (matchResultsSelfRight[idx] & (1ULL << i));

            if (matchLeft || matchRight)
            {
                atomicMin(&finalResults[queryIdx], dbIdx);
            }
        }
    }
}
