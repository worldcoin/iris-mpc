#define UINT_MAX 0xffffffff
#define ROTATIONS 15
#define ALL_ROTATIONS (2 * ROTATIONS + 1)
#define U8 unsigned char
#define MAX_MATCHES_LEN 256

extern "C" __global__ void xor_assign_u8(U8 *lhs, U8 *rhs, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        lhs[i] ^= rhs[i];
    }
}

extern "C" __global__ void matmul_correct_and_reduce(int *c, unsigned short *output, int *a0Sums, int *a1Sums, int *b0Sums, int *b1Sums, size_t dbLength, size_t numElements, size_t offset, unsigned short multiplier, unsigned short *rngMasks0, unsigned short *rngMasks1)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned int queryIdx = idx / dbLength;
        unsigned int dbIdx = idx % dbLength;
        int s0 = a0Sums[offset + dbIdx] + b0Sums[queryIdx];
        int s1 = a1Sums[offset + dbIdx] + b1Sums[queryIdx];
        unsigned short result = c[idx] + (s0 << 7) + ((s0 + s1) << 15);
        output[idx] = result * multiplier + rngMasks0[idx] - rngMasks1[idx];
    }
}

extern "C" __global__ void openResults(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned long long *output, size_t chunkLength, size_t queryLength, size_t offset, size_t numElements, size_t realChunkLen, size_t totalDbLen)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned long long result = result1[idx] ^ result2[idx] ^ result3[idx];
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / chunkLength;
            unsigned int dbIdx = (idx * 64 + i) % chunkLength;
            bool match = (result & (1ULL << i));

            // Check if we are out of bounds for the query or db
            if (queryIdx >= queryLength || dbIdx >= realChunkLen || !match)
            {
                continue;
            }

            unsigned int outputIdx = totalDbLen * (queryIdx / ALL_ROTATIONS) + dbIdx + offset;
            atomicOr(&output[outputIdx / 64], (1ULL << (outputIdx % 64)));
        }
    }
}

extern "C" __global__ void mergeDbResults(unsigned long long *matchResultsLeft, unsigned long long *matchResultsRight, unsigned int *finalResults, size_t queryLength, size_t dbLength, size_t numElements, unsigned int *matchCounter, unsigned int *allMatches, unsigned int *matchCounterLeft, unsigned int *matchCounterRight, unsigned int *partialResultsLeft, unsigned int *partialResultsRight)
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

            // Check bounds
            if (queryIdx >= queryLength || dbIdx >= dbLength)
                continue;

            // Check for partial results (only used for debugging)
            if (matchLeft)
            {
                unsigned int queryMatchCounter = atomicAdd(&matchCounterLeft[queryIdx], 1);
                if (queryMatchCounter < MAX_MATCHES_LEN)
                    partialResultsLeft[MAX_MATCHES_LEN * queryIdx + queryMatchCounter] = dbIdx;
            }
            if (matchRight)
            {
                unsigned int queryMatchCounter = atomicAdd(&matchCounterRight[queryIdx], 1);
                if (queryMatchCounter < MAX_MATCHES_LEN)
                    partialResultsRight[MAX_MATCHES_LEN * queryIdx + queryMatchCounter] = dbIdx;
            }

            // Current *AND* policy: only match, if both eyes match
            if (matchLeft && matchRight)
            {
                atomicMin(&finalResults[queryIdx], dbIdx);
                unsigned int queryMatchCounter = atomicAdd(&matchCounter[queryIdx], 1);
                if (queryMatchCounter < MAX_MATCHES_LEN)
                    allMatches[MAX_MATCHES_LEN * queryIdx + queryMatchCounter] = dbIdx;
            }
        }
    }
}

extern "C" __global__ void mergeDbResultsWithOrPolicyBitmap(unsigned long long *matchResultsLeft, unsigned long long *matchResultsRight, unsigned int *finalResults, size_t queryLength, size_t dbLength, unsigned int *matchCounter, unsigned int *allMatches, unsigned int *matchCounterLeft, unsigned int *matchCounterRight, unsigned int *partialResultsLeft, unsigned int *partialResultsRight, const unsigned long long *orPolicyBitmap) // 2D bitmap stored as 1D
{

    size_t rowStride64 = (dbLength + 63) / 64;

    size_t totalBits   = dbLength * queryLength;
    size_t numElements = (totalBits + 63) / 64; //  div_ceil

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        for (int i = 0; i < 64; i++)
        {
        
            size_t globalBit = idx * 64 + i;
            // Protect against any leftover bits if totalBits not multiple of 64
            if (globalBit >= totalBits) break;
            
            unsigned int queryIdx = globalBit / dbLength;
            unsigned int dbIdx = globalBit % dbLength;
            bool matchLeft = (matchResultsLeft[idx] & (1ULL << i));
            bool matchRight = (matchResultsRight[idx] & (1ULL << i));

            // Check bounds
            if (queryIdx >= queryLength || dbIdx >= dbLength)
                continue;

            // Check for partial results (only used for debugging)
            if (matchLeft)
            {
                unsigned int qmcL = atomicAdd(&matchCounterLeft[queryIdx], 1);
                if (qmcL < MAX_MATCHES_LEN)
                    partialResultsLeft[MAX_MATCHES_LEN * queryIdx + qmcL] = dbIdx;
            }
            if (matchRight)
            {
                unsigned int qmcR = atomicAdd(&matchCounterRight[queryIdx], 1);
                if (qmcR < MAX_MATCHES_LEN)
                    partialResultsRight[MAX_MATCHES_LEN * queryIdx + qmcR] = dbIdx;
            }
            size_t rowIndex = queryIdx * rowStride64;
            bool useOr = (orPolicyBitmap[rowIndex + (dbIdx / 64)]
                          & (1ULL << (dbIdx % 64))) != 0ULL;
                        
            // If useOr is true => (matchLeft || matchRight),
            // else => (matchLeft && matchRight).
            bool finalMatch = useOr ? (matchLeft || matchRight)
                                    : (matchLeft && matchRight);
    
            if (finalMatch)
            {
                atomicMin(&finalResults[queryIdx], dbIdx);
                unsigned int qmc = atomicAdd(&matchCounter[queryIdx], 1);
                if (qmc < MAX_MATCHES_LEN)
                    allMatches[MAX_MATCHES_LEN * queryIdx + qmc] = dbIdx;
            }
        }
    }
}


extern "C" __global__ void mergeBatchResults(unsigned long long *matchResultsSelfLeft, unsigned long long *matchResultsSelfRight, unsigned int *finalResults, size_t queryLength, size_t dbLength, size_t numElements, unsigned int *matchCounter, unsigned int *allMatches, unsigned int *__matchCounterLeft, unsigned int *__matchCounterRight, unsigned int *__partialResultsLeft, unsigned int *__partialResultsRight)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;

            // Check bounds
            if (queryIdx >= queryLength || dbIdx >= dbLength)
                continue;

            // Query is already considering rotations, ignore rotated db entries
            if ((dbIdx < ROTATIONS) || ((dbIdx - ROTATIONS) % ALL_ROTATIONS != 0))
                continue;

            // Only consider results above the diagonal
            if (queryIdx <= dbIdx / ALL_ROTATIONS)
                continue;

            bool matchLeft = (matchResultsSelfLeft[idx] & (1ULL << i));
            bool matchRight = (matchResultsSelfRight[idx] & (1ULL << i));

            // Current *AND* policy: only match if both eyes match
            if (matchLeft && matchRight)
            {
                atomicMin(&finalResults[queryIdx], UINT_MAX - 1);
                unsigned int queryMatchCounter = atomicAdd(&matchCounter[queryIdx], 1);
                if (queryMatchCounter < MAX_MATCHES_LEN)
                    allMatches[MAX_MATCHES_LEN * queryIdx + queryMatchCounter] = UINT_MAX - dbIdx / ALL_ROTATIONS;
            }
        }
    }
}
