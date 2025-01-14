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

extern "C" __global__ void openResultsBatch(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned long long *output, size_t chunkLength, size_t queryLength, size_t offset, size_t numElements, size_t realChunkLen, size_t totalDbLen)
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

extern "C" __global__ void openResults(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned long long *output, size_t chunkLength, size_t queryLength, size_t offset, size_t numElements, size_t realChunkLen, size_t totalDbLen, unsigned short *match_distances_buffer_codes_a, unsigned short *match_distances_buffer_codes_b, unsigned short *match_distances_buffer_masks_a, unsigned short *match_distances_buffer_masks_b, unsigned int *match_distances_counter, unsigned short *code_dots_a, unsigned short *code_dots_b, unsigned short *mask_dots_a, unsigned short *mask_dots_b)
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

            // Save the corresponding code and mask dots for later (match distributions)
            unsigned int match_distances_counter_idx = atomicAdd(&match_distances_counter[0], 1);
            if (match_distances_counter_idx < 1000)
            {
                match_distances_buffer_codes_a[match_distances_counter_idx] = code_dots_a[idx * 64 + i];
                match_distances_buffer_codes_b[match_distances_counter_idx] = code_dots_b[idx * 64 + i];
                match_distances_buffer_masks_a[match_distances_counter_idx] = mask_dots_a[idx * 64 + i];
                match_distances_buffer_masks_b[match_distances_counter_idx] = mask_dots_b[idx * 64 + i];
            }

            // Mark which results are matches with a bit in the output
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
