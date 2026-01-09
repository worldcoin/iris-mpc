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

extern "C" __global__ void openResults(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned long long *output, size_t chunkLength, size_t queryLength, size_t offset, size_t numElements, size_t realChunkLen, size_t totalDbLen, unsigned short *match_distances_buffer_codes_a, unsigned short *match_distances_buffer_codes_b, unsigned short *match_distances_buffer_masks_a, unsigned short *match_distances_buffer_masks_b, unsigned int *match_distances_counter, unsigned long long *match_distances_indices, unsigned int *partialResultsCounter, unsigned int *partialResultsQueryIndices, unsigned int *partialResultsDbIndices, signed char *partialResultsRotations, unsigned short *code_dots_a, unsigned short *code_dots_b, unsigned short *mask_dots_a, unsigned short *mask_dots_b, size_t max_bucket_distances, unsigned long long batch_id, size_t max_query_length, size_t max_db_length)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned long long result = result1[idx] ^ result2[idx] ^ result3[idx];
        for (int i = 0; i < 64; i++)
        {
            unsigned long long query_db_rot_idx = idx * 64 + i;
            unsigned int queryIdx = (query_db_rot_idx) / chunkLength;
            unsigned int dbIdx = (query_db_rot_idx) % chunkLength;
            bool match = (result & (1ULL << i));

            // Check if we are out of bounds for the query or db
            if (queryIdx >= queryLength || dbIdx >= realChunkLen || !match)
            {
                continue;
            }


            // Save the corresponding code and mask dots for later (match distributions)
            unsigned int match_distances_counter_idx = atomicAdd(&match_distances_counter[0], 1);
            if (match_distances_counter_idx < max_bucket_distances)
            {
                // Global index for the match distances is compromise of 3 fields:
                // 1. batch_id: to distinguish between different batches, range [0, ...]
                // 2. dbIdx: to distinguish between different database entries, range [0, max_db_length)]
                // 3. queryIdx: to distinguish between different queries + their rotations, range [0, max_batch_size*ALL_ROTATIONS)
                // They are combined into a single index to allow for efficient storage and retrieval, by just treating them as a single long long integer,
                // offsetting the different parts to avoid overlaps.
                unsigned long long match_id = batch_id * (max_db_length * max_query_length) // 1.
                                              + (dbIdx + offset) * (max_query_length) // 2.
                                              + queryIdx; // 3.
                match_distances_indices[match_distances_counter_idx] = match_id;
                match_distances_buffer_codes_a[match_distances_counter_idx] = code_dots_a[idx * 64 + i];
                match_distances_buffer_codes_b[match_distances_counter_idx] = code_dots_b[idx * 64 + i];
                match_distances_buffer_masks_a[match_distances_counter_idx] = mask_dots_a[idx * 64 + i];
                match_distances_buffer_masks_b[match_distances_counter_idx] = mask_dots_b[idx * 64 + i];
            }

            unsigned int matchCounter = atomicAdd(&partialResultsCounter[0], 1);
            if (matchCounter < MAX_MATCHES_LEN * queryLength)
            {
                partialResultsQueryIndices[matchCounter] = queryIdx / ALL_ROTATIONS;
                partialResultsDbIndices[matchCounter] = dbIdx + offset;
                partialResultsRotations[matchCounter] = (queryIdx % ALL_ROTATIONS) - ROTATIONS;  // Convert to signed range [-15, 15]
            }

            // Mark which results are matches with a bit in the output
            unsigned int outputIdx = totalDbLen * (queryIdx / ALL_ROTATIONS) + dbIdx + offset;
            atomicOr(&output[outputIdx / 64], (1ULL << (outputIdx % 64)));
        }
    }
}

extern "C" __global__ void openResultsWithIndexMapping(unsigned long long *result1, unsigned long long *result2, unsigned long long *result3, unsigned long long *output, size_t chunkLength, size_t queryLength, size_t numElements, size_t realChunkLen, size_t totalDbLen, unsigned int* indexMapping, unsigned int *partialResultsCounter, unsigned int *partialResultsQueryIndices, unsigned int *partialResultsDbIndices, signed char *partialResultsRotations, unsigned short *match_distances_buffer_codes_a, unsigned short *match_distances_buffer_codes_b, unsigned short *match_distances_buffer_masks_a, unsigned short *match_distances_buffer_masks_b, unsigned int *match_distances_counter, unsigned long long *match_distances_indices, unsigned short *code_dots_a, unsigned short *code_dots_b, unsigned short *mask_dots_a, unsigned short *mask_dots_b, size_t max_bucket_distances, unsigned long long batch_id, size_t max_query_length, size_t max_db_length)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        unsigned long long result = result1[idx] ^ result2[idx] ^ result3[idx];
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / chunkLength;
            unsigned int chunkDbIdx = (idx * 64 + i) % chunkLength;
            bool match = (result & (1ULL << i));

            // Check if we are out of bounds for the query or db
            if (queryIdx >= queryLength || chunkDbIdx >= realChunkLen || !match)
            {
                continue;
            }

            unsigned int dbIdx = indexMapping[chunkDbIdx];
            unsigned int matchCounter = atomicAdd(&partialResultsCounter[0], 1);
            if (matchCounter < MAX_MATCHES_LEN * queryLength)
            {
                partialResultsQueryIndices[matchCounter] = queryIdx / ALL_ROTATIONS;
                partialResultsDbIndices[matchCounter] = dbIdx;
                partialResultsRotations[matchCounter] = (queryIdx % ALL_ROTATIONS) - ROTATIONS;  // Convert to signed range [-15, 15]
            }

            // Save the corresponding code and mask dots for later (match distributions)
            unsigned int match_distances_counter_idx = atomicAdd(&match_distances_counter[0], 1);
            if (match_distances_counter_idx < max_bucket_distances)
            {
                // Global index for the match distances is compromise of 3 fields:
                // 1. batch_id: to distinguish between different batches, range [0, ...]
                // 2. dbIdx: to distinguish between different database entries, range [0, max_db_length)]
                // 3. queryIdx: to distinguish between different queries + their rotations, range [0, max_batch_size*ALL_ROTATIONS)
                // They are combined into a single index to allow for efficient storage and retrieval, by just treating them as a single long long integer,
                // offsetting the different parts to avoid overlaps.
                unsigned long long match_id = batch_id * (max_db_length * max_query_length) // 1.
                                              + (dbIdx) * (max_query_length) // 2.
                                              + queryIdx; // 3.
                match_distances_indices[match_distances_counter_idx] = match_id;
                match_distances_buffer_codes_a[match_distances_counter_idx] = code_dots_a[idx * 64 + i];
                match_distances_buffer_codes_b[match_distances_counter_idx] = code_dots_b[idx * 64 + i];
                match_distances_buffer_masks_a[match_distances_counter_idx] = mask_dots_a[idx * 64 + i];
                match_distances_buffer_masks_b[match_distances_counter_idx] = mask_dots_b[idx * 64 + i];
            }

            // Mark which results are matches with a bit in the output
            unsigned int outputIdx = totalDbLen * (queryIdx / ALL_ROTATIONS) + dbIdx;
            atomicOr(&output[outputIdx / 64], (1ULL << (outputIdx % 64)));
        }
    }
}

extern "C" __global__ void partialDbResults(unsigned long long *matchResults, unsigned int *partialResults, size_t queryLength, size_t dbLength, size_t numElements, unsigned int *matchCounter, unsigned int maxMatches)
{

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        for (int i = 0; i < 64; i++)
        {
            unsigned int queryIdx = (idx * 64 + i) / dbLength;
            unsigned int dbIdx = (idx * 64 + i) % dbLength;
            bool match = (matchResults[idx] & (1ULL << i));

            // Check bounds
            if (queryIdx >= queryLength || dbIdx >= dbLength)
                continue;

            // Check for partial results (only used for debugging)
            if (match)
            {
                unsigned int queryMatchCounter = atomicAdd(matchCounter, 1);
                if (queryMatchCounter < maxMatches)
                    partialResults[queryMatchCounter] = dbIdx;
            }
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

extern "C" __global__ void mergeDbResultsWithOrPolicyBitmap(unsigned long long *matchResultsLeft, unsigned long long *matchResultsRight, unsigned int *finalResults, size_t queryLength, size_t dbLength, size_t numElements, size_t maxDbLength, unsigned int *matchCounter, unsigned int *allMatches, unsigned int *matchCounterLeft, unsigned int *matchCounterRight, unsigned int *partialResultsLeft, unsigned int *partialResultsRight, const unsigned long long *orPolicyBitmap, size_t numDevices, size_t deviceId)
{

    size_t rowStride64 = (maxDbLength + 63) / 64;
    size_t totalBits   = maxDbLength * queryLength;

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numElements)
    {
        for (int i = 0; i < 64; i++)
        {

            size_t deviceBit = idx * 64 + i;

            // Protect against any leftover bits if totalBits not multiple of 64
            if (deviceBit >= totalBits) break;

            unsigned int queryIdx = deviceBit / dbLength;
            unsigned int dbIdx = deviceBit % dbLength;
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

            // Recalculate the original dbIdx from the device-specific dbIdx
            size_t originalDbIdx = dbIdx * numDevices + deviceId;
            size_t orPolicyBitmapIdx = rowIndex + (originalDbIdx / 64);

            // orPolicyBitmap represents a 2D array of bits of size (max batch size x maxDbLength)
            bool useOr = (orPolicyBitmap[orPolicyBitmapIdx]
                          & (1ULL << (originalDbIdx % 64))) != 0ULL;


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
