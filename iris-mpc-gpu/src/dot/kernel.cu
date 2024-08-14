#define UINT_MAX 0xffffffff
#define ROTATIONS 15
#define TOTAL_ROTATIONS (2 * ROTATIONS + 1)
#define IRIS_CODE_LENGTH 12800
#define U8 unsigned char

extern "C" __global__ void xor_assign_u8(U8 *lhs, U8 *rhs, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lhs[i] ^= rhs[i];
  }
}

extern "C" __global__ void matmul_correct_and_reduce(
    int *c, unsigned short *output, int *a0Sums, int *a1Sums, int *b0Sums,
    int *b1Sums, size_t dbLength, size_t numElements, size_t offset,
    unsigned short *rngMasks0, unsigned short *rngMasks1) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    unsigned int queryIdx = idx / dbLength;
    unsigned int dbIdx = idx % dbLength;
    int s0 = a0Sums[offset + dbIdx] + b0Sums[queryIdx];
    int s1 = a1Sums[offset + dbIdx] + b1Sums[queryIdx];
    output[idx] = c[idx] + (s0 << 7) + ((s0 + s1) << 15) + rngMasks0[idx] -
                  rngMasks1[idx];
  }
}

/// Takes the 3 additive shares of the result, and recombines them into a single
/// one. Furthermore, it combines the 31 ROTATIONs into a single bit using OR.
extern "C" __global__ void recombineResults(unsigned long long *result1,
                                            unsigned long long *result2,
                                            unsigned long long *result3,
                                            unsigned long long *output,
                                            size_t numQueries) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numQueries) {
    // each one of these is targeting 31 bits
    // we might need to access two elements
    // compute the bitmasks for the current rotations
    size_t result_idx = (idx * TOTAL_ROTATIONS) / 64;
    size_t inter_u64_offset = (idx * TOTAL_ROTATIONS) % 64;
    // if this is >= 31 then mask2 will be 0, which happens if inter_u64_offset
    // is <=33
    size_t inter_u64_offset2 = 64 - inter_u64_offset;
    unsigned long long mask1 =
        0x7FFFFFFF << inter_u64_offset; // 31 bits set, shifted by offset
    unsigned long long mask2 =
        0x7FFFFFFF >> inter_u64_offset2; // 31 bits set, shifted by offset2
    // if any bit is set in these two elements, we have a match
    unsigned long long result =
        (mask1 & (result1[result_idx] ^ result2[result_idx] ^
                  result3[result_idx])) != 0;
    // we need this if to not access out of bounds for the last element
    if (mask2 != 0) {
      result |= (mask2 & (result1[result_idx + 1] ^ result2[result_idx + 1] ^
                          result3[result_idx + 1])) != 0;
    }
    // set the bit in the output, atomically since we have multiple threads
    // accessing the same memory
    atomicOr(&output[idx / 64], result << (idx % 64));
  }
}

extern "C" __global__ void mapResults(unsigned long long *result_left,
                                      unsigned long long *result_right,
                                      unsigned int *output, size_t dbLength,
                                      size_t queryLength, size_t offset,
                                      size_t numElements, size_t realDbLen) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    // MATCHING logic: if both eyes match on same bit, we have a match
    unsigned long long result = result_left[idx] & result_right[idx];
    for (int i = 0; i < 64; i++) {
      unsigned int queryIdx = (idx * 64 + i) / dbLength;
      unsigned int dbIdx = (idx * 64 + i) % dbLength;
      bool match = (result & (1ULL << i));

      // Check if we are out of bounds for the query or db
      if (queryIdx >= queryLength || dbIdx >= realDbLen) {
        continue;
      }

      // return db element with smallest index
      if (match)
        atomicMin(&output[queryIdx], dbIdx + offset);
    }
  }
}

extern "C" __global__ void mergeResults(unsigned int *matchResultsSelf,
                                        unsigned int *matchResults,
                                        unsigned int *finalResults,
                                        size_t queryLength) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < queryLength) {
    bool match = false;

    // Check if there is a match in the db
    for (int r = 0; r <= ROTATIONS * 2; r++) {
      int oldIdx = idx * (2 * ROTATIONS + 1) + r;
      if (matchResults[oldIdx] != UINT_MAX) {
        finalResults[idx] = matchResults[oldIdx];
        match = true;
      }
    }

    // If there is a match in the db, we return the db index
    if (match)
      return;

    // Check if there is a match in the query itelf
    // We only need to check a single query, since we don't want to rotate
    // double
    int oldIdx = idx * (2 * ROTATIONS + 1) + ROTATIONS;
    if (matchResultsSelf[oldIdx] != UINT_MAX &&
        oldIdx != matchResultsSelf[oldIdx]) {
      finalResults[idx] = UINT_MAX - 1; // Set to UINT_MAX - 1 to indicate that
                                        // the match is in the query itself
      return;
    }

    finalResults[idx] = UINT_MAX;
  }
}
