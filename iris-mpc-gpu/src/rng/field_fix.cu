#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long long

#define P 65519

/**
 * the chacha12_block function
 */
extern "C" __global__ void fix_fe(uint32_t *d_ciphertext, uint32_t valid_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx * 1000 >= valid_size) {
    return;
  }
  // each thread looks at 1000 elements
  // and has 24 elements to copy from the fix array

  uint16_t *elements = (uint16_t *)d_ciphertext;
  uint16_t *fix = &elements[valid_size];
  uint16_t *my_chunk = &elements[idx * 1000];
  uint16_t *my_fix = &fix[idx * 24];

  int fix_idx = 0;

  for (int i = 0; i < 1000; i++) {
    while (my_chunk[i] >= P) {
      assert(fix_idx < 24); // should be bound with prob 2^-128 so this is fine
                            // to remove I guess
      my_chunk[i] = my_fix[fix_idx];
      fix_idx++;
    }
  }
}
