#define uint32_t unsigned int

#define THREADS_PER_BLOCK 256 // Probably the best

/* Left rotation of n by d bits */
#define ROTL32(n, d) (n << d) | (n >> (32 - d))

#define QUARTERROUND(arr, a, b, c, d) \
    arr[a] += arr[b];                 \
    arr[d] ^= arr[a];                 \
    arr[d] = ROTL32(arr[d], 16);      \
    arr[c] += arr[d];                 \
    arr[b] ^= arr[c];                 \
    arr[b] = ROTL32(arr[b], 12);      \
    arr[a] += arr[b];                 \
    arr[d] ^= arr[a];                 \
    arr[d] = ROTL32(arr[d], 8);       \
    arr[c] += arr[d];                 \
    arr[b] ^= arr[c];                 \
    arr[b] = ROTL32(arr[b], 7);

/**
 * the chacha12_block function
 */
extern "C" __global__ void chacha12(
    uint32_t *d_ciphertext,
    uint32_t *d_state)
{
    extern __shared__ uint32_t total[16 + (THREADS_PER_BLOCK * 16)];
    uint32_t *state = &total[0];
    // copy the default state to shared mem
    if (threadIdx.x < 16)
    {
        state[threadIdx.x] = d_state[threadIdx.x];
    }
    __syncthreads();

    uint32_t *block_ct = &total[16];
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    // Localized for each thread
    uint32_t *local_ct = &block_ct[threadIdx.x * 16];
    for (int i = 0; i < 16; i++)
        local_ct[i] = state[i];
    // Adjust counter relative to thread id
    local_ct[12] = state[12] + global_id;
    for (int i = 0; i < 6; i++)
    {
        QUARTERROUND(local_ct, 0, 4, 8, 12);
        QUARTERROUND(local_ct, 1, 5, 9, 13);
        QUARTERROUND(local_ct, 2, 6, 10, 14);
        QUARTERROUND(local_ct, 3, 7, 11, 15);
        QUARTERROUND(local_ct, 0, 5, 10, 15);
        QUARTERROUND(local_ct, 1, 6, 11, 12);
        QUARTERROUND(local_ct, 2, 7, 8, 13);
        QUARTERROUND(local_ct, 3, 4, 9, 14);
    }

    local_ct[0] += state[0];
    local_ct[1] += state[1];
    local_ct[2] += state[2];
    local_ct[3] += state[3];
    local_ct[4] += state[4];
    local_ct[5] += state[5];
    local_ct[6] += state[6];
    local_ct[7] += state[7];
    local_ct[8] += state[8];
    local_ct[9] += state[9];
    local_ct[10] += state[10];
    local_ct[11] += state[11];
    local_ct[12] += state[12] + global_id;
    local_ct[13] += state[13];
    local_ct[14] += state[14];
    local_ct[15] += state[15];

    // Copy back into global memory
    uint32_t *stream_ptr = &d_ciphertext[global_id * 16];
    for (int i = 0; i < 16; i++)
    {
        stream_ptr[i] = local_ct[i];
    }
}
