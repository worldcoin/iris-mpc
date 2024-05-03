/* ChaCha12 implementation for CUDA
 * Based on the <https://github.com/eoforhild/cudacha20/>, with the following LICENSE:
 *
 * MIT License
 *
 * Copyright (c) 2024 Phuc Dang
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

#define uint16_t unsigned short
#define uint32_t unsigned int
#define uint64_t unsigned long long

#define THREADS_PER_BLOCK 256 // needs to be kept in sync with the kernel launch

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

// asm quarterround from <https://www.cloud-conf.net/ispa2021/proc/pdfs/ISPA-BDCloud-SocialCom-SustainCom2021-3mkuIWCJVSdKJpBYM7KEKW/264600b171/264600b171.pdf>
// actually seems to be noticably slower then the naive above at least on the 1070 I tested on
// #define QUARTERROUND(arr, a, b, c, d)                            \
//     asm("add.u32 %0, %0, %1; \n\t"                               \
//         "xor.b32 %3, %3, %0; \n\t"                               \
//         "shf.l.clamp.b32 %3, %3, %3, %4; \n\t"                   \
//         "add.u32 %2, %2, %3; \n\t"                               \
//         "xor.b32 %1, %1, %2; \n\t"                               \
//         "shf.l.clamp.b32 %1, %1, %1, %5; \n\t"                   \
//         "add.u32 %0, %0, %1; \n\t"                               \
//         "xor.b32 %3, %3, %0; \n\t"                               \
//         "shf.l.clamp.b32 %3, %3, %3, %6; \n\t"                   \
//         "add.u32 %2, %2, %3; \n\t"                               \
//         "xor.b32 %1, %1, %2; \n\t"                               \
//         "shf.l.clamp.b32 %1, %1, %1, %7; \n\t"                   \
//         : "+r"(arr[a]), "+r"(arr[b]), "+r"(arr[c]), "+r"(arr[d]) \
//         : "r"(16), "r"(12), "r"(8), "r"(7))

/**
 * the chacha12_block function
 */
extern "C" __global__ void chacha12(uint32_t *d_ciphertext, uint32_t *d_state, size_t len)
{
    // 16 bytes of state per thread + first 16 bytes hold a copy of the global state, which speeds up the subsequent reads
    // (we would need 2 reads from global state, which is slower than 1 global read + 1 shared write and 2 shared reads)
    extern __shared__ uint32_t buffer[16 + THREADS_PER_BLOCK * 16];

    uint32_t *state = &buffer[0];
    // copy global state into shared memory
    // only the first 16 threads copy the global state
    if (threadIdx.x < 16)
    {
        state[threadIdx.x] = d_state[threadIdx.x];
    }
    // sync threads to make sure the global state is copied
    __syncthreads();

    // each thread gets 16 bytes of state in shared memory
    uint32_t *thread_state = &buffer[16 + threadIdx.x * 16];

    // copy state into thread-local buffer (from shared to shared mem)
    for (int i = 0; i < 16; i++)
        thread_state[i] = state[i];

    // Adjust counter relative to the iteration idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // the 64-bit counter part is in state[12] and 13, we add our local counter = idx here
    // may not overflow, caller has to ensure that
    uint64_t counter = state[12] | (state[13] << 32);
    counter += idx;
    thread_state[12] = counter & 0xFFFFFFFF;
    thread_state[13] = counter >> 32;
    // 6 double rounds (8 quarter rounds)
    for (int i = 0; i < 6; i++)
    {
        QUARTERROUND(thread_state, 0, 4, 8, 12);
        QUARTERROUND(thread_state, 1, 5, 9, 13);
        QUARTERROUND(thread_state, 2, 6, 10, 14);
        QUARTERROUND(thread_state, 3, 7, 11, 15);
        QUARTERROUND(thread_state, 0, 5, 10, 15);
        QUARTERROUND(thread_state, 1, 6, 11, 12);
        QUARTERROUND(thread_state, 2, 7, 8, 13);
        QUARTERROUND(thread_state, 3, 4, 9, 14);
    }

    // Add the original state to the computed state (this would be the second read of the global state)
    thread_state[0] += state[0];
    thread_state[1] += state[1];
    thread_state[2] += state[2];
    thread_state[3] += state[3];
    thread_state[4] += state[4];
    thread_state[5] += state[5];
    thread_state[6] += state[6];
    thread_state[7] += state[7];
    thread_state[8] += state[8];
    thread_state[9] += state[9];
    thread_state[10] += state[10];
    thread_state[11] += state[11];
    thread_state[12] += state[12] + idx;
    thread_state[13] += state[13];
    thread_state[14] += state[14];
    thread_state[15] += state[15];

    // Copy back into global memory
    uint32_t *stream_ptr = &d_ciphertext[idx * 16];
    for (int i = 0; i < 16; i++)
    {
        if (idx < len) {
            stream_ptr[i] = thread_state[i];
        }
    }
}

#define P 65519

/**
 * the chacha12_block function
 */
extern "C" __global__ void fix_fe(uint32_t *d_ciphertext, uint32_t valid_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx * 1000 >= valid_size)
    {
        return;
    }
    // each thread looks at 1000 elements
    // and has 24 elements to copy from the fix array

    uint16_t *elements = (uint16_t *)d_ciphertext;
    uint16_t *fix = &elements[valid_size];
    uint16_t *my_chunk = &elements[idx * 1000];
    uint16_t *my_fix = &fix[idx * 24];

    int fix_idx = 0;

    for (int i = 0; i < 1000; i++)
    {
        while (my_chunk[i] >= P)
        {
            assert(fix_idx < 24); // should be bound with prob 2^-128 so this is fine to remove I guess
            my_chunk[i] = my_fix[fix_idx];
            fix_idx++;
        }
    }
}
