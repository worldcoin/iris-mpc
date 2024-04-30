/* Simultaneous evaluation of two instances of ChaCha12 for CUDA
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
 * two interleaved chacha12_block functions
 */
extern "C" __global__ void chacha12_two(uint32_t *d_ciphertext, uint32_t *d_state1, uint32_t *d_state2)
{
    // 32 bytes of state per thread + first 32 bytes hold a copy of the global state (2x 16 bytes), which speeds up the subsequent reads
    // (we would need 2 reads from global state, which is slower than 1 global read + 1 shared write and 2 shared reads)
    extern __shared__ uint32_t buffer[32 + THREADS_PER_BLOCK * 32];

    uint32_t *state = &buffer[0];
    // copy global state into shared memory
    // only the first 32 threads copy the global state
    if (threadIdx.x < 16)
    {
        state[threadIdx.x] = d_state1[threadIdx.x];
    }
    if (threadIdx.x >= 16 && threadIdx.x < 32)
    {
        state[threadIdx.x] = d_state2[threadIdx.x - 16];
    }
    // sync threads to make sure the global state is copied
    __syncthreads();

    // each thread gets 16 bytes of state in shared memory
    uint32_t *thread_state1 = &buffer[32 + threadIdx.x * 32];
    uint32_t *thread_state2 = &thread_state1[16];

    // copy state into thread-local buffer (from shared to shared mem)
    for (int i = 0; i < 32; i++)
        thread_state1[i] = state[i];

    // Adjust counter relative to the iteration idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // the 64-bit counter part is in state[12] and state[13], we add our local counter = idx here
    uint64_t counter = thread_state1[12] | (thread_state2[13] << 32);
    counter += idx;
    thread_state1[12] = counter & 0xFFFFFFFF;
    thread_state1[13] = counter >> 32;
    counter = thread_state1[28] | (thread_state2[29] << 32);
    counter += idx;
    thread_state1[28] = counter & 0xFFFFFFFF;
    thread_state1[29] = counter >> 32;

    // 6 double rounds (8 quarter rounds)
    for (int i = 0; i < 6; i++)
    {
        QUARTERROUND(thread_state1, 0, 4, 8, 12);
        QUARTERROUND(thread_state2, 0, 4, 8, 12);
        QUARTERROUND(thread_state1, 1, 5, 9, 13);
        QUARTERROUND(thread_state2, 1, 5, 9, 13);
        QUARTERROUND(thread_state1, 2, 6, 10, 14);
        QUARTERROUND(thread_state2, 2, 6, 10, 14);
        QUARTERROUND(thread_state1, 3, 7, 11, 15);
        QUARTERROUND(thread_state2, 3, 7, 11, 15);
        QUARTERROUND(thread_state1, 0, 5, 10, 15);
        QUARTERROUND(thread_state2, 0, 5, 10, 15);
        QUARTERROUND(thread_state1, 1, 6, 11, 12);
        QUARTERROUND(thread_state2, 1, 6, 11, 12);
        QUARTERROUND(thread_state1, 2, 7, 8, 13);
        QUARTERROUND(thread_state2, 2, 7, 8, 13);
        QUARTERROUND(thread_state1, 3, 4, 9, 14);
        QUARTERROUND(thread_state2, 3, 4, 9, 14);
    }

    // Add the original state to the computed state (this would be the second read of the global state)
    thread_state1[0] += state[0];
    thread_state1[1] += state[1];
    thread_state1[2] += state[2];
    thread_state1[3] += state[3];
    thread_state1[4] += state[4];
    thread_state1[5] += state[5];
    thread_state1[6] += state[6];
    thread_state1[7] += state[7];
    thread_state1[8] += state[8];
    thread_state1[9] += state[9];
    thread_state1[10] += state[10];
    thread_state1[11] += state[11];
    thread_state1[12] += state[12] + idx;
    thread_state1[13] += state[13];
    thread_state1[14] += state[14];
    thread_state1[15] += state[15];
    thread_state2[0] += state[16];
    thread_state2[1] += state[17];
    thread_state2[2] += state[18];
    thread_state2[3] += state[19];
    thread_state2[4] += state[20];
    thread_state2[5] += state[21];
    thread_state2[6] += state[22];
    thread_state2[7] += state[23];
    thread_state2[8] += state[24];
    thread_state2[9] += state[25];
    thread_state2[10] += state[26];
    thread_state2[11] += state[27];
    thread_state2[12] += state[28] + idx;
    thread_state2[13] += state[29];
    thread_state2[14] += state[30];
    thread_state2[15] += state[31];

    // Copy back into global memory
    uint32_t *stream_ptr = &d_ciphertext[idx * 16];
    for (int i = 0; i < 16; i++)
    {
        stream_ptr[i] = thread_state1[i] ^ thread_state2[i];
    }
}

/**
 * two sequential chacha12_block functions
 */
extern "C" __global__ void chacha12_two_seq(uint32_t *d_ciphertext, uint32_t *d_state1, uint32_t *d_state2)
{
    // 32 bytes of state per thread + first 32 bytes hold a copy of the global state (2x 16 bytes), which speeds up the subsequent reads
    // (we would need 2 reads from global state, which is slower than 1 global read + 1 shared write and 2 shared reads)
    extern __shared__ uint32_t buffer[32 + THREADS_PER_BLOCK * 32];

    uint32_t *state = &buffer[0];
    // copy global state into shared memory
    // only the first 32 threads copy the global state
    if (threadIdx.x < 16)
    {
        state[threadIdx.x] = d_state1[threadIdx.x];
    }
    if (threadIdx.x >= 16 && threadIdx.x < 32)
    {
        state[threadIdx.x] = d_state2[threadIdx.x - 16];
    }
    // sync threads to make sure the global state is copied
    __syncthreads();

    // each thread gets 16 bytes of state in shared memory
    uint32_t *thread_state = &buffer[32 + threadIdx.x * 32];
    uint32_t *thread_state2 = &thread_state[16];

    // copy state into thread-local buffer (from shared to shared mem)
    for (int i = 0; i < 16; i++)
        thread_state[i] = state[i];

    // Adjust counter relative to the iteration idx
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // the 64-bit counter part is in state[12] and state[13], we add our local counter = idx here
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
    thread_state2[0] = thread_state[0] + state[0];
    thread_state2[1] = thread_state[1] + state[1];
    thread_state2[2] = thread_state[2] + state[2];
    thread_state2[3] = thread_state[3] + state[3];
    thread_state2[4] = thread_state[4] + state[4];
    thread_state2[5] = thread_state[5] + state[5];
    thread_state2[6] = thread_state[6] + state[6];
    thread_state2[7] = thread_state[7] + state[7];
    thread_state2[8] = thread_state[8] + state[8];
    thread_state2[9] = thread_state[9] + state[9];
    thread_state2[10] = thread_state[10] + state[10];
    thread_state2[11] = thread_state[11] + state[11];
    thread_state2[12] = thread_state[12] + state[12] + idx;
    thread_state2[13] = thread_state[13] + state[13];
    thread_state2[14] = thread_state[14] + state[14];
    thread_state2[15] = thread_state[15] + state[15];

    // second instance
    // copy state into thread-local buffer (from shared to shared mem)
    for (int i = 0; i < 16; i++)
        thread_state[i] = state[i + 16];

    // the 64-bit counter part is in state[12] and state[13], we add our local counter = idx here
    counter = state[12] | (state[13] << 32);
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
    thread_state[0] += state[16];
    thread_state[1] += state[17];
    thread_state[2] += state[18];
    thread_state[3] += state[19];
    thread_state[4] += state[20];
    thread_state[5] += state[21];
    thread_state[6] += state[22];
    thread_state[7] += state[23];
    thread_state[8] += state[24];
    thread_state[9] += state[25];
    thread_state[10] += state[26];
    thread_state[11] += state[27];
    thread_state[12] += state[28] + idx;
    thread_state[13] += state[29];
    thread_state[14] += state[30];
    thread_state[15] += state[31];

    // Copy back into global memory
    uint32_t *stream_ptr = &d_ciphertext[idx * 16];
    for (int i = 0; i < 16; i++)
    {
        stream_ptr[i] = thread_state[i] ^ thread_state2[i];
    }
}
