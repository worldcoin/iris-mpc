#define U16 unsigned short
#define U32 unsigned int
#define U64 unsigned long long
#define TYPE U64

#define MATCH_THRESHOLD_RATIO 0.375
#define B_BITS 20
#define B (1ULL << B_BITS)
#define A ((U64)((1. - 2. * MATCH_THRESHOLD_RATIO) * (double)B))

////////////////////////////////////////////////////////////////////////////////
// Basic Blocks (not parallelized)
////////////////////////////////////////////////////////////////////////////////

template <typename T> __device__ void not_inplace_inner(T *lhs) {
  *lhs = ~(*lhs);
}

template <typename T> __device__ void not_inner(T *res, T *lhs) {
  *res = ~(*lhs);
}

template <typename T> __device__ void xor_inner(T *res, T *lhs, T *rhs) {
  *res = *lhs ^ *rhs;
}

template <typename T> __device__ void xor_assign_inner(T *lhs, T *rhs) {
  *lhs ^= *rhs;
}

// Computes the local part of the multiplication (including randomness)
template <typename T>
__device__ void and_pre_inner(T *res_a, T *lhs_a, T *lhs_b, T *rhs_a, T *rhs_b,
                              T *r) {
  *res_a = (*lhs_a & *rhs_a) ^ (*lhs_b & *rhs_a) ^ (*lhs_a & *rhs_b) ^ *r;
}

template <typename T>
__device__ void or_pre_inner(T *res_a, T *lhs_a, T *lhs_b, T *rhs_a, T *rhs_b,
                             T *r) {
  and_pre_inner<T>(res_a, lhs_a, lhs_b, rhs_a, rhs_b, r); // AND with randomness
  *res_a ^= *lhs_a ^ *rhs_a; // XOR with the original values
}

__device__ void mul_lift_b(U64 *res, U16 *input) {
  *res = (U64)(*input) << B_BITS;
}

__device__ void u64_from_u16s(U64 *res, U16 *a, U16 *b, U16 *c, U16 *d) {
  *res = (U64)(*a) | ((U64)(*b) << 16) | ((U64)(*c) << 32) | ((U64)(*d) << 48);
}

__device__ void u64_from_u32s(U64 *res, U32 *a, U32 *b) {
  *res = (U64)(*a) | ((U64)(*b) << 32);
}

__device__ void transpose16x64(U64 *out, U16 *in) {
  // len of out = 16
  // len of in = 64

  for (U32 i = 0; i < 16; i++) {
    u64_from_u16s(&out[i], &in[i], &in[i + 16], &in[i + 32], &in[i + 48]);
  }

  U64 m = 0x00FF00FF00FF00FF;
  U32 j = 8;
  while (j != 0) {
    U32 k = 0;
    while (k < 16) {
      U64 t = ((out[k] >> j) ^ out[k + j]) & m;
      out[k + j] ^= t;
      out[k] ^= t << j;
      k = (k + j + 1) & ~j;
    }
    j >>= 1;
    m ^= (m << j);
  }
}

__device__ void transpose32x64(U64 *out, U32 *in) {
  // len of out = 32
  // len of in = 64

  for (U32 i = 0; i < 32; i++) {
    u64_from_u32s(&out[i], &in[i], &in[i + 32]);
  }

  U64 m = 0x0000FFFF0000FFFF;
  U32 j = 16;
  while (j != 0) {
    U32 k = 0;
    while (k < 32) {
      U64 t = ((out[k] >> j) ^ out[k + j]) & m;
      out[k + j] ^= t;
      out[k] ^= t << j;
      k = (k + j + 1) & ~j;
    }
    j >>= 1;
    m ^= (m << j);
  }
}

__device__ void transpose64x64(U64 *inout) {
  // len of inout = 64

  U64 m = 0x00000000FFFFFFFF;
  U32 j = 32;
  while (j != 0) {
    U32 k = 0;
    while (k < 64) {
      U64 t = ((inout[k] >> j) ^ inout[k + j]) & m;
      inout[k + j] ^= t;
      inout[k] ^= t << j;
      k = (k + j + 1) & ~j;
    }
    j >>= 1;
    m ^= (m << j);
  }
}

////////////////////////////////////////////////////////////////////////////////
// Higher level blocks
////////////////////////////////////////////////////////////////////////////////

// Performs the transpose for a and b in parallel
__device__ void u16_transpose_pack_u64(U64 *out_a, U64 *out_b, U16 *in_a,
                                       U16 *in_b, int in_len, int out_len) {
  // in has size in_len = 64 * n
  // out has size out_len, where each element is an array of n elements
  // Thus out itslef has n * out_len elements (split into n arrays)
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_len % 64 == 0);
  assert(out_len <= 16);
  int n = in_len / 64;

  // Make each transpose in parallel
  if (i < n) {
    U16 *chunk = &in_a[i * 64];
    U64 transposed[16];
    transpose16x64(transposed, chunk);

    for (U32 j = 0; j < out_len; j++) {
      out_a[j * n + i] = transposed[j];
    }
  } else if (i < 2 * n) {
    i -= n;
    U16 *chunk = &in_b[i * 64];
    U64 transposed[16];
    transpose16x64(transposed, chunk);

    for (U32 j = 0; j < out_len; j++) {
      out_b[j * n + i] = transposed[j];
    }
  }
}

// Performs the transpose for a and b in parallel
__device__ void u32_transpose_pack_u64(U64 *out_a, U64 *out_b, U32 *in_a,
                                       U32 *in_b, int in_len, int out_len) {
  // in has size in_len = 64 * n
  // out has size out_len, where each element is an array of n elements
  // Thus out itslef has n * out_len elements (split into n arrays)
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_len % 64 == 0);
  assert(out_len <= 32);
  int n = in_len / 64;

  // Make each transpose in parallel
  if (i < n) {
    U32 *chunk = &in_a[i * 64];
    U64 transposed[32];
    transpose32x64(transposed, chunk);

    for (U32 j = 0; j < out_len; j++) {
      out_a[j * n + i] = transposed[j];
    }
  } else if (i < 2 * n) {
    i -= n;
    U32 *chunk = &in_b[i * 64];
    U64 transposed[32];
    transpose32x64(transposed, chunk);

    for (U32 j = 0; j < out_len; j++) {
      out_b[j * n + i] = transposed[j];
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Test kernels
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void shared_assign(TYPE *des_a, TYPE *des_b, TYPE *src_a,
                                         TYPE *src_b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    des_a[i] = src_a[i];
    des_b[i] = src_b[i];
  }
}

extern "C" __global__ void shared_not_inplace(TYPE *lhs_a, TYPE *lhs_b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    not_inplace_inner<TYPE>(&lhs_a[i]);
    not_inplace_inner<TYPE>(&lhs_b[i]);
  }
}

extern "C" __global__ void shared_not(TYPE *res_a, TYPE *res_b, TYPE *lhs_a,
                                      TYPE *lhs_b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    not_inner<TYPE>(&res_a[i], &lhs_a[i]);
    not_inner<TYPE>(&res_b[i], &lhs_b[i]);
  }
}

extern "C" __global__ void shared_xor(TYPE *res_a, TYPE *res_b, TYPE *lhs_a,
                                      TYPE *lhs_b, TYPE *rhs_a, TYPE *rhs_b,
                                      int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_inner<TYPE>(&res_a[i], &lhs_a[i], &rhs_a[i]);
    xor_inner<TYPE>(&res_b[i], &lhs_b[i], &rhs_b[i]);
  }
}

extern "C" __global__ void shared_xor_assign(TYPE *lhs_a, TYPE *lhs_b,
                                             TYPE *rhs_a, TYPE *rhs_b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_assign_inner<TYPE>(&lhs_a[i], &rhs_a[i]);
    xor_assign_inner<TYPE>(&lhs_b[i], &rhs_b[i]);
  }
}

extern "C" __global__ void shared_and_pre(TYPE *res_a, TYPE *lhs_a, TYPE *lhs_b,
                                          TYPE *rhs_a, TYPE *rhs_b, TYPE *r,
                                          int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    and_pre_inner<TYPE>(&res_a[i], &lhs_a[i], &lhs_b[i], &rhs_a[i], &rhs_b[i],
                        &r[i]);
  }
}

extern "C" __global__ void shared_or_pre_assign(TYPE *lhs_a, TYPE *lhs_b,
                                                TYPE *rhs_a, TYPE *rhs_b,
                                                TYPE *r, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    U64 res_a;
    or_pre_inner<TYPE>(&res_a, &lhs_a[i], &lhs_b[i], &rhs_a[i], &rhs_b[i],
                       &r[i]);
    lhs_a[i] = res_a;
  }
}

extern "C" __global__ void shared_mul_lift_b(U64 *res_a, U64 *res_b, U16 *lhs_a,
                                             U16 *lhs_b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    mul_lift_b(&res_a[i], &lhs_a[i]);
    mul_lift_b(&res_b[i], &lhs_b[i]);
  }
}

extern "C" __global__ void shared_u16_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U16 *in_a, U16 *in_b,
                                                         int in_len,
                                                         int out_len) {
  u16_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
}

extern "C" __global__ void shared_u32_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U32 *in_a, U32 *in_b,
                                                         int in_len,
                                                         int out_len) {
  u32_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
}

extern "C" __global__ void
shared_u64_transpose_pack_u64_global_mem(U64 *out_a, U64 *out_b, U64 *in_a,
                                         U64 *in_b, int in_len, int out_len) {
  // in has size in_len = 64 * n
  // out has size out_len, where each element is an array of n elements
  // Thus out itslef has n * out_len elements (split into n arrays)
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_len % 64 == 0);
  assert(out_len <= 64);
  int n = in_len / 64;

  // Make each transpose in parallel
  if (i < n) {
    U64 *chunk = &in_a[i * 64];

    transpose64x64(chunk);

    for (U32 j = 0; j < out_len; j++) {
      out_a[j * n + i] = chunk[j];
    }
  } else if (i < 2 * n) {
    i -= n;
    U64 *chunk = &in_b[i * 64];
    transpose64x64(chunk);

    for (U32 j = 0; j < out_len; j++) {
      out_b[j * n + i] = chunk[j];
    }
  }
}

extern "C" __global__ void shared_u64_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U64 *in_a, U64 *in_b,
                                                         int in_len,
                                                         int out_len) {
  // in has size in_len = 64 * n
  // out has size out_len, where each element is an array of n elements
  // Thus out itslef has n * out_len elements (split into n arrays)
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_len % 64 == 0);
  assert(out_len <= 64);
  int n = in_len / 64;

  // Make each transpose in parallel
  U64 transposed[64];
  if (i < n) {
    U64 *chunk = &in_a[i * 64];
    for (U32 j = 0; j < 64; j++) {
      transposed[j] = chunk[j];
    }

    transpose64x64(transposed);

    for (U32 j = 0; j < out_len; j++) {
      out_a[j * n + i] = transposed[j];
    }
  } else if (i < 2 * n) {
    i -= n;
    U64 *chunk = &in_b[i * 64];
    for (U32 j = 0; j < 64; j++) {
      transposed[j] = chunk[j];
    }
    transpose64x64(transposed);

    for (U32 j = 0; j < out_len; j++) {
      out_b[j * n + i] = transposed[j];
    }
  }
}

extern "C" __global__ void packed_ot_sender(U32 *out_a, U32 *out_b, U64 *in_a,
                                            U64 *in_b, U32 *m0, U32 *m1,
                                            U32 *rand_ca, U32 *rand_cb,
                                            U32 *rand_wa1, U32 *rand_wa2,
                                            int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 32-bit
  // Thus, in has size n, out has size 64 * n
  // m0, m1, rand_ca, rand_cb, rand_wa1, rand_wa2 are same size as out

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    int wordindex = i / 64;
    int bitindex = i % 64;
    U32 my_bit_a = (in_a[wordindex] >> bitindex) & 1;
    U32 my_bit_b = (in_b[wordindex] >> bitindex) & 1;
    out_a[i] = rand_ca[i];
    out_b[i] = rand_cb[i];
    U32 c = rand_ca[i] + rand_cb[i];
    U32 xor_ab = my_bit_a ^ my_bit_b;
    m0[i] = ((xor_ab ^ 1) - c) ^ rand_wa1[i]; // Negation is included in OT
    m1[i] = (xor_ab - c) ^ rand_wa2[i];       // Negation is included in OT
  }
}

extern "C" __global__ void packed_ot_receiver(U32 *out_a, U32 *out_b, U64 *in_b,
                                              U32 *m0, U32 *m1, U32 *rand_ca,
                                              U32 *rand_wc, int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 32-bit
  // Thus, in has size n, out has size 64 * n
  // m0, m1, rand_ca, rand_wc are same size as out

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    int wordindex = i / 64;
    int bitindex = i % 64;
    bool my_bit_b = ((in_b[wordindex] >> bitindex) & 1) == 1;
    out_a[i] = rand_ca[i];
    if (my_bit_b) {
      out_b[i] = rand_wc[i] ^ m1[i];
    } else {
      out_b[i] = rand_wc[i] ^ m0[i];
    }
  }
}

extern "C" __global__ void packed_ot_helper(U32 *out_b, U64 *in_a, U32 *rand_cb,
                                            U32 *rand_wb1, U32 *rand_wb2,
                                            U32 *wc, int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 32-bit
  // Thus, in has size n, out has size 64 * n
  // rand_wb1, rand_wb2, wc are same size as out

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    int wordindex = i / 64;
    int bitindex = i % 64;
    bool my_bit_a = ((in_a[wordindex] >> bitindex) & 1) == 1;
    out_b[i] = rand_cb[i];
    if (my_bit_a) {
      wc[i] = rand_wb2[i];
    } else {
      wc[i] = rand_wb1[i];
    }
  }
}

extern "C" __global__ void collapse_u64_helper(U64 *inout_a, U64 *in_b,
                                               U64 *helper_a, U64 *helper_b,
                                               U64 *r, int next_bitsize) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    U64 mask = (1 << next_bitsize) - 1;
    *helper_a = *inout_a >> next_bitsize;
    *helper_b = *in_b >> next_bitsize;

    U64 res_a;
    or_pre_inner<TYPE>(&res_a, inout_a, in_b, helper_a, helper_b, r);
    *inout_a = res_a & mask;
  }
}
