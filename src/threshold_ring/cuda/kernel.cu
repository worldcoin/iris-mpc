#define U16 unsigned short
#define U32 unsigned int
#define U64 unsigned long long
#define TYPE U64

#define MATCH_THRESHOLD_RATIO 0.375
<<<<<<< HEAD
#define B_BITS 16
=======
#define B_BITS 20
>>>>>>> merge/threshold
#define B (1ULL << B_BITS)
#define A ((U64)((1. - 2. * MATCH_THRESHOLD_RATIO) * (double)B))

////////////////////////////////////////////////////////////////////////////////
// Basic Blocks (not parallelized)
////////////////////////////////////////////////////////////////////////////////

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

<<<<<<< HEAD
__device__ void mul_lift_b(U32 *res, U16 *input) {
  *res = (U32)(*input) << B_BITS;
=======
__device__ void mul_lift_b(U64 *res, U16 *input) {
  *res = (U64)(*input) << B_BITS;
>>>>>>> merge/threshold
}

__device__ void u64_from_u16s(U64 *res, U16 *a, U16 *b, U16 *c, U16 *d) {
  *res = (U64)(*a) | ((U64)(*b) << 16) | ((U64)(*c) << 32) | ((U64)(*d) << 48);
}

<<<<<<< HEAD
__device__ void u64_from_u32s(U64 *res, U32 *a, U32 *b) {
  *res = (U64)(*a) | ((U64)(*b) << 32);
}

=======
>>>>>>> merge/threshold
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

<<<<<<< HEAD
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
=======
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
>>>>>>> merge/threshold
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

<<<<<<< HEAD
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

__device__ void lift_mul_sub(U32 *mask, U16 *mask_corr1, U16 *mask_corr2,
                             U16 *code) {
  *mask -= (U32)(*mask_corr1) << 16;
  *mask -= (U32)(*mask_corr2) << 17;

  U32 a;
=======
__device__ void lift_mul_sub(U64 *mask, U32 *mask_corr1, U32 *mask_corr2,
                             U16 *code) {
  *mask -= (U64)(*mask_corr1) << 16;
  *mask -= (U64)(*mask_corr2) << 17;

  U64 a;
>>>>>>> merge/threshold
  mul_lift_b(&a, code);
  *mask *= A;
  *mask -= a;
}

__device__ void split_inner(U64 *x1_a, U64 *x1_b, U64 *x2_a, U64 *x2_b,
                            U64 *x3_a, U64 *x3_b, int id) {
  U64 tmp_a = *x1_a;
  U64 tmp_b = *x1_b;
  switch (id) {
  case 0:
    *x1_a = tmp_a;
    *x1_b = 0;
    *x2_a = 0;
    *x2_b = 0;
    *x3_a = 0;
    *x3_b = tmp_b;
    break;
  case 1:
    *x1_a = 0;
    *x1_b = tmp_b;
    *x2_a = tmp_a;
    *x2_b = 0;
    *x3_a = 0;
    *x3_b = 0;
    break;
  case 2:
    *x1_a = 0;
    *x1_b = 0;
    *x2_a = 0;
    *x2_b = tmp_b;
    *x3_a = tmp_a;
    *x3_b = 0;
    break;
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

extern "C" __global__ void shared_u16_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U16 *in_a, U16 *in_b,
                                                         int in_len,
                                                         int out_len) {
  u16_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
}

<<<<<<< HEAD
extern "C" __global__ void shared_u32_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U32 *in_a, U32 *in_b,
                                                         int in_len,
                                                         int out_len) {
  u32_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
=======
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
>>>>>>> merge/threshold
}

extern "C" __global__ void split(U64 *x1_a, U64 *x1_b, U64 *x2_a, U64 *x2_b,
                                 U64 *x3_a, U64 *x3_b, int n, int id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    split_inner(&x1_a[i], &x1_b[i], &x2_a[i], &x2_b[i], &x3_a[i], &x3_b[i], id);
  }
}

<<<<<<< HEAD
extern "C" __global__ void lift_split(U16 *in_a, U16 *in_b, U32 *lifted_a,
                                      U32 *lifted_b, U64 *x1_a, U64 *x1_b,
=======
extern "C" __global__ void lift_split(U16 *in_a, U16 *in_b, U64 *lifted_a,
                                      U64 *lifted_b, U64 *x1_a, U64 *x1_b,
>>>>>>> merge/threshold
                                      U64 *x2_a, U64 *x2_b, U64 *x3_a,
                                      U64 *x3_b, int chunk_size, int id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * chunk_size) {
<<<<<<< HEAD
    lifted_a[i] = (U32)(in_a[i]);
    lifted_b[i] = (U32)(in_b[i]);
=======
    lifted_a[i] = (U64)(in_a[i]);
    lifted_b[i] = (U64)(in_b[i]);
>>>>>>> merge/threshold
  }
  if (i < 16 * chunk_size) {
    split_inner(&x1_a[i], &x1_b[i], &x2_a[i], &x2_b[i], &x3_a[i], &x3_b[i], id);
  }
}

// Puts the results into mask_a, mask_b and x01
<<<<<<< HEAD
extern "C" __global__ void shared_lift_mul_sub(U32 *mask_a, U32 *mask_b,
                                               U16 *mask_corr_a,
                                               U16 *mask_corr_b, U16 *code_a,
=======
extern "C" __global__ void shared_lift_mul_sub(U64 *mask_a, U64 *mask_b,
                                               U32 *mask_corr_a,
                                               U32 *mask_corr_b, U16 *code_a,
>>>>>>> merge/threshold
                                               U16 *code_b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lift_mul_sub(&mask_a[i], &mask_corr_a[i], &mask_corr_a[i + n], &code_a[i]);
    lift_mul_sub(&mask_b[i], &mask_corr_b[i], &mask_corr_b[i + n], &code_b[i]);
  }
}

<<<<<<< HEAD
extern "C" __global__ void packed_ot_sender(U16 *out_a, U16 *out_b, U64 *in_a,
                                            U64 *in_b, U16 *m0, U16 *m1,
                                            U16 *rand_ca, U16 *rand_cb,
                                            U16 *rand_wa1, U16 *rand_wa2,
                                            int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 16-bit
=======
extern "C" __global__ void packed_ot_sender(U32 *out_a, U32 *out_b, U64 *in_a,
                                            U64 *in_b, U32 *m0, U32 *m1,
                                            U32 *rand_ca, U32 *rand_cb,
                                            U32 *rand_wa1, U32 *rand_wa2,
                                            int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 32-bit
>>>>>>> merge/threshold
  // Thus, in has size n, out has size 64 * n
  // m0, m1, rand_ca, rand_cb, rand_wa1, rand_wa2 are same size as out

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    int wordindex = i / 64;
    int bitindex = i % 64;
<<<<<<< HEAD
    U16 my_bit_a = (in_a[wordindex] >> bitindex) & 1;
    U16 my_bit_b = (in_b[wordindex] >> bitindex) & 1;
    out_a[i] = rand_ca[i];
    out_b[i] = rand_cb[i];
    U16 c = rand_ca[i] + rand_cb[i];
    U16 xor_ab = my_bit_a ^ my_bit_b;
=======
    U32 my_bit_a = (in_a[wordindex] >> bitindex) & 1;
    U32 my_bit_b = (in_b[wordindex] >> bitindex) & 1;
    out_a[i] = rand_ca[i];
    out_b[i] = rand_cb[i];
    U32 c = rand_ca[i] + rand_cb[i];
    U32 xor_ab = my_bit_a ^ my_bit_b;
>>>>>>> merge/threshold
    m0[i] = (xor_ab - c) ^ rand_wa1[i];
    m1[i] = ((xor_ab ^ 1) - c) ^ rand_wa2[i];
  }
}

<<<<<<< HEAD
extern "C" __global__ void packed_ot_receiver(U16 *out_a, U16 *out_b, U64 *in_b,
                                              U16 *m0, U16 *m1, U16 *rand_ca,
                                              U16 *rand_wc, int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 16-bit
=======
extern "C" __global__ void packed_ot_receiver(U32 *out_a, U32 *out_b, U64 *in_b,
                                              U32 *m0, U32 *m1, U32 *rand_ca,
                                              U32 *rand_wc, int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 32-bit
>>>>>>> merge/threshold
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

<<<<<<< HEAD
extern "C" __global__ void packed_ot_helper(U16 *out_b, U64 *in_a, U16 *rand_cb,
                                            U16 *rand_wb1, U16 *rand_wb2,
                                            U16 *wc, int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into U16-bit
=======
extern "C" __global__ void packed_ot_helper(U32 *out_b, U64 *in_a, U32 *rand_cb,
                                            U32 *rand_wb1, U32 *rand_wb2,
                                            U32 *wc, int n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 32-bit
>>>>>>> merge/threshold
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
