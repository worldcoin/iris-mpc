#define U16 unsigned short
#define U32 unsigned int
#define U64 unsigned long long
#define TYPE U64

#define B_BITS 16
#define B (1ULL << B_BITS)

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
__device__ void arithmetic_xor_inner(T *res_a, T *lhs_a, T *lhs_b, T *rhs_a,
                                     T *rhs_b, T *r1, T *r2) {
  T lhs_a_val = *lhs_a;
  T lhs_b_val = *lhs_b;
  T rhs_a_val = *rhs_a;
  T rhs_b_val = *rhs_b;
  T r1_val = *r1;
  T r2_val = *r2;

  T mul = (lhs_a_val * rhs_a_val) + (lhs_b_val * rhs_a_val) +
          (lhs_a_val * rhs_b_val) + r1_val - r2_val;
  *res_a = lhs_a_val + rhs_a_val - 2 * mul;
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

__device__ void mul_lift_b(U32 *res, U16 *input) {
  *res = (U32)(*input) << B_BITS;
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

////////////////////////////////////////////////////////////////////////////////
// Higher level blocks
////////////////////////////////////////////////////////////////////////////////

// Performs the transpose for a and b in parallel
__device__ void u16_transpose_pack_u64(U64 *out_a, U64 *out_b, U16 *in_a,
                                       U16 *in_b, size_t in_len,
                                       size_t out_len) {
  // in has size in_len = 64 * n
  // out has size out_len, where each element is an array of n elements
  // Thus out itslef has n * out_len elements (split into n arrays)
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_len % 64 == 0);
  assert(out_len <= 16);
  size_t n = in_len / 64;

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
                                       U32 *in_b, size_t in_len,
                                       size_t out_len) {
  // in has size in_len = 64 * n
  // out has size out_len, where each element is an array of n elements
  // Thus out itslef has n * out_len elements (split into n arrays)
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  assert(in_len % 64 == 0);
  assert(out_len <= 32);
  size_t n = in_len / 64;

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

__device__ void finalize_lift(U32 *mask, U32 *code_lift, U16 *mask_corr1,
                              U16 *mask_corr2, U16 *code) {
  *mask -= (U32)(*mask_corr1) << 16;
  *mask -= (U32)(*mask_corr2) << 17;

  mul_lift_b(code_lift, code);
}

__device__ void lift_mul_sub(U32 *mask, U16 *mask_corr1, U16 *mask_corr2,
                             U16 *code, U32 a) {
  U32 lifted;
  finalize_lift(mask, &lifted, mask_corr1, mask_corr2, code);
  *mask *= a;
  *mask -= lifted;
}

__device__ void lifted_sub(U32 *mask, U32 *code, U32 *output, U32 a) {
  *output = *mask * a - *code;
}

__device__ void prelifted_sub_ab(U32 *mask, U32 *code, U32 *output, U32 a) {
  *output = *mask * a - (*code << B_BITS);
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

__device__ void split_for_arithmetic_xor_inner(U32 *x1_a, U32 *x1_b, U32 *x2_a,
                                               U32 *x2_b, U32 *x3_a, U32 *x3_b,
                                               U32 inp_a, U32 inp_b, int id) {
  switch (id) {
  case 0:
    *x1_a = inp_a;
    *x1_b = 0;
    *x2_a = 0;
    *x2_b = 0;
    *x3_a = 0;
    *x3_b = inp_b;
    break;
  case 1:
    *x1_a = 0;
    *x1_b = inp_b;
    *x2_a = inp_a;
    *x2_b = 0;
    *x3_a = 0;
    *x3_b = 0;
    break;
  case 2:
    *x1_a = 0;
    *x1_b = 0;
    *x2_a = 0;
    *x2_b = inp_b;
    *x3_a = inp_a;
    *x3_b = 0;
    break;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Test kernels
////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void shared_assign(TYPE *des_a, TYPE *des_b, TYPE *src_a,
                                         TYPE *src_b, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    des_a[i] = src_a[i];
    des_b[i] = src_b[i];
  }
}

extern "C" __global__ void shared_xor(TYPE *res_a, TYPE *res_b, TYPE *lhs_a,
                                      TYPE *lhs_b, TYPE *rhs_a, TYPE *rhs_b,
                                      size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_inner<TYPE>(&res_a[i], &lhs_a[i], &rhs_a[i]);
    xor_inner<TYPE>(&res_b[i], &lhs_b[i], &rhs_b[i]);
  }
}

extern "C" __global__ void shared_xor_assign(TYPE *lhs_a, TYPE *lhs_b,
                                             TYPE *rhs_a, TYPE *rhs_b,
                                             size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_assign_inner<TYPE>(&lhs_a[i], &rhs_a[i]);
    xor_assign_inner<TYPE>(&lhs_b[i], &rhs_b[i]);
  }
}

extern "C" __global__ void xor_assign_u16(U16 *lhs, U16 *rhs, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_assign_inner<U16>(&lhs[i], &rhs[i]);
  }
}

extern "C" __global__ void xor_assign_u32(U32 *lhs, U32 *rhs, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_assign_inner<U32>(&lhs[i], &rhs[i]);
  }
}

extern "C" __global__ void xor_assign_u64(U64 *lhs, U64 *rhs, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xor_assign_inner<U64>(&lhs[i], &rhs[i]);
  }
}

extern "C" __global__ void
shared_arithmetic_xor_pre_assign_u32(U32 *lhs_a, U32 *lhs_b, U32 *rhs_a,
                                     U32 *rhs_b, U32 *r1, U32 *r2, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    U32 res_a;
    arithmetic_xor_inner<U32>(&res_a, &lhs_a[i], &lhs_b[i], &rhs_a[i],
                              &rhs_b[i], &r1[i], &r2[i]);
    lhs_a[i] = res_a;
  }
}

extern "C" __global__ void shared_and_pre(TYPE *res_a, TYPE *lhs_a, TYPE *lhs_b,
                                          TYPE *rhs_a, TYPE *rhs_b, TYPE *r,
                                          size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    and_pre_inner<TYPE>(&res_a[i], &lhs_a[i], &lhs_b[i], &rhs_a[i], &rhs_b[i],
                        &r[i]);
  }
}

extern "C" __global__ void shared_or_pre_assign(TYPE *lhs_a, TYPE *lhs_b,
                                                TYPE *rhs_a, TYPE *rhs_b,
                                                TYPE *r, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    TYPE res_a;
    or_pre_inner<TYPE>(&res_a, &lhs_a[i], &lhs_b[i], &rhs_a[i], &rhs_b[i],
                       &r[i]);
    lhs_a[i] = res_a;
  }
}

extern "C" __global__ void shared_u16_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U16 *in_a, U16 *in_b,
                                                         size_t in_len,
                                                         size_t out_len) {
  u16_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
}

extern "C" __global__ void shared_u32_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U32 *in_a, U32 *in_b,
                                                         size_t in_len,
                                                         size_t out_len) {
  u32_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
}

extern "C" __global__ void split(U64 *x1_a, U64 *x1_b, U64 *x2_a, U64 *x2_b,
                                 U64 *x3_a, U64 *x3_b, size_t n, int id) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    split_inner(&x1_a[i], &x1_b[i], &x2_a[i], &x2_b[i], &x3_a[i], &x3_b[i], id);
  }
}

extern "C" __global__ void
split_for_arithmetic_xor(U32 *x1_a, U32 *x1_b, U32 *x2_a, U32 *x2_b, U32 *x3_a,
                         U32 *x3_b, U64 *in_a, U64 *in_b, size_t n, int id) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    size_t wordindex = i / 64;
    size_t bitindex = i % 64;
    U32 my_bit_a = (in_a[wordindex] >> bitindex) & 1;
    U32 my_bit_b = (in_b[wordindex] >> bitindex) & 1;
    split_for_arithmetic_xor_inner(&x1_a[i], &x1_b[i], &x2_a[i], &x2_b[i],
                                   &x3_a[i], &x3_b[i], my_bit_a, my_bit_b, id);
  }
}

extern "C" __global__ void lift_split(U16 *in_a, U16 *in_b, U32 *lifted_a,
                                      U32 *lifted_b, U64 *x1_a, U64 *x1_b,
                                      U64 *x2_a, U64 *x2_b, U64 *x3_a,
                                      U64 *x3_b, size_t chunk_size, int id) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * chunk_size) {
    lifted_a[i] = (U32)(in_a[i]);
    lifted_b[i] = (U32)(in_b[i]);
  }
  if (i < 16 * chunk_size) {
    split_inner(&x1_a[i], &x1_b[i], &x2_a[i], &x2_b[i], &x3_a[i], &x3_b[i], id);
  }
}

// Puts the results into mask_a, mask_b and x01
extern "C" __global__ void shared_lift_mul_sub(U32 *mask_a, U32 *mask_b,
                                               U16 *mask_corr_a,
                                               U16 *mask_corr_b, U16 *code_a,
                                               U16 *code_b, U32 threshold_a, int id, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lift_mul_sub(&mask_a[i], &mask_corr_a[i], &mask_corr_a[i + n], &code_a[i], threshold_a);
    lift_mul_sub(&mask_b[i], &mask_corr_b[i], &mask_corr_b[i + n], &code_b[i], threshold_a);
    switch (id) {
    case 0:
      mask_a[i] -= 1; // Transforms the <= into <
      break;
    case 1:
      mask_b[i] -= 1; // Transforms the <= into <
      break;
    default:
      break;
    }
  }
}

// Puts the results into mask_a, mask_b and code_lift_a and code_lift_b
extern "C" __global__ void
shared_finalize_lift(U32 *mask_a, U32 *mask_b, U32 *code_lift_a,
                     U32 *code_lift_b, U16 *mask_corr_a, U16 *mask_corr_b,
                     U16 *code_a, U16 *code_b, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    finalize_lift(&mask_a[i], &code_lift_a[i], &mask_corr_a[i],
                  &mask_corr_a[i + n], &code_a[i]);
    finalize_lift(&mask_b[i], &code_lift_b[i], &mask_corr_b[i],
                  &mask_corr_b[i + n], &code_b[i]);
  }
}

// Corrects to be lifted values by adding the correction values for signed
// representation
extern "C" __global__ void
shared_pre_lift_u16_u32_signed(U16 *share_a, U16 *share_b, int id, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    switch (id) {
    case 0:
      share_a[i] += 1 << 15; // Fixes the signed lifting
      break;
    case 1:
      share_b[i] += 1 << 15; // Fixes the signed lifting
      break;
    }
  }
}

// Corrects lifted values by subtracting the correction values for signed
// representation
extern "C" __global__ void
shared_finalize_lift_u16_u32_signed(U32 *share_a, U32 *share_b, U32 *corr_a,
                                    U32 *corr_b, int id, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    share_a[i] -= (U32)(corr_a[i]) << 16;
    share_a[i] -= (U32)(corr_a[i + n]) << 17;
    share_b[i] -= (U32)(corr_b[i]) << 16;
    share_b[i] -= (U32)(corr_b[i + n]) << 17;
    switch (id) {
    case 0:
      share_a[i] -= 1 << 15; // Fixes the signed lifting
      break;
    case 1:
      share_b[i] -= 1 << 15; // Fixes the signed lifting
      break;
    }
  }
}

// Puts the results into output_a and output_b
extern "C" __global__ void shared_lifted_sub(U32 *mask_a, U32 *mask_b,
                                             U32 *code_a, U32 *code_b,
                                             U32 *output_a, U32 *output_b,
                                             U32 a, int id, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lifted_sub(&mask_a[i], &code_a[i], &output_a[i], a);
    lifted_sub(&mask_b[i], &code_b[i], &output_b[i], a);
    switch (id) {
    case 0:
      output_a[i] -= 1; // Transforms the <= into <
      break;
    case 1:
      output_b[i] -= 1; // Transforms the <= into <
      break;
    default:
      break;
    }
  }
}

// Puts the results into output_a and output_b, in contrast to lifted_sub,
// this also adds the b factor to code
extern "C" __global__ void shared_prelifted_sub_ab(U32 *mask_a, U32 *mask_b,
                                                   U32 *code_a, U32 *code_b,
                                                   U32 *output_a, U32 *output_b,
                                                   U32 a, int id, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    prelifted_sub_ab(&mask_a[i], &code_a[i], &output_a[i], a);
    prelifted_sub_ab(&mask_b[i], &code_b[i], &output_b[i], a);
    switch (id) {
    case 0:
      output_a[i] -= 1; // Transforms the <= into <
      break;
    case 1:
      output_b[i] -= 1; // Transforms the <= into <
      break;
    default:
      break;
    }
  }
}

extern "C" __global__ void packed_bit_inject_party_0_a(U16 *y, U64 *in_b,
                                                       const U16 *rand_01,
                                                       const U16 *rand_02,
                                                       size_t n) {
  // in is bits packed in 64 bit integers
  // Thus, in has size n, y, rand_01, rand_02
  // have size 64 * n

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    size_t wordindex = i / 64;
    size_t bitindex = i % 64;
    U16 my_bit_b = (in_b[wordindex] >> bitindex) & 1;
    y[i] = my_bit_b * rand_01[i] - rand_02[i];
  }
}

extern "C" __global__ void
packed_bit_inject_party_0_b(U16 *out_a, U16 *out_b, const U64 *in_b,
                            const U16 *rand_01, const U16 *rand_02,
                            const U16 *y, const U16 *z, size_t n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into 16-bit
  // Thus, in has size n, out has size 64 * n
  // rand_01, rand_02, y, z have the same size
  // as out

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    size_t wordindex = i / 64;
    size_t bitindex = i % 64;
    U16 my_bit_b = (in_b[wordindex] >> bitindex) & 1;
    out_a[i] = rand_01[i] - (y[i] << 1);
    out_b[i] = -((rand_02[i] + z[i]) << 1) + my_bit_b;
  }
}

extern "C" __global__ void packed_bit_inject_party_1_a(U16 *x, const U64 *in_a,
                                                       const U64 *in_b,
                                                       const U16 *rand_01,
                                                       size_t n) {
  // in is bits packed in 64 bit integers
  // Thus, in has size n, x and rand_01 have
  // size 64 * n

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    size_t wordindex = i / 64;
    size_t bitindex = i % 64;
    U16 my_bit_a = ((in_a[wordindex] >> bitindex) & 1);
    U16 my_bit_b = ((in_b[wordindex] >> bitindex) & 1);
    U16 xor_ab = my_bit_a ^ my_bit_b;
    x[i] = xor_ab - rand_01[i];
  }
}

extern "C" __global__ void packed_bit_inject_party_1_b(U16 *out_a, U16 *out_b,
                                                       const U16 *rand_01,
                                                       const U16 *rand_12,
                                                       const U16 *x,
                                                       const U16 *y, size_t n) {
  // out, rand_01, rand_02, x, y have size 64 * n
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    out_a[i] = x[i] - (rand_12[i] << 1);
    out_b[i] = rand_01[i] - (y[i] << 1);
  }
}

extern "C" __global__ void packed_bit_inject_party_2_a(U16 *z, const U64 *in_a,
                                                       const U16 *rand_12,
                                                       const U16 *x, size_t n) {
  // in is bits packed in 64 bit integers
  // Thus, in has size n, z, rand_12 and x have
  // size 64 * n

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    size_t wordindex = i / 64;
    size_t bitindex = i % 64;
    U16 my_bit_a = (in_a[wordindex] >> bitindex) & 1;
    z[i] = my_bit_a * x[i] - rand_12[i];
  }
}

extern "C" __global__ void
packed_bit_inject_party_2_b(U16 *out_a, U16 *out_b, const U64 *in_a,
                            const U16 *rand_02, const U16 *rand_12,
                            const U16 *x, const U16 *z, size_t n) {
  // in is bits packed in 64 bit integers
  // out is each bit injected into U16-bit
  // Thus, in has size n, out has size 64 * n
  // rand_02, rand_12, x, z have the same
  // size as out

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < 64 * n) {
    size_t wordindex = i / 64;
    size_t bitindex = i % 64;
    U16 my_bit_a = (in_a[wordindex] >> bitindex) & 1;
    out_a[i] = -((rand_02[i] + z[i]) << 1) + my_bit_a;
    out_b[i] = x[i] - (rand_12[i] << 1);
  }
}

extern "C" __global__ void collapse_u64_helper(U64 *inout_a, U64 *in_b,
                                               U64 *helper_a, U64 *helper_b,
                                               U64 *r, size_t next_bitsize) {

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i == 0) {
    U64 mask = (1 << next_bitsize) - 1;
    *helper_a = *inout_a >> next_bitsize;
    *helper_b = *in_b >> next_bitsize;

    U64 res_a;
    or_pre_inner<TYPE>(&res_a, inout_a, in_b, helper_a, helper_b, r);
    *inout_a = res_a & mask;
  }
}
