#define U16 unsigned short
#define U32 unsigned int
#define U64 unsigned long long
#define TYPE U64

#define MATCH_THRESHOLD_RATIO 0.34
#define B_BITS 20
#define B (1 << B_BITS)
#define A ((U64)((1. - 2. * MATCH_THRESHOLD_RATIO) * (double)B))
#define P ((1ULL << 16) - 17)
#define P2K (P << B_BITS)

////////////////////////////////////////////////////////////////////////////////
// Basic Blocks (not parallelized)
////////////////////////////////////////////////////////////////////////////////

template <typename T> __global__ void not_inplace_inner(T *lhs) {
  *lhs = ~(*lhs);
}

template <typename T> __global__ void not_inner(T *res, T *lhs) {
  *res = ~(*lhs);
}

template <typename T> __global__ void xor_inner(T *res, T *lhs, T *rhs) {
  *res = *lhs ^ *rhs;
}

template <typename T> __global__ void xor_assign_inner(T *lhs, T *rhs) {
  *lhs ^= *rhs;
}

template <typename T>
__global__ void or_pre_inner(T *res_a, T *lhs_a, T *lhs_b, T *rhs_a, T *rhs_b,
                             T *r) {
  and_pre_inner<T>(res_a, lhs_a, lhs_b, rhs_a, rhs_b, r); // AND with randomness
  *res_a ^= *lhs_a ^ *lhs_b; // XOR with the original values
}

// Computes the local part of the multiplication (including randomness)
template <typename T>
__global__ void and_pre_inner(T *res_a, T *lhs_a, T *lhs_b, T *rhs_a, T *rhs_b,
                              T *r) {
  *res_a = (*lhs_a & *rhs_a) ^ (*lhs_b & *rhs_a) ^ (*lhs_a & *rhs_b) ^ *r;
}

__global__ void mul_lift_b(U64 *res, U16 *input) {
  *res = (U64)(*input) << B_BITS;
}

__global__ void u64_from_u16s(U64 *res, U16 *a, U16 *b, U16 *c, U16 *d) {
  *res = (U64)(*a) | ((U64)(*b) << 16) | ((U64)(*c) << 32) | ((U64)(*d) << 48);
}

__global__ void u64_from_u32s(U64 *res, U32 *a, U32 *b) {
  *res = (U64)(*a) | ((U64)(*b) << 32);
}

__global__ void transpose16x64(U64 *out, U16 *in) {
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

__global__ void transpose32x64(U64 *out, U32 *in) {
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

// TODO: do this transpose in shared memory which should be much faster...
__global__ void transpose64x64(U64 *inout) {
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
__global__ void u16_transpose_pack_u64(U64 *out_a, U64 *out_b, U16 *in_a,
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
__global__ void u32_transpose_pack_u64(U64 *out_a, U64 *out_b, U32 *in_a,
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

// Performs the transpose for a and b in parallel
// Overwrites the input!
__global__ void u64_transpose_pack_u64(U64 *out_a, U64 *out_b, U64 *in_a,
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

__global__ void lift_mul_sub(U64 *mask, U16 *code) {
  U64 a;
  mul_lift_b(&a, code);
  *mask *= A;
  *mask += P2K;
  *mask -= a;
  *mask %= P2K;
}

// Puts the results into x_a, x_b and x01
__global__ void split_msb_fp(U64 *x_a, U64 *x_b, U64 *x01, U64 *r, int id) {
  // I don't add the bitmod to the randomness, since the bits gets removed later
  // anyways

  switch (id) {
  case 0:
    *x01 = *r;
    *x_a = 0;
    break;
  case 1:
    *x01 = *r ^ ((*x_a + *x_b) % P2K);
    *x_a = 0;
    *x_b = 0;
    break;
  case 2:
    *x01 = *r;
    *x_b = 0;
    break;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Test kernels
////////////////////////////////////////////////////////////////////////////////

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

extern "C" __global__ void shared_or_pre(TYPE *res_a, TYPE *lhs_a, TYPE *lhs_b,
                                         TYPE *rhs_a, TYPE *rhs_b, TYPE *r,
                                         int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    or_pre_inner<TYPE>(&res_a[i], &lhs_a[i], &lhs_b[i], &rhs_a[i], &rhs_b[i],
                       &r[i]);
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

extern "C" __global__ void shared_u64_transpose_pack_u64(U64 *out_a, U64 *out_b,
                                                         U64 *in_a, U64 *in_b,
                                                         int in_len,
                                                         int out_len) {
  u64_transpose_pack_u64(out_a, out_b, in_a, in_b, in_len, out_len);
}

// Puts the results into mask_a, mask_b and x01
extern "C" __global__ void shared_lift_mul_sub_split(U64 *x01, U64 *mask_a,
                                                     U64 *mask_b, U16 *code_a,
                                                     U16 *code_b, U64 *r, int n,
                                                     int id) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    lift_mul_sub(&mask_a[i], &code_a[i]);
    lift_mul_sub(&mask_b[i], &code_b[i]);

    split_msb_fp(&mask_a[i], &mask_b[i], &x01[i], &r[i], id);
  }
}

extern "C" __global__ void shared_split1(U16 *inp_a, U16 *inp_b, U64 *xa_a,
                                         U64 *xa_b, U32 *xp_a, U32 *xp_b,
                                         U32 *xpp_a, U32 *xpp_b, int n,
                                         int id) {
  assert(n % 64 == 0);

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    xa_a[i] = (U64)(inp_a[i]);
    xa_b[i] = (U64)(inp_b[i]);

    switch (id) {
    case 0:
      U64 subbed_p = ((U64)(inp_a[i]) + P2K - P) % P2K;
      U64 subbed_pp = ((U64)(inp_a[i]) + P2K - 2 * P) % P2K;
      xp_a[i] = (U32)(subbed_p);
      xpp_a[i] = (U32)(subbed_pp);
      xp_b[i] = (U32)(inp_b[i]);
      xpp_b[i] = (U32)(inp_b[i]);
      break;
    case 1:
      subbed_p = ((U64)(inp_b[i]) + P2K - P) % P2K;
      subbed_pp = ((U64)(inp_b[i]) + P2K - 2 * P) % P2K;
      xp_a[i] = (U32)(inp_a[i]);
      xpp_a[i] = (U32)(inp_a[i]);
      xp_b[i] = (U32)(subbed_p);
      xpp_b[i] = (U32)(subbed_pp);
      break;
    case 2:
      xp_a[i] = (U32)(inp_a[i]);
      xpp_a[i] = (U32)(inp_a[i]);
      xp_b[i] = (U32)(inp_b[i]);
      xpp_b[i] = (U32)(inp_b[i]);
      break;
    }
  }
}

extern "C" __global__ void shared_split2(U64 *xp_a, U64 *xp_b, U64 *xp1_a,
                                         U64 *xp1_b, U64 *xp2_a, U64 *xp2_b,
                                         U64 *xp3_a, U64 *xp3_b, int n,
                                         int id) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    switch (id) {
    case 0:
      xp1_a[i] = xp_a[i];
      xp3_b[i] = xp_b[i];
      break;
    case 1:
      xp2_a[i] = xp_a[i];
      xp1_b[i] = xp_b[i];
      break;
    case 2:
      xp3_a[i] = xp_a[i];
      xp2_b[i] = xp_b[i];
      break;
    }
  }
}
