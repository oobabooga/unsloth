// Plain RDNA3 / RDNA3.5 (gfx11, wave32) WMMA GEMM kernels, one KIND per shared object:
//   0 f16   v_wmma_f32_16x16x16_f16    A,B fp16           -> fp32 acc -> bf16 out
//   1 bf16  v_wmma_f32_16x16x16_bf16   A,B bf16           -> fp32 acc -> bf16 out
//   2 iu8   v_wmma_i32_16x16x16_iu8    A,B int8 (signed)  -> int32 acc -> * sa[m] * sw[n] -> bf16
//   3 iu4   v_wmma_i32_16x16x16_iu4    A,B int4 packed    -> int32 acc -> * sa[m] * sw[n] -> bf16
//   4 w4a16 bf16 WMMA; B = int4 weights (packed, signed) * group scale, dequantised in registers
// Layout: A is [M, K] row-major, W is [N, K] row-major (nn.Linear weight), C = A @ W^T is [M, N] bf16.
// int4 packing: element k of a row sits in byte k/2, low nibble for even k (two's complement).
// No LDS: every lane loads its own fragment row straight from global memory (lanes 16..31 repeat 0..15),
// the wave keeps a TM x TN grid of 16x16 accumulators in registers. Simple on purpose.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef KIND
#define KIND 2
#endif

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef _Float16 v16h __attribute__((ext_vector_type(16)));

__device__ __forceinline__ unsigned short f2bf(float f) {
  unsigned int u = __float_as_uint(f);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}

#if KIND == 0
typedef v16h AT; typedef v16h BT; typedef v8f CT;
__device__ __forceinline__ CT mma(AT a, BT b, CT c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
__device__ __forceinline__ AT loadA(const void* A, int row, int k0, int K) {
  return *(const v16h*)((const _Float16*)A + (size_t)row * K + k0);
}
__device__ __forceinline__ BT loadB(const void* W, int col, int k0, int K, const float* sw, int G) {
  return *(const v16h*)((const _Float16*)W + (size_t)col * K + k0);
}
#elif KIND == 1 || KIND == 4
typedef v16s AT; typedef v16s BT; typedef v8f CT;
__device__ __forceinline__ CT mma(AT a, BT b, CT c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c); }
__device__ __forceinline__ AT loadA(const void* A, int row, int k0, int K) {
  return *(const v16s*)((const short*)A + (size_t)row * K + k0);
}
#if KIND == 1
__device__ __forceinline__ BT loadB(const void* W, int col, int k0, int K, const float* sw, int G) {
  return *(const v16s*)((const short*)W + (size_t)col * K + k0);
}
#else
__device__ __forceinline__ BT loadB(const void* W, int col, int k0, int K, const float* sw, int G) {
  v2i p = *(const v2i*)((const unsigned char*)W + (size_t)col * (K / 2) + k0 / 2);
  const float s = sw[(size_t)col * (K / G) + k0 / G];
  BT b;
#pragma unroll
  for (int e = 0; e < 16; ++e) {
    int q = (p[e >> 3] << (28 - 4 * (e & 7))) >> 28;  // sign-extended nibble
    b[e] = (short)f2bf((float)q * s);
  }
  return b;
}
#endif
#elif KIND == 2
typedef v4i AT; typedef v4i BT; typedef v8i CT;
__device__ __forceinline__ CT mma(AT a, BT b, CT c) { return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, false); }
__device__ __forceinline__ AT loadA(const void* A, int row, int k0, int K) {
  return *(const v4i*)((const int8_t*)A + (size_t)row * K + k0);
}
__device__ __forceinline__ BT loadB(const void* W, int col, int k0, int K, const float* sw, int G) {
  return *(const v4i*)((const int8_t*)W + (size_t)col * K + k0);
}
#elif KIND == 3
typedef v2i AT; typedef v2i BT; typedef v8i CT;
__device__ __forceinline__ CT mma(AT a, BT b, CT c) { return __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, c, false); }
__device__ __forceinline__ AT loadA(const void* A, int row, int k0, int K) {
  return *(const v2i*)((const unsigned char*)A + (size_t)row * (K / 2) + k0 / 2);
}
__device__ __forceinline__ BT loadB(const void* W, int col, int k0, int K, const float* sw, int G) {
  return *(const v2i*)((const unsigned char*)W + (size_t)col * (K / 2) + k0 / 2);
}
#endif

#define SCALED (KIND == 2 || KIND == 3)

template <int TM, int TN>
__global__ __launch_bounds__(128) void gemm_kernel(const void* __restrict__ A, const void* __restrict__ W,
                                                   unsigned short* __restrict__ C, const float* __restrict__ sa,
                                                   const float* __restrict__ sw, int M, int N, int K, int G) {
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int r = lane & 15, half = lane >> 4;
  const int m0 = blockIdx.y * (2 * TM * 16) + (wave >> 1) * TM * 16;
  const int n0 = blockIdx.x * (2 * TN * 16) + (wave & 1) * TN * 16;
  CT acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = (CT){};
#pragma unroll 2
  for (int k0 = 0; k0 < K; k0 += 16) {
    AT a[TM];
    BT b[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = loadA(A, m0 + i * 16 + r, k0, K);
#pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = loadB(W, n0 + j * 16 + r, k0, K, sw, G);
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] = mma(a[i], b[j], acc[i][j]);
  }
#pragma unroll
  for (int i = 0; i < TM; ++i) {
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int col = n0 + j * 16 + r;
#if SCALED
      const float scol = sw[col];
#endif
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int row = m0 + i * 16 + 2 * e + half;
#if SCALED
        float v = (float)acc[i][j][e] * sa[row] * scol;
#else
        float v = acc[i][j][e];
#endif
        C[(size_t)row * N + col] = f2bf(v);
      }
    }
  }
}

// Peak issue rate: 8 independent accumulator chains per wave on register-resident fragments.
__global__ __launch_bounds__(128) void peak_kernel(int iters, float* out) {
  AT a, b;
#if KIND == 0
  for (int e = 0; e < 16; ++e) { a[e] = (_Float16)((threadIdx.x + e) & 7); b[e] = (_Float16)((threadIdx.x * 3 + e) & 7); }
#elif KIND == 1 || KIND == 4
  for (int e = 0; e < 16; ++e) { a[e] = (short)(0x3f80 + ((threadIdx.x + e) & 7)); b[e] = (short)(0x3f80 + ((threadIdx.x * 3 + e) & 7)); }
#elif KIND == 2
  for (int e = 0; e < 4; ++e) { a[e] = 0x01020304 * ((threadIdx.x + e) & 3); b[e] = 0x01010101 * ((threadIdx.x * 3 + e) & 3); }
#else
  for (int e = 0; e < 2; ++e) { a[e] = 0x11111111 * ((threadIdx.x + e) & 3); b[e] = 0x01010101 * ((threadIdx.x * 3 + e) & 3); }
#endif
  CT acc[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) acc[j] = (CT){};
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = mma(a, b, acc[j]);
  }
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j)
#pragma unroll
    for (int e = 0; e < 8; ++e) s += (float)acc[j][e];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// One 16x16x16 product by one wave; raw accumulators as out[lane * 8 + e] (fp32), for layout discovery.
__global__ void tile_kernel(const void* A, const void* W, float* out) {
#if KIND == 4
  return;
#else
  const int lane = threadIdx.x & 31, r = lane & 15;
  AT a = loadA(A, r, 0, 16);
  BT b = loadB(W, r, 0, 16, nullptr, 16);
  CT c = (CT){};
  c = mma(a, b, c);
  for (int e = 0; e < 8; ++e) out[lane * 8 + e] = (float)c[e];
#endif
}

template <int TM, int TN>
static int launch(const void* A, const void* W, void* C, const float* sa, const float* sw, int M, int N, int K, int G,
                  hipStream_t s) {
  const int BM = 2 * TM * 16, BN = 2 * TN * 16;
  if (M % BM || N % BN || K % 32) return -2;
  dim3 grid(N / BN, M / BM);
  hipLaunchKernelGGL((gemm_kernel<TM, TN>), grid, dim3(128), 0, s, A, W, (unsigned short*)C, sa, sw, M, N, K, G);
  return (int)hipGetLastError();
}

extern "C" {
int wm_kind() { return KIND; }
int wm_ncfg() { return 4; }
int wm_cfg(int cfg, int* tm, int* tn) {
  static const int t[4][2] = {{2, 2}, {2, 4}, {4, 2}, {4, 4}};
  if (cfg < 0 || cfg > 3) return -1;
  *tm = t[cfg][0];
  *tn = t[cfg][1];
  return 0;
}
int wm_gemm(int cfg, const void* A, const void* W, void* C, const float* sa, const float* sw, int M, int N, int K,
            int G, void* stream) {
  hipStream_t s = (hipStream_t)stream;
  switch (cfg) {
    case 0: return launch<2, 2>(A, W, C, sa, sw, M, N, K, G, s);
    case 1: return launch<2, 4>(A, W, C, sa, sw, M, N, K, G, s);
    case 2: return launch<4, 2>(A, W, C, sa, sw, M, N, K, G, s);
    case 3: return launch<4, 4>(A, W, C, sa, sw, M, N, K, G, s);
  }
  return -1;
}
int wm_peak(int blocks, int iters, float* out, void* stream) {
  hipLaunchKernelGGL(peak_kernel, dim3(blocks), dim3(128), 0, (hipStream_t)stream, iters, out);
  return (int)hipGetLastError();
}
int wm_tile(const void* A, const void* W, float* out, void* stream) {
  hipLaunchKernelGGL(tile_kernel, dim3(1), dim3(32), 0, (hipStream_t)stream, A, W, out);
  return (int)hipGetLastError();
}
}
