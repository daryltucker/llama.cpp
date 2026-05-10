// xkv-sidecar.h — xKV cross-layer KV basis sidecar for inference-time rank-r projection
// arxiv 2503.18893: adjacent transformer layers share a low-rank KV subspace.
// This patch applies per-layer rank-r projection at attention time using a precomputed
// SVD basis, reducing effective KV entropy without changing the KV cache layout.
// Basis is produced offline by tools/xkv_precompute_basis.py → saved as .xkv binary.

#pragma once

#include "ggml.h"
#include <string>
#include <vector>

// Binary format produced by save_xkv() in xkv_precompute_basis.py
// Header (32 bytes, all uint32 little-endian):
//   magic[4]     = "XKVB"
//   version      = 1
//   n_layers, n_groups, kv_dim, group_size, rank, _reserved
// Per group (n_groups times):
//   g_start, g_end, _pad[2]
//   float32[G * kv_dim * rank]  basis_K per layer in ggml column-major layout
//   float32[G * kv_dim * rank]  basis_V per layer in ggml column-major layout
//
// ggml column-major layout for a [kv_dim, rank] tensor:
//   element (d, r) at float index  d + r*kv_dim
//   produced in Python as: slice.T.astype(np.float32).ravel('C')
//   where slice = basis_K[il_local*kv_dim:(il_local+1)*kv_dim, :]  shape (kv_dim, rank)

struct xkv_sidecar {
    int n_layers;
    int kv_dim;    // per-layer kv dimension = num_kv_heads * head_dim
    int group_size;
    int rank;

    // Per-layer basis tensors; shape [kv_dim, rank] in ggml convention.
    // Projection: latent = basis^T @ k_flat  (ggml: mul_mat(basis, k_flat))
    //             k_approx = basis @ latent   (ggml: mul_mat(ggml_transpose(basis), latent))
    std::vector<ggml_tensor *> basis_K;  // [n_layers]
    std::vector<ggml_tensor *> basis_V;  // [n_layers]

    // Owns all tensor memory (allocated in ggml_init'd context).
    ggml_context * ctx_ggml = nullptr;
    std::vector<uint8_t> buf;

    ggml_tensor * get_K(int il) const { return (il >= 0 && il < n_layers) ? basis_K[il] : nullptr; }
    ggml_tensor * get_V(int il) const { return (il >= 0 && il < n_layers) ? basis_V[il] : nullptr; }
};

xkv_sidecar * xkv_sidecar_load(const char * path);
void          xkv_sidecar_free(xkv_sidecar * sc);
