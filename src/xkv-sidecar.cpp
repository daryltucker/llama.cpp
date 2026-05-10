// xkv-sidecar.cpp — loader for precomputed xKV cross-layer KV basis files (.xkv)
// See xkv-sidecar.h for format description.

#include "xkv-sidecar.h"
#include "ggml.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>

static bool xkv_read_u32(FILE * f, uint32_t & out) {
    return fread(&out, sizeof(uint32_t), 1, f) == 1;
}

xkv_sidecar * xkv_sidecar_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[xkv] cannot open sidecar: %s\n", path);
        return nullptr;
    }

    // --- header ---
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "XKVB", 4) != 0) {
        fprintf(stderr, "[xkv] bad magic in %s\n", path);
        fclose(f);
        return nullptr;
    }

    uint32_t version, n_layers, n_groups, kv_dim, group_size, rank, _reserved;
    if (!xkv_read_u32(f, version)    || version != 1 ||
        !xkv_read_u32(f, n_layers)   ||
        !xkv_read_u32(f, n_groups)   ||
        !xkv_read_u32(f, kv_dim)     ||
        !xkv_read_u32(f, group_size) ||
        !xkv_read_u32(f, rank)       ||
        !xkv_read_u32(f, _reserved)) {
        fprintf(stderr, "[xkv] truncated header in %s\n", path);
        fclose(f);
        return nullptr;
    }

    fprintf(stderr, "[xkv] loading %s  n_layers=%u  n_groups=%u  kv_dim=%u  group_size=%u  rank=%u\n",
            path, n_layers, n_groups, kv_dim, group_size, rank);

    // Allocate ggml context — one tensor per layer × 2 (K + V), size = kv_dim * rank floats each.
    const size_t tensor_size = (size_t)kv_dim * rank * sizeof(float);
    const size_t buf_size    = 2 * n_layers * (ggml_tensor_overhead() + tensor_size) + 4096;

    xkv_sidecar * sc = new xkv_sidecar();
    sc->n_layers   = (int)n_layers;
    sc->kv_dim     = (int)kv_dim;
    sc->group_size = (int)group_size;
    sc->rank       = (int)rank;
    sc->buf.resize(buf_size);

    ggml_init_params iparams = {
        /* .mem_size   = */ buf_size,
        /* .mem_buffer = */ sc->buf.data(),
        /* .no_alloc   = */ false,
    };
    sc->ctx_ggml = ggml_init(iparams);
    if (!sc->ctx_ggml) {
        fprintf(stderr, "[xkv] ggml_init failed\n");
        fclose(f);
        delete sc;
        return nullptr;
    }

    sc->basis_K.resize(n_layers, nullptr);
    sc->basis_V.resize(n_layers, nullptr);

    // Allocate all tensors up front
    for (uint32_t il = 0; il < n_layers; il++) {
        sc->basis_K[il] = ggml_new_tensor_2d(sc->ctx_ggml, GGML_TYPE_F32, kv_dim, rank);
        sc->basis_V[il] = ggml_new_tensor_2d(sc->ctx_ggml, GGML_TYPE_F32, kv_dim, rank);
    }

    // Read groups and populate tensors
    std::vector<float> tmp(group_size * kv_dim * rank);
    for (uint32_t g = 0; g < n_groups; g++) {
        uint32_t g_start, g_end, pad0, pad1;
        if (!xkv_read_u32(f, g_start) || !xkv_read_u32(f, g_end) ||
            !xkv_read_u32(f, pad0)    || !xkv_read_u32(f, pad1)) {
            fprintf(stderr, "[xkv] truncated group header %u\n", g);
            fclose(f);
            xkv_sidecar_free(sc);
            return nullptr;
        }
        uint32_t G = g_end - g_start;

        // basis_K for this group: G layers × kv_dim * rank floats (ggml col-major layout)
        size_t group_floats = (size_t)G * kv_dim * rank;
        if (group_floats > tmp.size()) tmp.resize(group_floats);

        if (fread(tmp.data(), sizeof(float), group_floats, f) != group_floats) {
            fprintf(stderr, "[xkv] truncated basis_K group %u\n", g);
            fclose(f);
            xkv_sidecar_free(sc);
            return nullptr;
        }
        for (uint32_t il_local = 0; il_local < G; il_local++) {
            uint32_t il = g_start + il_local;
            memcpy(sc->basis_K[il]->data,
                   tmp.data() + (size_t)il_local * kv_dim * rank,
                   tensor_size);
        }

        // basis_V for this group
        if (fread(tmp.data(), sizeof(float), group_floats, f) != group_floats) {
            fprintf(stderr, "[xkv] truncated basis_V group %u\n", g);
            fclose(f);
            xkv_sidecar_free(sc);
            return nullptr;
        }
        for (uint32_t il_local = 0; il_local < G; il_local++) {
            uint32_t il = g_start + il_local;
            memcpy(sc->basis_V[il]->data,
                   tmp.data() + (size_t)il_local * kv_dim * rank,
                   tensor_size);
        }
    }

    fclose(f);
    fprintf(stderr, "[xkv] loaded %u layer bases (K + V), rank=%u, kv_dim=%u\n",
            n_layers, rank, kv_dim);
    return sc;
}

void xkv_sidecar_free(xkv_sidecar * sc) {
    if (!sc) return;
    if (sc->ctx_ggml) ggml_free(sc->ctx_ggml);
    delete sc;
}
