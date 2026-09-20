#include "gated_delta_net.cuh"
#include "gated_delta_net_chunk.cuh"
#include "ggml-cuda/common.cuh"

// Sum of squares of the head vector this warp holds (rows_per_lane values per lane), reproducing the standalone
// l2_norm_f32<32> bit for bit where the shapes allow it: that kernel runs 32 threads per row, thread t accumulating
// x[t], x[t+32], x[t+64], x[t+96] in that order (fused multiply-adds), then a 32-wide xor tree. With a 64-lane warp holding
// x[lane] and x[lane+64], lane t < 32 fetches x[t+32] and x[t+96] from lane t+32 and repeats that order; lanes 32-63 then
// take the result from lane t-32. Other shapes fall back to a plain reduction (same values up to rounding order).
template <int warp_size, int rows_per_lane>
static __device__ __forceinline__ float gdn_l2_sumsq(const float * v, const int lane) {
    if constexpr (warp_size == 64 && rows_per_lane == 2) {
        const float x32 = __shfl_xor_sync(0xffffffff, v[0], 32, 64);
        const float x96 = __shfl_xor_sync(0xffffffff, v[1], 32, 64);
        float tmp = fmaf(v[0], v[0], 0.0f);
        tmp = fmaf(x32, x32, tmp);
        tmp = fmaf(v[1], v[1], tmp);
        tmp = fmaf(x96, x96, tmp);
        tmp = warp_reduce_sum<32>(tmp);
        return __shfl_sync(0xffffffff, tmp, lane & 31, 64);
    } else {
        float ss = 0.0f;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            ss += v[r] * v[r];
        }
        return warp_reduce_sum<warp_size>(ss);
    }
}

template <int S_v, bool KDA, bool keep_rs_t>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     float *       state,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale,
                                     int64_t       state_slot_stride,
                                     int           K,
                                     const ggml_cuda_gdn_prefuse_dev pf) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only: [S_v, S_v, H, n_seqs] — seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) — same per-(seq,head) offset as before.
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = pf.sig_b ? 1.0f / (1.0f + expf(-*beta_t)) : *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }
        // folded L2 norms (same arithmetic as l2_norm_f32: x * rsqrt(max(sum x^2, eps^2)) over the head vector, which
        // this warp holds in full across its lanes)
        if (pf.norm_k) {
            const float ssk = gdn_l2_sumsq<warp_size, rows_per_lane>(k_reg, lane);
            const float sc  = pf.norm_kind == 0 ? rsqrtf(fmaxf(ssk, pf.eps_k * pf.eps_k))
                                                : rsqrtf(ssk * (1.0f / S_v) + pf.eps_k) * pf.scale_k;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                k_reg[r] *= sc;
            }
        }
        if (pf.norm_q) {
            const float ssq = gdn_l2_sumsq<warp_size, rows_per_lane>(q_reg, lane);
            const float sc  = pf.norm_kind == 0 ? rsqrtf(fmaxf(ssq, pf.eps_q * pf.eps_q))
                                                : rsqrtf(ssq * (1.0f / S_v) + pf.eps_q) * pf.scale_q;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                q_reg[r] *= sc;
            }
        }

        if constexpr (!KDA) {
            float gv = *g_t;
            if (pf.gate_alpha) {
                // folded ADD(dt_bias) -> softplus -> MUL(ssm_a), same formulas as the unary kernels
                gv = gv + pf.dt_bias[h_idx];
                gv = (gv > 20.0f) ? gv : logf(1.0f + expf(gv));
                gv = gv * pf.ssm_a[h_idx];
            }
            const float g_val = expf(gv);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        attn_data += S_v * H;

        if constexpr (keep_rs_t) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const int i = r * warp_size + lane;
                    curr_state[col * S_v + i] = s_shard[r];
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i          = r * warp_size + lane;
            state[col * S_v + i] = s_shard[r];
        }
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * state_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t state_slot_stride, int K, const ggml_cuda_gdn_prefuse_dev & pf, cudaStream_t stream) {
    const int CS = KDA ? 16 : 64;

    // The chunked kernel emits per-token snapshots, so retained-state prefill no longer needs serial arithmetic.
    // The opt-out is for diagnosis and does not change ring-off calls.
    static const bool chunk_snap_off = [] {
        const char * value = getenv("LLAMA_GDN_NO_CHUNK_SNAPSHOTS");
        return value != nullptr && atoi(value) != 0;
    }();
    const bool chunk_eligible = n_tokens >= 2 * CS && S_v <= 128;

    static bool chunk_snap_announced = false;
    if constexpr (keep_rs_t) {
        if (chunk_eligible && !chunk_snap_announced) {
            chunk_snap_announced = true;
            GGML_LOG_WARN("%s: gdn-chunk-snapshots: keep_rs chunked path %s\n",
                    __func__, chunk_snap_off ? "SUPPRESSED by LLAMA_GDN_NO_CHUNK_SNAPSHOTS" :
                                              "TAKEN (default)");
        }
    }

    if (chunk_eligible && (!keep_rs_t || !chunk_snap_off)) {
        launch_gated_delta_net_chunk<KDA, keep_rs_t>(
            q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3,
            sv1, sv2, sv3, sb1, sb2, sb3,
            neqk1, rq3, scale, K, stream);
        return;
    }

    const int device = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    const int cc = ggml_cuda_info().devices[device].cc;
    const int num_warps = cc == GGML_CUDA_CC_VEGA20 && n_tokens == 1 ? 2 : 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<16, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, pf);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_cuda<32, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, pf);
            break;
        case 64: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<64, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, pf);
            break;
        }
        case 128: {
            ggml_cuda_kernel_launch(gated_delta_net_cuda<128, KDA, keep_rs_t>, launch_params,
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, pf);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

static void ggml_cuda_op_gated_delta_net_impl(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_cuda_gated_delta_net_fused_cache * cache) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    // S3 step 3: producers folded into this kernel (recorded by ggml_cuda_gdn_prefuse_producer while walking the graph)
    ggml_cuda_gdn_prefuse_dev pf;
    const ggml_tensor * q_eff = src_q;
    const ggml_tensor * k_eff = src_k;
    const ggml_tensor * g_eff = src_g;
    const ggml_tensor * b_eff = src_beta;
    {
        const auto it = ctx.gdn_prefuse.find(dst);
        if (it != ctx.gdn_prefuse.end()) {
            const ggml_backend_cuda_context::gdn_prefuse_info & info = it->second;
            GGML_ASSERT((info.q_raw == nullptr) == (info.k_raw == nullptr));
            if (info.q_raw) {
                q_eff = info.q_raw; k_eff = info.k_raw;
                pf.norm_q = true; pf.norm_k = true; pf.eps_q = info.eps_q; pf.eps_k = info.eps_k;
                pf.norm_kind = info.norm_kind; pf.scale_q = info.scale_q; pf.scale_k = info.scale_k;
            }
            if (info.beta_raw) {
                b_eff = info.beta_raw; pf.sig_b = true;
            }
            if (info.alpha_raw) {
                GGML_ASSERT(!kda);
                g_eff = info.alpha_raw; pf.gate_alpha = true;
                pf.dt_bias = (const float *) info.dt_bias->data;
                pf.ssm_a   = (const float *) info.ssm_a->data;
            }
        }
    }
    const float * q_d = (const float *) q_eff->data;
    const float * k_d = (const float *) k_eff->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) g_eff->data;
    const float * b_d = (const float *) b_eff->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(q_eff));
    GGML_ASSERT(ggml_is_contiguous_rows(k_eff));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(q_eff, k_eff));
    GGML_ASSERT(ggml_is_contiguous(g_eff) && ggml_is_contiguous(b_eff));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = q_eff->nb[1] / sizeof(float);
    const int64_t sq2 = q_eff->nb[2] / sizeof(float);
    const int64_t sq3 = q_eff->nb[3] / sizeof(float);
    GGML_UNUSED(nbq1); GGML_UNUSED(nbq2); GGML_UNUSED(nbq3);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> gdn_out tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, pf, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, pf, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, pf, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, pf, stream);
        }
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, nullptr);
}

void ggml_cuda_op_gated_delta_net_fused_cache(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_cuda_gated_delta_net_fused_cache cache) {
    ggml_cuda_op_gated_delta_net_impl(ctx, dst, &cache);
}
