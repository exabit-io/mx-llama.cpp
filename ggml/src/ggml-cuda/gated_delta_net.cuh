#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// Device-side view of ggml_backend_cuda_context::gdn_prefuse_info (S3 step 3): work folded into the GDN kernel.
struct ggml_cuda_gdn_prefuse_dev {
    bool          norm_q     = false; // q = q / max(||q||, eps_q)   (L2_NORM folded)
    bool          norm_k     = false;
    bool          sig_b      = false; // beta = sigmoid(beta_raw)
    bool          gate_alpha = false; // g = softplus(alpha + dt_bias[h]) * ssm_a[h]
    float         eps_q      = 0.0f;
    float         eps_k      = 0.0f;
    const float * dt_bias    = nullptr;
    const float * ssm_a      = nullptr;
};
