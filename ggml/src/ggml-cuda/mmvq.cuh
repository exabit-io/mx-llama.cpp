#include "common.cuh"

// Max. batch size for which to use MMVQ kernels (dense MUL_MAT). gfx906 ran 16 here through v1.0.0 (2026-09-07, measured
// on stock b10288). On v1.1.0 MMQ is ahead at every width timed from 10 to 16 columns, by 1.0 % rising to 1.8 % in decode
// (four dies, Q8_0, 2026-11-27), so upstream's 8 again. The 9..16-column kernels stay in mmvq.cu: changing this one
// number back is all a new comparison needs.
#define MMVQ_MAX_BATCH_SIZE 8
#define MMVQ_MMID_MAX_BATCH_SIZE 8 // Max. batch size for the MUL_MAT_ID (MoE) MMVQ path; unchanged.

bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11);

// Returns the maximum batch size for which MMVQ should be used for MUL_MAT_ID,
// based on the quantization type and GPU architecture (compute capability).
int get_mmvq_mmid_max_batch(ggml_type type, int cc);

void ggml_cuda_mul_mat_vec_q(ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst, const ggml_cuda_mm_fusion_args_host * fusion = nullptr);

void ggml_cuda_op_mul_mat_vec_q(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
