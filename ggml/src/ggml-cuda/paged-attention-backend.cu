/**
 * GGML CUDA Backend for PagedAttention
 *
 * This file provides the CUDA backend implementation for the GGML_OP_PAGED_ATTENTION operation.
 * It bridges GGML's operation framework with the PagedAttention CUDA kernels.
 *
 * NOTE: PagedAttention is currently experimental and only supported on CUDA.
 * MUSA support is disabled due to compiler compatibility issues.
 */

// PagedAttention is not yet supported on MUSA
#ifndef GGML_USE_MUSA

#include "common.cuh"
#include "paged-attention.cuh"

// Extract parameters from GGML tensor
static void ggml_cuda_op_paged_attention_get_params(
    const ggml_tensor * dst,
    float * scale,
    int32_t * block_size) {

    const float * params = (const float *)dst->op_params;
    *scale = params[0];
    *block_size = (int32_t)params[1];
}

// Main CUDA backend function for PagedAttention
void ggml_cuda_op_paged_attention(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {

    const ggml_tensor * q            = dst->src[0];  // query
    const ggml_tensor * k_cache      = dst->src[1];  // key cache (paged)
    const ggml_tensor * v_cache      = dst->src[2];  // value cache (paged)
    const ggml_tensor * block_tables = dst->src[3];  // block tables
    const ggml_tensor * seq_lens     = dst->src[4];  // sequence lengths

    // Extract parameters
    float scale;
    int32_t block_size;
    ggml_cuda_op_paged_attention_get_params(dst, &scale, &block_size);

    // Get tensor dimensions
    const int64_t head_size = q->ne[0];
    const int64_t n_heads = q->ne[1];
    const int64_t n_tokens = q->ne[2];  // TODO: use for validation
    const int64_t n_seqs = q->ne[3];

    const int64_t n_kv_heads = k_cache->ne[2];
    const int64_t num_blocks = k_cache->ne[0];  // TODO: use for validation

    const int64_t max_blocks_per_seq = block_tables->ne[0];

    // Suppress unused variable warnings
    GGML_UNUSED(n_tokens);
    GGML_UNUSED(num_blocks);

    // Get pointers
    void * out_ptr = dst->data;
    const void * q_ptr = q->data;
    const void * k_cache_ptr = k_cache->data;
    const void * v_cache_ptr = v_cache->data;
    const int32_t * block_tables_ptr = (const int32_t *)block_tables->data;
    const int32_t * seq_lens_ptr = (const int32_t *)seq_lens->data;

    // Calculate max sequence length (needed to decide V1 vs V2)
    int max_seq_len = 0;
    for (int i = 0; i < n_seqs; i++) {
        if (seq_lens_ptr[i] > max_seq_len) {
            max_seq_len = seq_lens_ptr[i];
        }
    }

    // Get CUDA stream
    cudaStream_t stream = ctx.stream();

    // Decide whether to use V1 or V2
    const bool use_v1 = ggml_cuda_paged_attention::should_use_v1(
        max_seq_len, n_seqs, n_heads);

    // Launch appropriate kernel
    if (use_v1) {
        ggml_cuda_paged_attention::paged_attention_v1_launcher(
            out_ptr,
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            n_seqs,
            n_heads,
            n_kv_heads,
            head_size,
            block_size,
            max_blocks_per_seq,
            block_tables_ptr,
            seq_lens_ptr,
            max_seq_len,
            scale,
            nullptr,  // alibi_slopes (TODO: add support if needed)
            q->type,
            k_cache->type,
            stream);
    } else {
        ggml_cuda_paged_attention::paged_attention_v2_launcher(
            out_ptr,
            q_ptr,
            k_cache_ptr,
            v_cache_ptr,
            n_seqs,
            n_heads,
            n_kv_heads,
            head_size,
            block_size,
            max_blocks_per_seq,
            block_tables_ptr,
            seq_lens_ptr,
            max_seq_len,
            scale,
            nullptr,  // alibi_slopes
            q->type,
            k_cache->type,
            stream);
    }

    // Check for errors
    CUDA_CHECK(cudaGetLastError());
}

// Check if PagedAttention is supported for given configuration
bool ggml_cuda_can_paged_attention(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k_cache = dst->src[1];

    // Check data types
    if (q->type != GGML_TYPE_F16 && q->type != GGML_TYPE_F32) {
        return false;
    }

    if (k_cache->type != GGML_TYPE_F16 && k_cache->type != GGML_TYPE_F32) {
        return false;
    }

    // Check head size is supported
    const int64_t head_size = q->ne[0];
    const int supported_head_sizes[] = {32, 64, 80, 96, 112, 120, 128, 192, 256};
    bool head_size_supported = false;

    for (int hs : supported_head_sizes) {
        if (head_size == hs) {
            head_size_supported = true;
            break;
        }
    }

    if (!head_size_supported) {
        return false;
    }

    // Extract block size and check it's supported
    float scale;
    int32_t block_size;
    ggml_cuda_op_paged_attention_get_params(dst, &scale, &block_size);

    if (block_size != 8 && block_size != 16 && block_size != 32) {
        return false;
    }

    return true;
}

#else // GGML_USE_MUSA

// Stub implementations for MUSA (PagedAttention not yet supported)
#include "common.cuh"

void ggml_cuda_op_paged_attention(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("PagedAttention is not yet supported on MUSA");
}

bool ggml_cuda_supports_paged_attention(const ggml_tensor * dst) {
    GGML_UNUSED(dst);
    return false;
}

#endif // GGML_USE_MUSA
