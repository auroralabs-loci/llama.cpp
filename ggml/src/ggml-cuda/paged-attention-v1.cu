// PagedAttention is not yet supported on MUSA
#ifndef GGML_USE_MUSA

#include "paged-attention.cuh"
#include "common.cuh"

namespace ggml_cuda_paged_attention {

//
// Main PagedAttention V1 Kernel
//
// This kernel computes attention for one sequence and one head per thread block.
// It reads K/V from paged blocks based on the block table.
//

template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const cache_t* __restrict__ k_cache,
    const cache_t* __restrict__ v_cache,
    const int num_kv_heads,
    const float scale,
    const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens,
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride) {

    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int num_heads = gridDim.x;
    const int thread_idx = threadIdx.x;

    const int seq_len = seq_lens[seq_idx];
    if (seq_len == 0) return;

    const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    // Shared memory for logits and reduction
    extern __shared__ char shared_mem[];
    float* logits = reinterpret_cast<float*>(shared_mem);
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
    __shared__ float red_smem[2 * NUM_WARPS];

    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane = thread_idx % WARP_SIZE;

    // Get KV head index (for GQA/MQA)
    const int num_queries_per_kv = num_heads / num_kv_heads;
    const int kv_head_idx = head_idx / num_queries_per_kv;

    // ALiBi bias (if applicable)
    const float alibi_slope = alibi_slopes ? alibi_slopes[head_idx] : 0.0f;

    // Step 1: Load query vector
    // Each thread loads part of the query
    constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
    constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;
    const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
    const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

    constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
    using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
    constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
    constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
    __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];

    // TODO: Load query vectors in vectorized fashion
    // For now, simplified version:
    if (thread_group_idx < NUM_VECS_PER_THREAD) {
        const int vec_idx = thread_group_offset + thread_group_idx * THREAD_GROUP_SIZE;
        if (vec_idx * VEC_SIZE < HEAD_SIZE) {
            // Load would go here
            // q_vecs[thread_group_offset][thread_group_idx] = ...
        }
    }
    __syncthreads();

    // Step 2: Compute Q·K for all tokens
    float qk_max = -FLT_MAX;

    for (int block_idx = warp_idx; block_idx < num_seq_blocks; block_idx += NUM_WARPS) {
        const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

        // Load K vectors from this block and compute dot products
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            const int token_idx = block_idx * BLOCK_SIZE + i;
            if (token_idx >= seq_len) break;

            // TODO: Vectorized K loading and Q·K computation
            // For now, placeholder:
            float qk = 0.0f;  // Would compute: scale * dot(q, k)

            // Add ALiBi bias if applicable
            if (alibi_slope != 0.0f) {
                qk += alibi_slope * (token_idx - seq_len + 1);
            }

            // Store logit
            if (thread_idx == 0) {
                logits[token_idx] = qk;
            }

            qk_max = fmaxf(qk_max, qk);
        }
    }

    // Step 3: Warp-level reduction to find max logit
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
        qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
    }
    if (lane == 0) {
        red_smem[warp_idx] = qk_max;
    }
    __syncthreads();

    // Block-level reduction
    qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
    #pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
        qk_max = fmaxf(qk_max, SHFL_XOR_SYNC(qk_max, mask));
    }
    qk_max = SHFL_SYNC(qk_max, 0);

    // Step 4: Compute softmax
    float exp_sum = 0.0f;
    for (int i = thread_idx; i < seq_len; i += NUM_THREADS) {
        float val = __expf(logits[i] - qk_max);
        logits[i] = val;
        exp_sum += val;
    }
    exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

    // Normalize
    const float inv_sum = __fdividef(1.0f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < seq_len; i += NUM_THREADS) {
        logits[i] *= inv_sum;
    }
    __syncthreads();

    // Step 5: Compute attention output (softmax · V)
    constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
    constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
    constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
    constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

    float accs[NUM_ROWS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        accs[i] = 0.0f;
    }

    // TODO: Vectorized V loading and attention computation
    // This would iterate through blocks, load V vectors, multiply by softmax weights,
    // and accumulate into accs[]

    // Step 6: Warp-level reduction of attention output
    #pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        float acc = accs[i];
        #pragma unroll
        for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
            acc += SHFL_XOR_SYNC(acc, mask);
        }
        accs[i] = acc;
    }

    __syncthreads();

    // Step 7: Block-level reduction and write output
    float* out_smem = reinterpret_cast<float*>(shared_mem);

    // TODO: Full reduction across warps and final output write
    // For now, simplified version:
    if (warp_idx == 0 && lane < HEAD_SIZE) {
        scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
        from_float(out_ptr[lane], accs[0]);
    }
}

//
// Launcher function
//
// Handles type dispatch and kernel launch configuration
//

void paged_attention_v1_launcher(
    void* out,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_size,
    int block_size,
    int max_num_blocks_per_seq,
    const int* block_tables,
    const int* seq_lens,
    int max_seq_len,
    float scale,
    const float* alibi_slopes,
    ggml_type q_type,
    ggml_type kv_cache_type,
    cudaStream_t stream) {

    // Determine thread block configuration
    constexpr int NUM_THREADS = 128;
    dim3 grid(num_heads, num_seqs, 1);
    dim3 block(NUM_THREADS);

    // Calculate shared memory size
    const int padded_max_seq_len = DIVIDE_ROUND_UP(max_seq_len, block_size) * block_size;
    const int logits_size = padded_max_seq_len * sizeof(float);
    const int outputs_size = (NUM_THREADS / WARP_SIZE / 2) * head_size * sizeof(float);
    const int shared_mem_size = max(logits_size, outputs_size);

    // Compute strides
    const int q_stride = num_heads * head_size;
    const int kv_block_stride = num_kv_heads * head_size * block_size;
    const int kv_head_stride = head_size * block_size;

    // TODO: Type dispatch based on q_type and kv_cache_type
    // For now, simplified version assuming FP16:

    if (q_type == GGML_TYPE_F16 && kv_cache_type == GGML_TYPE_F16) {
        // Dispatch based on head size and block size
        if (head_size == 128 && block_size == 16) {
            paged_attention_v1_kernel<half, half, 128, 16, NUM_THREADS>
                <<<grid, block, shared_mem_size, stream>>>(
                    (half*)out, (const half*)query,
                    (const half*)key_cache, (const half*)value_cache,
                    num_kv_heads, scale, block_tables, seq_lens,
                    max_num_blocks_per_seq, alibi_slopes,
                    q_stride, kv_block_stride, kv_head_stride);
        }
        // TODO: Add cases for other head sizes: 32, 64, 80, 96, 112, 120, 192, 256
        // TODO: Add cases for other block sizes: 8, 32
    }
    // TODO: Add support for other data types (F32, quantized types)

    CUDA_CHECK(cudaGetLastError());
}

} // namespace ggml_cuda_paged_attention

#endif // GGML_USE_MUSA
