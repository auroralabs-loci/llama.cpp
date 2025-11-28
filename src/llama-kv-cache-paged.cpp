#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-cparams.h"
#include "llama-hparams.h"
#include "llama-model.h"
#include "llama-kv-cache.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

//
// llama_kv_cache_paged implementation
//

llama_kv_cache_paged::llama_kv_cache_paged(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   block_size,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse)
    : model(model),
      hparams(model.hparams),
      type_k(type_k),
      type_v(type_v),
      n_seq_max(n_seq_max),
      block_size(block_size),
      num_blocks((kv_size + block_size - 1) / block_size) {  // ceil division

    GGML_ASSERT(block_size > 0 && block_size <= 256);
    GGML_ASSERT((block_size & (block_size - 1)) == 0 && "block_size must be power of 2");

    // Check environment variable for debug output
    const char * debug_env = std::getenv("LLAMA_KV_CACHE_DEBUG");
    if (debug_env) {
        debug = std::atoi(debug_env);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: initializing paged KV cache with %u blocks of size %u (total capacity: %u tokens)\n",
                __func__, num_blocks, block_size, num_blocks * block_size);
    }

    // Build layer list (same as standard KV cache)
    const int32_t n_layer = hparams.n_layer;

    for (int32_t il = 0; il < n_layer; ++il) {
        if (filter && !filter(il)) {
            continue;
        }

        // Check if this layer should reuse memory from another layer
        const int32_t il_reuse = reuse ? reuse(il) : -1;

        if (il_reuse >= 0) {
            // Reuse memory from another layer
            auto it = map_layer_ids.find(il_reuse);
            GGML_ASSERT(it != map_layer_ids.end() && "layer to reuse not found");
            map_layer_ids[il] = it->second;
            continue;
        }

        kv_layer layer;
        layer.il = il;

        // Initialize block storage
        layer.blocks.resize(num_blocks);
        for (uint32_t i = 0; i < num_blocks; ++i) {
            layer.blocks[i].id = i;
            layer.blocks[i].is_free = true;
            layer.blocks[i].ref_count = 0;
        }

        // Add to layer list
        const int32_t il_kv = static_cast<int32_t>(layers.size());
        layers.push_back(std::move(layer));
        map_layer_ids[il] = il_kv;
    }

    // Initialize free block list
    for (uint32_t i = 0; i < num_blocks; ++i) {
        free_blocks.push_back(i);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: created %zu layers with %u blocks each\n",
                __func__, layers.size(), num_blocks);
    }

    // Allocate tensor memory for blocks
    const int32_t n_embd_k_gqa = hparams.n_embd_k_gqa();
    // const int32_t n_embd_v_gqa = hparams.n_embd_v_gqa();  // unused for now
    const int32_t n_head_kv = hparams.n_head_kv();

    // Create context map for different buffer types
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*layers.size()*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);
            return ctx;
        }
        return it->second.get();
    };

    // Create tensors for each layer
    for (auto & layer : layers) {
        const int32_t il = layer.il;

        // Determine buffer type (CPU or GPU)
        bool offload = model.dev_layer(il) != nullptr;
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for paged kv cache");
        }

        // Create tensors for all blocks in this layer
        // Shape: [num_blocks, block_size, n_head_kv, head_size]
        const int64_t head_size = n_embd_k_gqa / n_head_kv;
        layer.k_all_blocks = ggml_new_tensor_4d(ctx, type_k, head_size, n_head_kv, block_size, num_blocks);
        layer.v_all_blocks = ggml_new_tensor_4d(ctx, type_v, head_size, n_head_kv, block_size, num_blocks);

        ggml_format_name(layer.k_all_blocks, "paged_cache_k_l%d", il);
        ggml_format_name(layer.v_all_blocks, "paged_cache_v_l%d", il);

        // Update individual block pointers to reference parts of the contiguous tensor
        for (uint32_t i = 0; i < num_blocks; ++i) {
            // Create views into the all_blocks tensors
            // Each block is a slice: [head_size, n_head_kv, block_size, 1]
            const size_t offset = i * layer.k_all_blocks->nb[3];
            layer.blocks[i].k_data = ggml_view_3d(ctx, layer.k_all_blocks,
                head_size, n_head_kv, block_size,
                layer.k_all_blocks->nb[1], layer.k_all_blocks->nb[2], offset);
            layer.blocks[i].v_data = ggml_view_3d(ctx, layer.v_all_blocks,
                head_size, n_head_kv, block_size,
                layer.v_all_blocks->nb[1], layer.v_all_blocks->nb[2], offset);
        }
    }

    // Allocate buffers for all contexts
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for paged kv cache");
        }

        if (debug > 0) {
            fprintf(stderr, "%s: %10s paged KV buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name(buf),
                    ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
        }

        // Clear buffer to avoid NaN values
        ggml_backend_buffer_clear(buf, 0);

        // Store context and buffer pair
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }
}

//
// llama_memory_i interface implementation
//

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(balloc);
    GGML_UNUSED(n_ubatch);
    GGML_UNUSED(embd_all);
    // TODO: Implement batch initialization
    // For now, return error status
    return llama_memory_context_ptr(
        new llama_kv_cache_context(LLAMA_MEMORY_STATUS_FAILED_PREPARE));
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    // TODO: Implement full cache initialization
    return llama_memory_context_ptr(
        new llama_kv_cache_context(LLAMA_MEMORY_STATUS_SUCCESS));
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(
        llama_context * lctx,
        bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);
    // TODO: Implement update initialization
    return llama_memory_context_ptr(
        new llama_kv_cache_context(LLAMA_MEMORY_STATUS_NO_UPDATE));
}

bool llama_kv_cache_paged::get_can_shift() const {
    // PagedAttention doesn't support context shifting
    // (blocks are allocated independently)
    return false;
}

void llama_kv_cache_paged::clear(bool data) {
    GGML_UNUSED(data);
    // Free all block tables
    block_tables.clear();
    seq_meta.clear();

    // Reset all blocks to free state
    for (auto & layer : layers) {
        for (auto & block : layer.blocks) {
            block.ref_count = 0;
            block.is_free = true;
        }
    }

    // Rebuild free block list
    free_blocks.clear();
    for (uint32_t i = 0; i < num_blocks; ++i) {
        free_blocks.push_back(i);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: cleared paged KV cache\n", __func__);
    }
}

bool llama_kv_cache_paged::seq_rm(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1) {
    // Remove tokens in range [p0, p1) from sequence
    auto it = block_tables.find(seq_id);
    if (it == block_tables.end()) {
        return false;
    }

    // For simplicity, if removing from middle of sequence, just fail
    // Full implementation would handle partial block removal
    if (p0 != -1 && p1 != -1) {
        fprintf(stderr, "%s: partial sequence removal not yet supported in paged cache\n", __func__);
        return false;
    }

    // Remove entire sequence
    auto & blocks = it->second;
    for (uint32_t block_id : blocks) {
        free_block(block_id);
    }

    block_tables.erase(it);
    seq_meta.erase(seq_id);

    if (debug > 0) {
        fprintf(stderr, "%s: removed sequence %d (%zu blocks freed)\n",
                __func__, seq_id, blocks.size());
    }

    return true;
}

void llama_kv_cache_paged::seq_cp(
        llama_seq_id seq_id_src,
        llama_seq_id seq_id_dst,
        llama_pos p0,
        llama_pos p1) {
    GGML_UNUSED(p1);
    // Copy sequence - in paged attention, this is efficient via block sharing
    auto it_src = block_tables.find(seq_id_src);
    if (it_src == block_tables.end()) {
        return;
    }

    // For simplicity, copy entire sequence (ignore p0, p1 for now)
    GGML_UNUSED(p0);
    auto & src_blocks = it_src->second;

    // Increment reference count on all blocks
    for (uint32_t block_id : src_blocks) {
        for (auto & layer : layers) {
            if (block_id < layer.blocks.size()) {
                layer.blocks[block_id].ref_count++;
            }
        }
    }

    // Share the block table
    block_tables[seq_id_dst] = src_blocks;

    // Copy metadata
    auto it_meta = seq_meta.find(seq_id_src);
    if (it_meta != seq_meta.end()) {
        seq_meta[seq_id_dst] = it_meta->second;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: copied sequence %d to %d (%zu blocks shared)\n",
                __func__, seq_id_src, seq_id_dst, src_blocks.size());
    }
}

void llama_kv_cache_paged::seq_keep(llama_seq_id seq_id) {
    // Remove all sequences except the specified one
    std::vector<llama_seq_id> to_remove;

    for (const auto & entry : block_tables) {
        if (entry.first != seq_id) {
            to_remove.push_back(entry.first);
        }
    }

    for (llama_seq_id sid : to_remove) {
        seq_rm(sid, -1, -1);
    }

    if (debug > 0) {
        fprintf(stderr, "%s: kept only sequence %d\n", __func__, seq_id);
    }
}

void llama_kv_cache_paged::seq_add(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        llama_pos shift) {
    GGML_UNUSED(p1);
    // Shift positions in sequence
    auto it = seq_meta.find(seq_id);
    if (it == seq_meta.end()) {
        return;
    }

    // Update position metadata
    if (p0 >= 0 && it->second.pos_min >= p0) {
        it->second.pos_min += shift;
    }
    if (p0 >= 0 && it->second.pos_max >= p0) {
        it->second.pos_max += shift;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: shifted sequence %d by %d\n", __func__, seq_id, shift);
    }
}

void llama_kv_cache_paged::seq_div(
        llama_seq_id seq_id,
        llama_pos p0,
        llama_pos p1,
        int d) {
    GGML_UNUSED(p0);
    GGML_UNUSED(p1);
    // Divide positions (used for attention scaling)
    // For paged attention, this is mostly metadata-only
    auto it = seq_meta.find(seq_id);
    if (it == seq_meta.end()) {
        return;
    }

    if (debug > 0) {
        fprintf(stderr, "%s: divided sequence %d positions by %d\n", __func__, seq_id, d);
    }

    // Position division would affect logical positioning but not block allocation
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = seq_meta.find(seq_id);
    return (it != seq_meta.end()) ? it->second.pos_min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = seq_meta.find(seq_id);
    return (it != seq_meta.end()) ? it->second.pos_max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    // TODO: Implement memory breakdown
    return std::map<ggml_backend_buffer_type_t, size_t>();
}

void llama_kv_cache_paged::state_write(
        llama_io_write_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) const {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    // TODO: Implement state serialization
    fprintf(stderr, "%s: state saving not yet implemented for paged cache\n", __func__);
}

void llama_kv_cache_paged::state_read(
        llama_io_read_i & io,
        llama_seq_id seq_id,
        llama_state_seq_flags flags) {
    GGML_UNUSED(io);
    GGML_UNUSED(seq_id);
    GGML_UNUSED(flags);
    // TODO: Implement state deserialization
    fprintf(stderr, "%s: state loading not yet implemented for paged cache\n", __func__);
}

//
// PagedAttention specific functions
//

const std::vector<uint32_t> & llama_kv_cache_paged::get_block_table(llama_seq_id seq_id) const {
    static const std::vector<uint32_t> empty;
    auto it = block_tables.find(seq_id);
    return (it != block_tables.end()) ? it->second : empty;
}

std::vector<int32_t> llama_kv_cache_paged::get_seq_lens() const {
    std::vector<int32_t> lens;
    lens.reserve(seq_meta.size());

    for (const auto & entry : seq_meta) {
        lens.push_back(static_cast<int32_t>(entry.second.length));
    }

    return lens;
}

ggml_tensor * llama_kv_cache_paged::get_k_blocks(int32_t il) const {
    // Map model layer ID to KV cache layer ID
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const int32_t il_kv = it->second;
    if (il_kv < 0 || il_kv >= static_cast<int32_t>(layers.size())) {
        return nullptr;
    }

    return layers[il_kv].k_all_blocks;
}

ggml_tensor * llama_kv_cache_paged::get_v_blocks(int32_t il) const {
    // Map model layer ID to KV cache layer ID
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return nullptr;
    }

    const int32_t il_kv = it->second;
    if (il_kv < 0 || il_kv >= static_cast<int32_t>(layers.size())) {
        return nullptr;
    }

    return layers[il_kv].v_all_blocks;
}

ggml_tensor * llama_kv_cache_paged::build_block_tables_tensor(ggml_context * ctx) const {
    // Build block tables tensor for all active sequences
    // Shape: [max_blocks_per_seq, n_seqs]

    if (block_tables.empty()) {
        return nullptr;
    }

    // Find maximum number of blocks per sequence
    size_t max_blocks = 0;
    for (const auto & [seq_id, blocks] : block_tables) {
        max_blocks = std::max(max_blocks, blocks.size());
    }

    const size_t n_seqs = block_tables.size();

    // Create tensor
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, n_seqs);
    ggml_set_input(tensor);

    // Fill with block IDs (will be done during set_input)
    // For now, the structure is created
    return tensor;
}

ggml_tensor * llama_kv_cache_paged::build_seq_lens_tensor(ggml_context * ctx) const {
    // Build sequence lengths tensor
    // Shape: [n_seqs]

    if (seq_meta.empty()) {
        return nullptr;
    }

    const size_t n_seqs = seq_meta.size();

    // Create tensor
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_seqs);
    ggml_set_input(tensor);

    return tensor;
}

//
// Block management (private)
//

uint32_t llama_kv_cache_paged::allocate_block() {
    if (free_blocks.empty()) {
        fprintf(stderr, "%s: ERROR: out of free blocks!\n", __func__);
        return UINT32_MAX;
    }

    uint32_t block_id = free_blocks.back();
    free_blocks.pop_back();

    // Mark block as allocated in all layers
    for (auto & layer : layers) {
        if (block_id < layer.blocks.size()) {
            layer.blocks[block_id].is_free = false;
            layer.blocks[block_id].ref_count = 1;
        }
    }

    if (debug > 1) {
        fprintf(stderr, "%s: allocated block %u (%zu free remaining)\n",
                __func__, block_id, free_blocks.size());
    }

    return block_id;
}

void llama_kv_cache_paged::free_block(uint32_t block_id) {
    if (block_id >= num_blocks) {
        return;
    }

    // Decrement reference count
    for (auto & layer : layers) {
        if (block_id < layer.blocks.size()) {
            auto & block = layer.blocks[block_id];

            if (block.ref_count > 0) {
                block.ref_count--;
            }

            // Free block if reference count reaches zero
            if (block.ref_count == 0 && !block.is_free) {
                block.is_free = true;
                free_blocks.push_back(block_id);

                if (debug > 1) {
                    fprintf(stderr, "%s: freed block %u (%zu free blocks total)\n",
                            __func__, block_id, free_blocks.size());
                }
            }
        }
    }
}

void llama_kv_cache_paged::allocate_blocks_for_sequence(
        llama_seq_id seq_id,
        uint32_t num_tokens) {
    // Calculate number of blocks needed
    uint32_t num_blocks_needed = (num_tokens + block_size - 1) / block_size;

    if (debug > 0) {
        fprintf(stderr, "%s: allocating %u blocks for sequence %d (%u tokens)\n",
                __func__, num_blocks_needed, seq_id, num_tokens);
    }

    // Allocate blocks
    auto & blocks = block_tables[seq_id];
    blocks.reserve(num_blocks_needed);

    for (uint32_t i = 0; i < num_blocks_needed; ++i) {
        uint32_t block_id = allocate_block();
        if (block_id == UINT32_MAX) {
            fprintf(stderr, "%s: ERROR: failed to allocate block %u/%u for sequence %d\n",
                    __func__, i, num_blocks_needed, seq_id);
            return;
        }
        blocks.push_back(block_id);
    }

    // Update sequence metadata
    auto & meta = seq_meta[seq_id];
    meta.length = num_tokens;
    meta.pos_min = 0;
    meta.pos_max = static_cast<llama_pos>(num_tokens - 1);
}

//
// Helper functions (private)
//

size_t llama_kv_cache_paged::total_size() const {
    return size_k_bytes() + size_v_bytes();
}

size_t llama_kv_cache_paged::size_k_bytes() const {
    // TODO: Calculate actual memory size based on tensor layouts
    return 0;
}

size_t llama_kv_cache_paged::size_v_bytes() const {
    // TODO: Calculate actual memory size based on tensor layouts
    return 0;
}
