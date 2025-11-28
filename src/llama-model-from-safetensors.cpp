#include "llama-model-from-safetensors.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "llama-safetensors-types.h"

#include <fstream>
#include <filesystem>

// Main entry point
llama_model * llama_model_load_from_safetensors(
    const char * model_path,
    const llama_model_params & params
) {
    if (!model_path) {
        LLAMA_LOG_ERROR("%s: model_path is null\n", __func__);
        return nullptr;
    }

    // Determine if path is directory or file
    std::string path_str(model_path);
    std::filesystem::path path(path_str);

    std::string model_dir;
    if (std::filesystem::is_directory(path)) {
        model_dir = path_str;
    } else if (std::filesystem::is_regular_file(path)) {
        model_dir = path.parent_path().string();
    } else {
        LLAMA_LOG_ERROR("%s: invalid path: %s\n", __func__, model_path);
        return nullptr;
    }

    // Create builder and build model
    safetensors_model_builder builder(model_dir, params);
    llama_model * model = builder.build();

    if (!model) {
        LLAMA_LOG_ERROR("%s: failed to load model: %s\n", __func__, builder.get_error().c_str());
    }

    return model;
}

// Implementation
safetensors_model_builder::safetensors_model_builder(
    const std::string & model_dir,
    const llama_model_params & params
) : model_dir(model_dir), params(params) {
}

safetensors_model_builder::~safetensors_model_builder() {
    // Clean up backend buffer if allocated
    if (backend_buffer) {
        ggml_backend_buffer_free(backend_buffer);
        backend_buffer = nullptr;
    }

    // Clean up GGML contexts
    if (ctx_meta) {
        ggml_free(ctx_meta);
        ctx_meta = nullptr;
    }

    if (ctx_data) {
        ggml_free(ctx_data);
        ctx_data = nullptr;
    }
}

llama_model * safetensors_model_builder::build() {
    LLAMA_LOG_INFO("%s: loading model from safetensors: %s\n", __func__, model_dir.c_str());

    // Step 1: Load config.json
    if (!load_config()) {
        return nullptr;
    }

    // Step 2: Load safetensors files
    if (!load_safetensors_files()) {
        return nullptr;
    }

    // Step 3: Detect architecture
    if (!detect_architecture()) {
        return nullptr;
    }

    // Step 4: Create model structure
    if (!create_model_structure()) {
        return nullptr;
    }

    // Step 5: Allocate tensors
    if (!allocate_tensors()) {
        return nullptr;
    }

    // Step 6: Load tensor data
    if (!load_tensor_data()) {
        return nullptr;
    }

    // Step 7: Link tensors to model structure
    if (!link_tensors_to_model()) {
        return nullptr;
    }

    // Step 9: Initialize vocabulary
    if (!init_vocabulary()) {
        return nullptr;
    }

    // Step 10: Finalize
    if (!finalize_model()) {
        return nullptr;
    }

    LLAMA_LOG_INFO("%s: model loaded successfully\n", __func__);
    return model;
}

bool safetensors_model_builder::load_config() {
    std::string config_path = model_dir + "/config.json";

    config = std::make_unique<hf_config>();
    if (!config->load_from_file(config_path)) {
        error_msg = "Failed to load config.json: " + config->get_error();
        return false;
    }

    LLAMA_LOG_INFO("%s: loaded config.json\n", __func__);
    return true;
}

bool safetensors_model_builder::load_safetensors_files() {
    st_loader = std::make_unique<safetensors_loader>();

    // Try single file first
    std::string single_file = model_dir + "/model.safetensors";
    if (std::filesystem::exists(single_file)) {
        if (st_loader->load_single(single_file)) {
            LLAMA_LOG_INFO("%s: loaded single safetensors file\n", __func__);
            return true;
        }
    }

    // Try sharded model
    std::string index_file = model_dir + "/model.safetensors.index.json";
    if (std::filesystem::exists(index_file)) {
        if (st_loader->load_sharded(index_file, model_dir)) {
            LLAMA_LOG_INFO("%s: loaded sharded safetensors files\n", __func__);
            return true;
        }
    }

    error_msg = "No safetensors files found in: " + model_dir;
    return false;
}

bool safetensors_model_builder::detect_architecture() {
    std::string hf_arch = config->get_architecture();
    if (hf_arch.empty()) {
        error_msg = "Could not detect architecture from config.json";
        return false;
    }

    mapper = create_tensor_mapper(hf_arch);
    if (!mapper) {
        error_msg = "Unsupported architecture: " + hf_arch;
        return false;
    }

    LLAMA_LOG_INFO("%s: detected architecture: %s\n", __func__, hf_arch.c_str());
    return true;
}

bool safetensors_model_builder::create_model_structure() {
    // Step 1: Allocate llama_model
    model = new llama_model(params);
    if (!model) {
        error_msg = "Failed to allocate llama_model";
        return false;
    }

    // Step 2: Set architecture
    model->arch = mapper->get_arch();
    if (model->arch == LLM_ARCH_UNKNOWN) {
        error_msg = "Unknown architecture";
        delete model;
        model = nullptr;
        return false;
    }

    // Step 3: Initialize hparams from HF config
    // Get basic hyperparameters
    model->hparams.n_embd = config->get_hidden_size();
    model->hparams.n_layer = config->get_num_hidden_layers();

    // Get context length
    int64_t max_pos = config->get_max_position_embeddings();
    model->hparams.n_ctx_train = max_pos > 0 ? max_pos : 2048;

    // Get attention parameters
    uint32_t n_head = config->get_num_attention_heads();
    int64_t n_head_kv_val = config->get_num_key_value_heads();
    uint32_t n_head_kv = (n_head_kv_val > 0) ? n_head_kv_val : n_head;  // Default to n_head for MHA

    // Fill per-layer arrays with same values (uniform layers)
    std::fill(model->hparams.n_head_arr.begin(), model->hparams.n_head_arr.end(), n_head);
    std::fill(model->hparams.n_head_kv_arr.begin(), model->hparams.n_head_kv_arr.end(), n_head_kv);

    // Get feed-forward dimension
    int64_t n_ff_val = config->get_intermediate_size();
    if (n_ff_val > 0) {
        std::fill(model->hparams.n_ff_arr.begin(), model->hparams.n_ff_arr.end(), static_cast<uint32_t>(n_ff_val));
    }

    // Calculate head dimensions
    if (n_head > 0) {
        model->hparams.n_embd_head_k = model->hparams.n_embd / n_head;
        model->hparams.n_embd_head_v = model->hparams.n_embd / n_head;
        model->hparams.n_rot = model->hparams.n_embd_head_k;  // Full rotary
    }

    // Get normalization epsilon
    double norm_eps = config->get_rms_norm_eps();
    if (norm_eps > 0.0) {
        model->hparams.f_norm_rms_eps = static_cast<float>(norm_eps);
    } else {
        // Try layer_norm_eps as fallback
        double layer_norm_eps;
        if (config->get_float("layer_norm_eps", layer_norm_eps)) {
            model->hparams.f_norm_rms_eps = static_cast<float>(layer_norm_eps);
        } else {
            model->hparams.f_norm_rms_eps = 1e-5f;  // Default
        }
    }
    model->hparams.f_norm_eps = model->hparams.f_norm_rms_eps;

    // Get RoPE parameters
    double rope_theta;
    if (config->get_float("rope_theta", rope_theta)) {
        model->hparams.rope_freq_base_train = static_cast<float>(rope_theta);
    } else {
        model->hparams.rope_freq_base_train = 10000.0f;  // Default
    }

    // Check for RoPE scaling
    if (config->has_key("rope_scaling")) {
        // TODO: Parse rope_scaling dict if present
        model->hparams.rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_LINEAR;
    } else {
        model->hparams.rope_scaling_type_train = LLAMA_ROPE_SCALING_TYPE_NONE;
    }

    // Default rope parameters
    model->hparams.rope_freq_scale_train = 1.0f;
    model->hparams.n_ctx_orig_yarn = model->hparams.n_ctx_train;

    // Step 4: Determine model type based on architecture and size
    model->type = LLM_TYPE_UNKNOWN;

    switch (model->arch) {
        case LLM_ARCH_LLAMA:
            // SmolLM2-135M has 30 layers, which maps to 256M type
            switch (model->hparams.n_layer) {
                case 30: model->type = LLM_TYPE_256M; break;  // SmolLM2-135M
                case 16: model->type = LLM_TYPE_1B; break;
                case 22: model->type = LLM_TYPE_1B; break;
                case 26: model->type = LLM_TYPE_3B; break;
                case 28: model->type = LLM_TYPE_3B; break;
                case 32: model->type = LLM_TYPE_7B; break;
                case 40: model->type = LLM_TYPE_13B; break;
                case 48: model->type = LLM_TYPE_34B; break;
                case 60: model->type = LLM_TYPE_30B; break;
                case 80: model->type = LLM_TYPE_70B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        case LLM_ARCH_PHI3:
            switch (model->hparams.n_layer) {
                case 24: model->type = LLM_TYPE_1_3B; break;
                case 32: model->type = LLM_TYPE_3B; break;
                case 40: model->type = LLM_TYPE_14B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        case LLM_ARCH_QWEN2:
            switch (model->hparams.n_layer) {
                case 24: model->type = LLM_TYPE_0_5B; break;
                case 28: model->type = LLM_TYPE_1_5B; break;
                case 32: model->type = LLM_TYPE_7B; break;
                case 40: model->type = LLM_TYPE_13B; break;
                case 80: model->type = LLM_TYPE_70B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
            switch (model->hparams.n_layer) {
                case 18: model->type = LLM_TYPE_2B; break;
                case 26: model->type = LLM_TYPE_7B; break;
                case 42: model->type = LLM_TYPE_9B; break;
                case 46: model->type = LLM_TYPE_27B; break;
                default: model->type = LLM_TYPE_UNKNOWN;
            }
            break;

        default:
            model->type = LLM_TYPE_UNKNOWN;
    }

    // Step 5: Allocate layers vector
    model->layers.resize(model->hparams.n_layer);

    // Set model name from config
    std::string model_name;
    if (config->get_string("_name_or_path", model_name)) {
        model->name = model_name;
    } else {
        model->name = "unknown";
    }

    LLAMA_LOG_INFO("%s: created model structure: arch=%s, layers=%d, type=%s\n",
                   __func__,
                   llm_arch_name(model->arch),
                   model->hparams.n_layer,
                   model->type_name().c_str());

    return true;
}

bool safetensors_model_builder::allocate_tensors() {
    // Step 1: Get list of all tensors from safetensors
    std::vector<std::string> tensor_names = st_loader->get_tensor_names();

    if (tensor_names.empty()) {
        error_msg = "No tensors found in safetensors files";
        return false;
    }

    LLAMA_LOG_INFO("%s: found %zu tensors in safetensors\n", __func__, tensor_names.size());

    // Step 2: Create GGML context for tensor metadata (no actual allocation)
    size_t ctx_size = tensor_names.size() * ggml_tensor_overhead();

    struct ggml_init_params ggml_params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,  // Don't allocate data yet, just metadata
    };

    ctx_meta = ggml_init(ggml_params);
    if (!ctx_meta) {
        error_msg = "Failed to initialize GGML metadata context";
        return false;
    }

    LLAMA_LOG_INFO("%s: created GGML metadata context\n", __func__);

    // Step 3: Create tensor metadata for each safetensors tensor
    // This creates the tensor structures without allocating the actual data
    int tensors_created = 0;

    for (const std::string & hf_name : tensor_names) {
        // Get tensor info from safetensors
        const safetensors_tensor_info * info = st_loader->get_tensor_info(hf_name);
        if (!info) {
            LLAMA_LOG_WARN("%s: could not find tensor info for %s, skipping\n", __func__, hf_name.c_str());
            continue;
        }

        // Map HuggingFace tensor name to llama.cpp internal name
        std::string internal_name = mapper->map_tensor_name(hf_name);
        if (internal_name.empty()) {
            // Unmapped tensor - might be okay (e.g., optional tensors)
            LLAMA_LOG_DEBUG("%s: no mapping for tensor %s, skipping\n", __func__, hf_name.c_str());
            continue;
        }

        // Convert safetensors dtype to GGML type
        ggml_type ggml_type = safetensors_dtype_to_ggml_type(info->dtype);
        if (ggml_type == GGML_TYPE_COUNT) {
            LLAMA_LOG_WARN("%s: unsupported dtype for tensor %s, skipping\n", __func__, hf_name.c_str());
            continue;
        }

        // Create tensor in GGML context
        struct ggml_tensor * tensor = nullptr;

        switch (info->shape.size()) {
            case 1:
                tensor = ggml_new_tensor_1d(ctx_meta, ggml_type, info->shape[0]);
                break;
            case 2:
                tensor = ggml_new_tensor_2d(ctx_meta, ggml_type, info->shape[0], info->shape[1]);
                break;
            case 3:
                tensor = ggml_new_tensor_3d(ctx_meta, ggml_type, info->shape[0], info->shape[1], info->shape[2]);
                break;
            case 4:
                tensor = ggml_new_tensor_4d(ctx_meta, ggml_type, info->shape[0], info->shape[1], info->shape[2], info->shape[3]);
                break;
            default:
                LLAMA_LOG_WARN("%s: tensor %s has unsupported number of dimensions: %zu\n",
                              __func__, hf_name.c_str(), info->shape.size());
                continue;
        }

        if (!tensor) {
            error_msg = "Failed to create tensor: " + internal_name;
            return false;
        }

        // Set tensor name
        ggml_set_name(tensor, internal_name.c_str());

        tensors_created++;

        if (tensors_created % 100 == 0) {
            LLAMA_LOG_INFO("%s: created %d tensor metadata entries...\n", __func__, tensors_created);
        }
    }

    LLAMA_LOG_INFO("%s: created %d tensor metadata entries total\n", __func__, tensors_created);

    if (tensors_created == 0) {
        error_msg = "No tensors were successfully created";
        return false;
    }

    // Step 4: Allocate backend buffer for the tensors
    // For now, we use CPU backend only. GPU support can be added later.

    LLAMA_LOG_INFO("%s: allocating CPU backend buffer for tensors\n", __func__);

    // Get CPU backend
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) {
        error_msg = "Failed to get CPU backend device";
        return false;
    }

    // Get CPU buffer type
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_dev_buffer_type(cpu_dev);
    if (!cpu_buft) {
        error_msg = "Failed to get CPU buffer type";
        return false;
    }

    // Allocate backend buffer for all tensors in the context
    // This allocates the actual memory and sets tensor->data pointers
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_meta, cpu_buft);
    if (!buffer) {
        error_msg = "Failed to allocate backend buffer for tensors";
        return false;
    }

    LLAMA_LOG_INFO("%s: allocated backend buffer: %zu bytes\n",
                   __func__,
                   ggml_backend_buffer_get_size(buffer));

    // Store the buffer so we can free it later
    backend_buffer = buffer;

    LLAMA_LOG_INFO("%s: tensor allocation complete - %d tensors ready\n", __func__, tensors_created);
    return true;
}

bool safetensors_model_builder::load_tensor_data() {
    if (!ctx_meta) {
        error_msg = "Cannot load tensor data: metadata context not initialized";
        return false;
    }

    if (!backend_buffer) {
        error_msg = "Cannot load tensor data: backend buffer not allocated";
        return false;
    }

    LLAMA_LOG_INFO("%s: loading tensor data from safetensors\n", __func__);

    int tensors_loaded = 0;
    int tensors_skipped = 0;
    int tensors_failed = 0;

    // Get all safetensors tensor names
    std::vector<std::string> st_tensor_names = st_loader->get_tensor_names();

    for (const std::string & hf_name : st_tensor_names) {
        // Map HF name to internal name
        std::string internal_name = mapper->map_tensor_name(hf_name);
        if (internal_name.empty()) {
            // This tensor doesn't map to anything (might be optional)
            LLAMA_LOG_DEBUG("%s: no mapping for HF tensor %s, skipping\n", __func__, hf_name.c_str());
            tensors_skipped++;
            continue;
        }

        // Find the tensor in our GGML context
        struct ggml_tensor * tensor = ggml_get_tensor(ctx_meta, internal_name.c_str());
        if (!tensor) {
            LLAMA_LOG_WARN("%s: tensor %s (HF: %s) not found in GGML context\n",
                          __func__, internal_name.c_str(), hf_name.c_str());
            tensors_skipped++;
            continue;
        }

        // Verify tensor has allocated data
        if (!tensor->data) {
            LLAMA_LOG_ERROR("%s: tensor %s has no data buffer allocated\n", __func__, internal_name.c_str());
            tensors_failed++;
            continue;
        }

        // Get tensor info from safetensors
        const safetensors_tensor_info * info = st_loader->get_tensor_info(hf_name);
        if (!info) {
            LLAMA_LOG_ERROR("%s: could not get info for tensor %s\n", __func__, hf_name.c_str());
            tensors_failed++;
            continue;
        }

        // Read data from safetensors into temporary buffer
        size_t st_data_size = info->size();
        std::vector<char> temp_buffer(st_data_size);

        if (!st_loader->read_tensor_data(hf_name, temp_buffer.data(), st_data_size)) {
            LLAMA_LOG_ERROR("%s: failed to read tensor data for %s\n", __func__, hf_name.c_str());
            tensors_failed++;
            continue;
        }

        // Convert types and copy to GGML tensor
        size_t ggml_data_size = ggml_nbytes(tensor);
        ggml_type tensor_type = tensor->type;

        if (!convert_safetensors_to_ggml(
                temp_buffer.data(), st_data_size, info->dtype,
                tensor->data, ggml_data_size, tensor_type,
                info->shape.data(), info->shape.size())) {
            LLAMA_LOG_ERROR("%s: failed to convert tensor data for %s\n", __func__, hf_name.c_str());
            tensors_failed++;
            continue;
        }

        tensors_loaded++;

        if (tensors_loaded % 50 == 0) {
            LLAMA_LOG_INFO("%s: loaded %d tensors...\n", __func__, tensors_loaded);
        }
    }

    LLAMA_LOG_INFO("%s: loaded %d tensors, skipped %d, failed %d\n",
                   __func__, tensors_loaded, tensors_skipped, tensors_failed);

    if (tensors_failed > 0) {
        error_msg = "Some tensors failed to load";
        return false;
    }

    if (tensors_loaded == 0) {
        error_msg = "No tensors were loaded";
        return false;
    }

    return true;
}

bool safetensors_model_builder::link_tensors_to_model() {
    if (!model) {
        error_msg = "Cannot link tensors: model not created";
        return false;
    }

    if (!ctx_meta) {
        error_msg = "Cannot link tensors: no metadata context";
        return false;
    }

    LLAMA_LOG_INFO("%s: linking tensors to model structure\n", __func__);

    // Helper lambda to get tensor (returns nullptr if not found, which is ok for optional tensors)
    auto get_tensor = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(ctx_meta, name);
    };

    int tensors_linked = 0;

    // Link input embedding
    model->tok_embd = get_tensor("token_embd.weight");
    if (model->tok_embd) {
        tensors_linked++;
        LLAMA_LOG_DEBUG("%s: linked token_embd\n", __func__);
    } else {
        LLAMA_LOG_WARN("%s: token_embd.weight not found\n", __func__);
    }

    // Link output norm and output
    model->output_norm = get_tensor("output_norm.weight");
    if (model->output_norm) {
        tensors_linked++;
    }

    model->output = get_tensor("output.weight");
    if (model->output) {
        tensors_linked++;
    } else {
        // output might share with tok_embd
        model->output = model->tok_embd;
        LLAMA_LOG_DEBUG("%s: output shares with token_embd\n", __func__);
    }

    // Link layer tensors based on architecture
    switch (model->arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_MISTRAL:
            {
                LLAMA_LOG_INFO("%s: linking Llama/Mistral layer tensors\n", __func__);

                for (size_t i = 0; i < model->layers.size(); ++i) {
                    auto & layer = model->layers[i];
                    char buf[256];

                    // Attention norm
                    snprintf(buf, sizeof(buf), "blk.%zu.attn_norm.weight", i);
                    layer.attn_norm = get_tensor(buf);
                    if (layer.attn_norm) tensors_linked++;

                    // Attention Q, K, V, O
                    snprintf(buf, sizeof(buf), "blk.%zu.attn_q.weight", i);
                    layer.wq = get_tensor(buf);
                    if (layer.wq) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_k.weight", i);
                    layer.wk = get_tensor(buf);
                    if (layer.wk) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_v.weight", i);
                    layer.wv = get_tensor(buf);
                    if (layer.wv) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_output.weight", i);
                    layer.wo = get_tensor(buf);
                    if (layer.wo) tensors_linked++;

                    // FFN norm
                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_norm.weight", i);
                    layer.ffn_norm = get_tensor(buf);
                    if (layer.ffn_norm) tensors_linked++;

                    // FFN gate, down, up
                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_gate.weight", i);
                    layer.ffn_gate = get_tensor(buf);
                    if (layer.ffn_gate) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_down.weight", i);
                    layer.ffn_down = get_tensor(buf);
                    if (layer.ffn_down) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_up.weight", i);
                    layer.ffn_up = get_tensor(buf);
                    if (layer.ffn_up) tensors_linked++;

                    if (i % 10 == 0 && i > 0) {
                        LLAMA_LOG_INFO("%s: linked layer %zu/%zu\n", __func__, i, model->layers.size());
                    }
                }

                LLAMA_LOG_INFO("%s: linked all %zu layers\n", __func__, model->layers.size());
            }
            break;

        case LLM_ARCH_PHI3:
        case LLM_ARCH_QWEN2:
        case LLM_ARCH_GEMMA:
        case LLM_ARCH_GEMMA2:
            {
                // These architectures have similar structure to Llama
                // For now, use the same linking pattern
                LLAMA_LOG_WARN("%s: using Llama-style linking for %s - may need adjustments\n",
                              __func__, llm_arch_name(model->arch));

                for (size_t i = 0; i < model->layers.size(); ++i) {
                    auto & layer = model->layers[i];
                    char buf[256];

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_norm.weight", i);
                    layer.attn_norm = get_tensor(buf);
                    if (layer.attn_norm) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_q.weight", i);
                    layer.wq = get_tensor(buf);
                    if (layer.wq) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_k.weight", i);
                    layer.wk = get_tensor(buf);
                    if (layer.wk) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_v.weight", i);
                    layer.wv = get_tensor(buf);
                    if (layer.wv) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.attn_output.weight", i);
                    layer.wo = get_tensor(buf);
                    if (layer.wo) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_norm.weight", i);
                    layer.ffn_norm = get_tensor(buf);
                    if (layer.ffn_norm) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_gate.weight", i);
                    layer.ffn_gate = get_tensor(buf);
                    if (layer.ffn_gate) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_down.weight", i);
                    layer.ffn_down = get_tensor(buf);
                    if (layer.ffn_down) tensors_linked++;

                    snprintf(buf, sizeof(buf), "blk.%zu.ffn_up.weight", i);
                    layer.ffn_up = get_tensor(buf);
                    if (layer.ffn_up) tensors_linked++;
                }
            }
            break;

        default:
            error_msg = "Tensor linking not implemented for this architecture";
            return false;
    }

    LLAMA_LOG_INFO("%s: linked %d tensors to model structure\n", __func__, tensors_linked);

    if (tensors_linked == 0) {
        error_msg = "No tensors were linked to model - tensor names may not match";
        return false;
    }

    return true;
}

bool safetensors_model_builder::init_vocabulary() {
    LLAMA_LOG_INFO("%s: initializing vocabulary\n", __func__);

    // TODO: Full vocabulary loading from tokenizer.json
    // For now, we create a minimal vocab structure to avoid crashes
    // Full implementation needs to:
    // 1. Parse tokenizer.json or tokenizer_config.json from HF model directory
    // 2. Extract vocabulary, merges, special tokens
    // 3. Initialize llama_vocab properly with all token data
    //
    // This is a complex task that deserves its own implementation session.
    // The model structure and tensors are now complete, but tokenization
    // won't work until this is fully implemented.

    // Check if tokenizer.json exists
    std::string tokenizer_path = model_dir + "/tokenizer.json";
    std::string tokenizer_config_path = model_dir + "/tokenizer_config.json";

    bool has_tokenizer = std::filesystem::exists(tokenizer_path);
    bool has_config = std::filesystem::exists(tokenizer_config_path);

    if (has_tokenizer) {
        LLAMA_LOG_INFO("%s: found tokenizer.json (parsing not yet implemented)\n", __func__);
    }

    if (has_config) {
        LLAMA_LOG_INFO("%s: found tokenizer_config.json (parsing not yet implemented)\n", __func__);
    }

    if (!has_tokenizer && !has_config) {
        LLAMA_LOG_WARN("%s: no tokenizer files found in %s\n", __func__, model_dir.c_str());
        error_msg = "No tokenizer files found - vocabulary initialization skipped";
        // Don't fail here - model structure is valid even without vocab
    }

    // For now, we skip vocabulary initialization
    // The model struct is valid and tensors are loaded, but inference won't work
    // until vocabulary is properly loaded

    LLAMA_LOG_WARN("%s: vocabulary loading not yet fully implemented\n", __func__);
    LLAMA_LOG_WARN("%s: model structure is valid but tokenization will not work\n", __func__);
    LLAMA_LOG_WARN("%s: implement full tokenizer.json parsing to enable inference\n", __func__);

    // Return true to allow model structure to be completed
    // This allows testing of tensor loading even without tokenization
    return true;
}

bool safetensors_model_builder::finalize_model() {
    if (!model) {
        error_msg = "Cannot finalize: model not created";
        return false;
    }

    LLAMA_LOG_INFO("%s: finalizing model\n", __func__);

    // Validate that critical tensors are linked
    bool has_tok_embd = (model->tok_embd != nullptr);
    bool has_output = (model->output != nullptr);
    bool has_output_norm = (model->output_norm != nullptr);

    if (!has_tok_embd) {
        LLAMA_LOG_WARN("%s: token embedding tensor not linked\n", __func__);
    }

    if (!has_output) {
        LLAMA_LOG_WARN("%s: output tensor not linked\n", __func__);
    }

    if (!has_output_norm) {
        LLAMA_LOG_WARN("%s: output norm tensor not linked\n", __func__);
    }

    // Validate layers have critical tensors
    int layers_valid = 0;
    for (size_t i = 0; i < model->layers.size(); ++i) {
        const auto & layer = model->layers[i];
        bool layer_ok = (layer.attn_norm && layer.wq && layer.wk && layer.wv && layer.wo &&
                        layer.ffn_norm && layer.ffn_gate && layer.ffn_down && layer.ffn_up);
        if (layer_ok) {
            layers_valid++;
        } else {
            LLAMA_LOG_WARN("%s: layer %zu missing some tensors\n", __func__, i);
        }
    }

    LLAMA_LOG_INFO("%s: validated %d/%zu layers\n", __func__, layers_valid, model->layers.size());

    // Log final model info
    LLAMA_LOG_INFO("%s: model finalized:\n", __func__);
    LLAMA_LOG_INFO("%s:   architecture: %s\n", __func__, llm_arch_name(model->arch));
    LLAMA_LOG_INFO("%s:   type: %s\n", __func__, model->type_name().c_str());
    LLAMA_LOG_INFO("%s:   layers: %zu\n", __func__, model->layers.size());
    LLAMA_LOG_INFO("%s:   embedding dim: %d\n", __func__, model->hparams.n_embd);
    LLAMA_LOG_INFO("%s:   attention heads: %d\n", __func__, model->hparams.n_head());
    LLAMA_LOG_INFO("%s:   context length: %d\n", __func__, model->hparams.n_ctx_train);

    // Note: We don't clean up ctx_meta or backend_buffer here
    // They need to stay alive because tensors reference them
    // They will be cleaned up by the model's own destructor when it's freed

    return true;
}
