#pragma once

// Internal header for chat template format implementations.
// This header is NOT part of the public API and should only be included by:
// - common/chat.cpp (main implementation)
// - common/chat-parsers/*.cpp (per-format implementations)

#include "chat.h"
#include "chat-parser.h"
#include "chat-peg-parser.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "regex-partial.h"

#include <minja/chat-template.hpp>

#include <nlohmann/json.hpp>

#include <chrono>
#include <functional>
#include <optional>
#include <string>

// JSON type alias
using json = nlohmann::ordered_json;

// Template type alias (from minja)
typedef minja::chat_template common_chat_template;

// Parameters for template-based format initialization functions
struct templates_params {
    json messages;
    json tools;
    common_chat_tool_choice tool_choice;
    json json_schema;
    bool parallel_tool_calls;
    common_reasoning_format reasoning_format;
    bool stream;
    std::string grammar;
    bool add_generation_prompt = true;
    bool enable_thinking = true;
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    json extra_context;
    bool add_bos;
    bool add_eos;
    bool is_inference = true;
    // When true, use experimental new PEG parsers from chat-parsers/*.cpp instead of legacy parsers
    bool experimental_new_parsers = false;
};

// Helper to iterate over function tools
inline void foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

// Helper to iterate over function parameters
inline void foreach_parameter(const json & function, const std::function<void(const std::string &, const json &, bool)> & fn) {
    if (!function.contains("parameters") || !function.at("parameters").is_object()) {
        return;
    }
    const auto & params = function.at("parameters");
    if (!params.contains("properties") || !params.at("properties").is_object()) {
        return;
    }
    const auto & props = params.at("properties");
    std::set<std::string> required;
    if (params.contains("required") && params.at("required").is_array()) {
        params.at("required").get_to(required);
    }
    for (const auto & [name, prop] : props.items()) {
        bool is_required = (required.find(name) != required.end());
        fn(name, prop, is_required);
    }
}

// Format time for template contexts
inline std::string format_time(const std::chrono::system_clock::time_point & now, const std::string & format) {
    auto time = std::chrono::system_clock::to_time_t(now);
    auto local_time = *std::localtime(&time);
    std::ostringstream ss;
    ss << std::put_time(&local_time, format.c_str());
    return ss.str();
}

// Apply chat template with inputs
inline std::string apply(
    const common_chat_template & tmpl,
    const struct templates_params & inputs,
    const std::optional<json> & messages_override = std::nullopt,
    const std::optional<json> & tools_override = std::nullopt,
    const std::optional<json> & additional_context = std::nullopt)
{
    minja::chat_template_inputs tmpl_inputs;
    tmpl_inputs.messages = messages_override ? *messages_override : inputs.messages;
    if (tools_override) {
        tmpl_inputs.tools = *tools_override;
    } else {
        tmpl_inputs.tools = inputs.tools.empty() ? json() : inputs.tools;
    }
    tmpl_inputs.add_generation_prompt = inputs.add_generation_prompt;
    tmpl_inputs.extra_context = inputs.extra_context;
    tmpl_inputs.extra_context["enable_thinking"] = inputs.enable_thinking;
    if (additional_context) {
        tmpl_inputs.extra_context.merge_patch(*additional_context);
    }

    minja::chat_template_options tmpl_opts;
    auto result = tmpl.apply(tmpl_inputs, tmpl_opts);
    if (inputs.add_bos && string_starts_with(result, tmpl.bos_token())) {
        result = result.substr(tmpl.bos_token().size());
    }
    if (inputs.add_eos && string_ends_with(result, tmpl.eos_token())) {
        result = result.substr(0, result.size() - tmpl.eos_token().size());
    }
    return result;
}

// Type for format initialization functions
typedef common_chat_params (*common_chat_format_init_fn)(
    const common_chat_template & tmpl,
    const struct templates_params & params
);

// Type for format initialization functions that need extra inputs
typedef common_chat_params (*common_chat_format_init_fn_with_inputs)(
    const common_chat_template & tmpl,
    const struct templates_params & params,
    const common_chat_templates_inputs & inputs
);

// Type for llama_3_x style init that takes extra bool
typedef common_chat_params (*common_chat_format_init_fn_llama3x)(
    const common_chat_template & tmpl,
    const struct templates_params & params,
    bool allow_python_tag_builtin_tools
);

// Forward declarations for experimental new PEG parser implementations in chat-parsers/
common_chat_params common_chat_params_init_mistral_nemo_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_magistral_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_command_r7b_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_deepseek_r1_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_deepseek_v3_1_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_firefunction_v2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_hermes_2_pro_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_llama_3_x_peg(const common_chat_template & tmpl, const struct templates_params & inputs, bool allow_python_tag_builtin_tools);
common_chat_params common_chat_params_init_ministral_3_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_nemotron_v3_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_seed_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_nemotron_v2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_lfm2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_apertus_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_minimax_m2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_qwen3_coder_xml_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_kimi_k2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_apriel_1_5_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_xiaomi_mimo_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_glm_4_5_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_granite_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_functionary_v3_1_llama_3_1_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_functionary_v3_2_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_gpt_oss_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
common_chat_params common_chat_params_init_generic_peg(const common_chat_template & tmpl, const struct templates_params & inputs);
