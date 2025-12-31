#include "chat-peg-parser.h"

#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;
using Tag = common_chat_peg_tag;

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count <= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

static std::string_view trim_space(std::string_view sv) {
    // Trim leading whitespace
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.front()))) {
        sv.remove_prefix(1);
    }
    // Trim trailing whitespace
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        sv.remove_suffix(1);
    }
    return sv;
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        map(node);
    });
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    auto tag = static_cast<Tag>(node.tag_id);
    if (tag == Tag::REASONING) {
        // Concatenate to handle multiple REASONING tags (trim trailing space like functional mapper)
        auto text = std::string(trim_trailing_space(node.text));
        if (!text.empty()) {
            result.reasoning_content += text;
        }
    } else if (tag == Tag::CONTENT) {
        // Concatenate to handle multiple CONTENT tags (no trimming, like functional mapper)
        result.content += std::string(node.text);
    } else if (tag != Tag::NONE) {
        throw std::runtime_error("Unexpected tag for this mapper: " + std::to_string(static_cast<int>(tag)));
    }
}

void common_chat_peg_native_mapper::map(const common_peg_ast_node & node) {
    auto tag = static_cast<Tag>(node.tag_id);
    switch (tag) {
        case Tag::TOOL:
        case Tag::TOOL_CLOSE:
        case Tag::REASONING_BLOCK:
            // Structural wrappers - do nothing.
            break;
        case Tag::TOOL_OPEN:
            // Be lazy: don't create tool call here, wait for TOOL_NAME.
            // This avoids creating spurious tool calls during partial parsing.
            current_tool = nullptr;
            pending_tool_id.clear();
            break;
        case Tag::TOOL_ID:
            // Skip partial nodes - the ID isn't complete yet
            if (node.is_partial) {
                break;
            }
            {
                auto text = std::string(trim_trailing_space(node.text));
                // Strip surrounding quotes if present (JSON string value)
                if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
                    text = text.substr(1, text.size() - 2);
                }
                if (current_tool) {
                    current_tool->id = text;
                } else {
                    // Buffer ID - TOOL_ID may come before TOOL_NAME (e.g., Command R7B)
                    pending_tool_id = text;
                }
            }
            break;
        case Tag::TOOL_NAME:
            // Skip partial nodes - the name isn't complete yet.
            // Note: Using p.atomic(p.literal_tag(Tag::TOOL_NAME, name)) in parsers would
            // achieve the same effect by preventing partial nodes from being created,
            // but this mapper-level check is more defensive and handles all parsers uniformly.
            if (node.is_partial) {
                break;
            }
            // Create tool call lazily on TOOL_NAME, not on TOOL_OPEN.
            result.tool_calls.emplace_back();
            current_tool = &result.tool_calls.back();
            current_tool->name = std::string(trim_trailing_space(node.text));
            // Apply pending ID if any
            if (!pending_tool_id.empty()) {
                current_tool->id = pending_tool_id;
                pending_tool_id.clear();
            }
            break;
        case Tag::TOOL_ARGS:
            if (current_tool) {
                current_tool->arguments = std::string(trim_trailing_space(node.text));
            }
            break;
        case Tag::REASONING:
        case Tag::CONTENT:
        case Tag::NONE:
            common_chat_peg_mapper::map(node);
            break;
        default:
            throw std::runtime_error("Unexpected tag for this mapper: " + std::to_string(static_cast<int>(tag)));
    }
}

void common_chat_peg_constructed_mapper::map(const common_peg_ast_node & node) {
    auto tag = static_cast<Tag>(node.tag_id);
    switch (tag) {
        case Tag::TOOL:
        case Tag::TOOL_ARG:
            // Structural wrappers - do nothing.
            break;
        case Tag::TOOL_OPEN:
            current_tool = nullptr;
            arg_count = 0;
            break;
        case Tag::TOOL_NAME:
            // Skip partial nodes - the name isn't complete yet.
            // Note: Using p.atomic(p.literal_tag(Tag::TOOL_NAME, name)) in parsers would
            // achieve the same effect by preventing partial nodes from being created,
            // but this mapper-level check is more defensive and handles all parsers uniformly.
            if (node.is_partial) {
                break;
            }
            if (current_tool) {
                throw std::runtime_error("bad state");
            }
            result.tool_calls.emplace_back();
            current_tool = &result.tool_calls.back();
            current_tool->name = std::string(node.text);
            current_tool->arguments = "{";
            break;
        case Tag::TOOL_ARG_OPEN:
            needs_closing_quote = false;
            break;
        case Tag::TOOL_ARG_NAME:
            // Skip partial nodes - the name isn't complete yet
            if (node.is_partial) {
                break;
            }
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                if (arg_count > 0) {
                    current_tool->arguments += ",";
                }
                current_tool->arguments += json(trim_trailing_space(node.text)).dump() + ":";
                ++arg_count;
            }
            break;
        case Tag::TOOL_ARG_STRING_VALUE:
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                // Serialize to JSON, but exclude the end quote
                // Use trim_space to remove leading/trailing whitespace from raw string values
                std::string dumped = json(trim_space(node.text)).dump();
                current_tool->arguments += dumped.substr(0, dumped.size() - 1);
                needs_closing_quote = true;
            }
            break;
        case Tag::TOOL_ARG_CLOSE:
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool && needs_closing_quote) {
                current_tool->arguments += "\"";
                needs_closing_quote = false;
            }
            break;
        case Tag::TOOL_ARG_JSON_VALUE:
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                current_tool->arguments += std::string(trim_trailing_space(node.text));
            }
            break;
        case Tag::TOOL_CLOSE:
            // Skip partial nodes - we shouldn't close arguments until we've seen
            // the full closing tag.
            if (node.is_partial) {
                break;
            }
            if (!current_tool) {
                throw std::runtime_error("bad state");
            }
            if (current_tool) {
                if (needs_closing_quote) {
                    current_tool->arguments += "\"";
                    needs_closing_quote = false;
                }
                current_tool->arguments += "}";
                current_tool = nullptr;
            }
            break;
        case Tag::REASONING:
        case Tag::CONTENT:
        case Tag::NONE:
            common_chat_peg_mapper::map(node);
            break;
        default:
            throw std::runtime_error("Unexpected tag for this mapper: " + std::to_string(static_cast<int>(tag)));
    }
}

// ============================================================================
// Functional mapper implementations (used by experimental new PEG parsers in chat-parsers/)
// ============================================================================

// Helper: Convert JSON value to arguments string (handles object, string, null cases)
static std::string json_to_arguments(const json & j) {
    if (j.is_object()) {
        return j.dump();
    }
    if (j.is_string()) {
        return j.get<std::string>();
    }
    if (!j.is_null()) {
        return j.dump();
    }
    return "{}";
}

// Helper: Populate tool call from JSON object with configurable field names
static void populate_tool_from_json(
    common_chat_tool_call & tool,
    const json & item,
    const char * name_key,
    const char * id_key,
    const char * args_key
) {
    if (item.contains(name_key)) {
        tool.name = item.at(name_key).get<std::string>();
    }
    if (id_key && item.contains(id_key)) {
        const auto & id = item.at(id_key);
        tool.id = id.is_string() ? id.get<std::string>() : std::to_string(id.get<int>());
    }
    if (item.contains(args_key)) {
        tool.arguments = json_to_arguments(item.at(args_key));
    } else {
        tool.arguments = "{}";
    }
}

// Helper: Handle base content tags (REASONING, CONTENT)
static void handle_base_tags(common_chat_msg & result, const common_peg_ast_node & node) {
    switch (static_cast<Tag>(node.tag_id)) {
        case Tag::REASONING:
            result.reasoning_content += std::string(trim_trailing_space(node.text));
            break;
        case Tag::CONTENT:
            // Don't trim content - preserve trailing whitespace for interleaved content
            result.content += std::string(node.text);
            break;
        default:
            break;
    }
}

common_chat_peg_mapper_func common_chat_peg_base_mapper() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        return [&result](const common_peg_ast_node & node) {
            handle_base_tags(result, node);
        };
    };
}

common_chat_peg_mapper_func common_chat_peg_native_mapper_func() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        common_chat_tool_call * current_tool = nullptr;
        std::string pending_tool_id;  // Buffer ID in case it comes before TOOL_NAME

        return [&result, current_tool, pending_tool_id](const common_peg_ast_node & node) mutable {
            handle_base_tags(result, node);

            switch (static_cast<Tag>(node.tag_id)) {
                case Tag::TOOL_OPEN:
                    // Be lazy: don't create tool call here, wait for TOOL_NAME.
                    // This avoids creating spurious tool calls during partial parsing.
                    current_tool = nullptr;
                    pending_tool_id.clear();
                    break;
                case Tag::TOOL_ID:
                    // Skip partial nodes - the ID isn't complete yet
                    if (node.is_partial) {
                        break;
                    }
                    {
                        auto text = std::string(trim_trailing_space(node.text));
                        // HACK: Strip surrounding quotes if present (JSON string value)
                        // TODO(ochafik): clean this up - ideally the parser should capture
                        // the string content without quotes, not the full JSON string value
                        if (text.size() >= 2 && text.front() == '"' && text.back() == '"') {
                            text = text.substr(1, text.size() - 2);
                        }
                        if (current_tool) {
                            current_tool->id = text;
                        } else {
                            // Buffer ID - TOOL_ID may come before TOOL_NAME (e.g., Command R7B)
                            pending_tool_id = text;
                        }
                    }
                    break;
                case Tag::TOOL_NAME:
                    // Skip partial nodes - see comment in common_chat_peg_mapper.
                    // Note: Using p.atomic(p.literal_tag(Tag::TOOL_NAME, name)) in parsers would
                    // achieve the same effect by preventing partial nodes from being created,
                    // but this mapper-level check is more defensive and handles all parsers uniformly.
                    if (node.is_partial) {
                        break;
                    }
                    // Create tool call lazily on TOOL_NAME, not on TOOL_OPEN.
                    result.tool_calls.emplace_back();
                    current_tool = &result.tool_calls.back();
                    current_tool->name = std::string(trim_trailing_space(node.text));
                    // Apply pending ID if any
                    if (!pending_tool_id.empty()) {
                        current_tool->id = pending_tool_id;
                        pending_tool_id.clear();
                    }
                    break;
                case Tag::TOOL_ARGS:
                    if (current_tool) {
                        current_tool->arguments = std::string(trim_trailing_space(node.text));
                    }
                    break;
                default:
                    break;
            }
        };
    };
}

common_chat_peg_mapper_func common_chat_peg_constructed_mapper_func() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        common_chat_tool_call * current_tool = nullptr;
        int arg_count = 0;
        bool needs_closing_quote = false;
        bool args_complete = false;  // True if TOOL_ARGS set complete arguments

        return [&result, current_tool, arg_count, needs_closing_quote, args_complete](const common_peg_ast_node & node) mutable {
            handle_base_tags(result, node);

            switch (static_cast<Tag>(node.tag_id)) {
                case Tag::TOOL_OPEN:
                    // Be lazy: don't create tool call here, wait for TOOL_NAME
                    // This avoids creating spurious tool calls during backtracking
                    current_tool = nullptr;
                    arg_count = 0;
                    args_complete = false;
                    break;
                case Tag::TOOL_NAME:
                    // Create tool call lazily on TOOL_NAME, not on TOOL_OPEN.
                    // Skip partial nodes - see comment in common_chat_peg_mapper.
                    if (node.is_partial) {
                        break;
                    }
                    result.tool_calls.emplace_back();
                    current_tool = &result.tool_calls.back();
                    current_tool->name = std::string(node.text);
                    current_tool->arguments = "{";
                    break;
                case Tag::TOOL_ARG_OPEN:
                    needs_closing_quote = false;
                    break;
                case Tag::TOOL_ARG_NAME:
                    // Skip partial nodes - the name isn't complete yet
                    if (node.is_partial) {
                        break;
                    }
                    if (current_tool) {
                        if (arg_count > 0) {
                            current_tool->arguments += ",";
                        }
                        current_tool->arguments += json(trim_trailing_space(node.text)).dump() + ":";
                        ++arg_count;
                    }
                    break;
                case Tag::TOOL_ARG_STRING_VALUE:
                    if (current_tool) {
                        // Trim trailing whitespace and serialize to JSON, but exclude the end quote
                        std::string trimmed = string_strip(std::string(node.text));
                        std::string dumped = json(trimmed).dump();
                        current_tool->arguments += dumped.substr(0, dumped.size() - 1);
                        needs_closing_quote = true;
                    }
                    break;
                case Tag::TOOL_ARG_CLOSE:
                    if (current_tool && needs_closing_quote) {
                        current_tool->arguments += "\"";
                        needs_closing_quote = false;
                    }
                    break;
                case Tag::TOOL_ARG_JSON_VALUE:
                    if (current_tool) {
                        current_tool->arguments += std::string(trim_trailing_space(node.text));
                    }
                    break;
                case Tag::TOOL_ARGS:
                    // For formats that use both constructed args and complete JSON args
                    // (e.g., Llama 3.x with builtin tools), replace the arguments entirely
                    if (current_tool) {
                        current_tool->arguments = std::string(trim_trailing_space(node.text));
                        args_complete = true;
                    }
                    break;
                case Tag::TOOL_CLOSE:
                    // Skip partial nodes - we shouldn't close arguments until we've seen
                    // the full closing tag (e.g., </function></tool_call>).
                    if (node.is_partial) {
                        break;
                    }
                    if (current_tool && !args_complete) {
                        if (needs_closing_quote) {
                            current_tool->arguments += "\"";
                            needs_closing_quote = false;
                        }
                        current_tool->arguments += "}";
                    }
                    break;
                default:
                    break;
            }
        };
    };
}

// Short form mapper: handles {"function_name": {"arg1": value1}} format (used by Apertus)
// The entire JSON array is captured in TOOL_ARGS, and we parse it to extract individual tool calls
common_chat_peg_mapper_func common_chat_peg_short_form_mapper() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        return [&result](const common_peg_ast_node & node) mutable {
            handle_base_tags(result, node);

            switch (static_cast<Tag>(node.tag_id)) {
                case Tag::TOOL_ARGS: {
                    // Parse the JSON array - format is [{"func_name": {...}}, ...]
                    try {
                        auto arr = json::parse(node.text);
                        if (!arr.is_array()) {
                            break;
                        }
                        for (const auto & item : arr) {
                            if (!item.is_object() || item.size() != 1) {
                                continue;
                            }
                            // The key is the function name, the value is the arguments
                            auto it = item.begin();
                            result.tool_calls.emplace_back();
                            auto & tool = result.tool_calls.back();
                            tool.name = it.key();
                            tool.arguments = json_to_arguments(it.value());
                        }
                    } catch (...) {
                        // JSON parse error - ignore
                    }
                    break;
                }
                default:
                    break;
            }
        };
    };
}

// Generic mapper: handles {"tool_call": {...}}, {"tool_calls": [...]}, or {"response": "..."} format
// The entire JSON is captured in TOOL_ARGS or CONTENT
common_chat_peg_mapper_func common_chat_peg_generic_mapper() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        return [&result](const common_peg_ast_node & node) mutable {
            switch (static_cast<Tag>(node.tag_id)) {
                case Tag::TOOL_ARGS: {
                    try {
                        auto data = json::parse(node.text);
                        if (data.contains("tool_calls") && data.at("tool_calls").is_array()) {
                            for (const auto & tc : data.at("tool_calls")) {
                                result.tool_calls.emplace_back();
                                populate_tool_from_json(result.tool_calls.back(), tc, "name", "id", "arguments");
                            }
                        } else if (data.contains("tool_call") && data.at("tool_call").is_object()) {
                            result.tool_calls.emplace_back();
                            populate_tool_from_json(result.tool_calls.back(), data.at("tool_call"), "name", "id", "arguments");
                        } else if (data.contains("response")) {
                            const auto & resp = data.at("response");
                            result.content = resp.is_string() ? resp.get<std::string>() : resp.dump();
                        }
                    } catch (...) {
                        // JSON parse error - ignore
                    }
                    break;
                }
                case Tag::CONTENT: {
                    // Content can be either:
                    // 1. Plain text (when no tools are available)
                    // 2. A JSON string value extracted from {"response": "..."}
                    result.content += std::string(node.text);
                    break;
                }
                default:
                    break;
            }
        };
    };
}

// OpenAI-style array mapper: handles [{"name": "func", "arguments": {...}, "id": "..."}] format
// Used by Mistral Nemo, Magistral, FireFunction, and similar formats
common_chat_peg_mapper_func common_chat_peg_oai_array_mapper() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        return [&result](const common_peg_ast_node & node) mutable {
            handle_base_tags(result, node);

            switch (static_cast<Tag>(node.tag_id)) {
                case Tag::TOOL_ARGS: {
                    try {
                        auto arr = json::parse(node.text);
                        if (!arr.is_array()) {
                            break;
                        }
                        for (const auto & item : arr) {
                            if (!item.is_object()) {
                                continue;
                            }
                            result.tool_calls.emplace_back();
                            populate_tool_from_json(result.tool_calls.back(), item, "name", "id", "arguments");
                        }
                    } catch (...) {
                        // JSON parse error - ignore
                    }
                    break;
                }
                default:
                    break;
            }
        };
    };
}

// Command R7B mapper: handles [{"tool_call_id": "0", "tool_name": "func", "parameters": {...}}] format
// The entire JSON array is captured in TOOL_ARGS, and we parse it to extract individual tool calls
common_chat_peg_mapper_func common_chat_peg_command_r7b_mapper() {
    return [](common_chat_msg & result) -> common_chat_peg_map_func {
        return [&result](const common_peg_ast_node & node) mutable {
            handle_base_tags(result, node);

            switch (static_cast<Tag>(node.tag_id)) {
                case Tag::TOOL_ARGS: {
                    try {
                        auto arr = json::parse(node.text);
                        if (!arr.is_array()) {
                            break;
                        }
                        for (const auto & item : arr) {
                            if (!item.is_object()) {
                                continue;
                            }
                            result.tool_calls.emplace_back();
                            populate_tool_from_json(result.tool_calls.back(), item, "tool_name", "tool_call_id", "parameters");
                        }
                    } catch (...) {
                        // JSON parse error - ignore
                    }
                    break;
                }
                default:
                    break;
            }
        };
    };
}
