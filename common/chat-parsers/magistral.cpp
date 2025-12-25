// Magistral tool call format
// Format: [THINK]...[/THINK][TOOL_CALLS][{"name":"func","arguments":{},"id":"abc123def"}]

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_magistral_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MAGISTRAL;

    data.preserved_tokens = {
        "[THINK]",
        "[/THINK]",
    };

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    // Build the PEG parser
    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Optional reasoning block
        auto reasoning = extract_reasoning
            ? p.optional("[THINK]" + p.tag(Tag::REASONING, p.until("[/THINK]")) + "[/THINK]")
            : p.eps();

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Tool call parser: content followed by [TOOL_CALLS] and JSON array
            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("[TOOL_CALLS]"))
                + p.tag(Tag::TOOL_ARGS, p.json())
            );

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, min_calls, max_calls));

            if (require_tools) {
                return reasoning << tool_calls;
            }
            return reasoning << p.tag(Tag::CONTENT, p.until("[TOOL_CALLS]")) << tool_calls;
        }

        // Content only parser
        return reasoning << p.tag(Tag::CONTENT, p.rest());
    });

    data.parser = parser.save();

    if (has_tools) {
        data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            auto schemas = json::array();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                schemas.push_back({
                    {"type", "object"},
                    {"properties", {
                        {"name", {
                            {"type", "string"},
                            {"const", function.at("name")},
                        }},
                        {"arguments", function.at("parameters")},
                        {"id", {
                            {"type", "string"},
                            {"pattern", "^[a-zA-Z0-9]{9}$"},
                        }},
                    }},
                    {"required", json::array({"name", "arguments", "id"})},
                });
            });
            auto schema = json {
                {"type", "array"},
                {"items", schemas.size() == 1 ? schemas[0] : json {{"anyOf", schemas}}},
                {"minItems", 1},
            };
            if (!inputs.parallel_tool_calls) {
                schema["maxItems"] = 1;
            }
            builder.add_rule("root", "\"[TOOL_CALLS]\" " + builder.add_schema("tool_calls", schema));
        });
        if (data.grammar_lazy) {
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
        } else {
            data.grammar_triggers.clear();
        }
        data.preserved_tokens.push_back("[TOOL_CALLS]");
    } else {
        data.grammar_lazy = false;
        if (!inputs.json_schema.is_null()) {
            if (!inputs.grammar.empty()) {
                throw std::runtime_error("Either \"json_schema\" or \"grammar\" can be specified, but not both");
            }
            data.grammar = json_schema_to_grammar(inputs.json_schema);
        } else {
            data.grammar = inputs.grammar;
        }
    }

    return data;
}
