// Mistral Nemo tool call format
// Format: [TOOL_CALLS][{"name":"func","arguments":{},"id":"abc123def"}]

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_mistral_nemo_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_MISTRAL_NEMO;

    data.preserved_tokens = {
        "[TOOL_CALLS]",
    };

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();

    // Build the PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // Tool call parser: [TOOL_CALLS] followed by a JSON array of tool calls
            // The template generates: [TOOL_CALLS][{"name": "fn1", ...}, {"name": "fn2", ...}]
            // So we capture [TOOL_CALLS] once, then the entire JSON array
            auto tool_call = p.tag(Tag::TOOL,
                p.atomic_tag(Tag::TOOL_OPEN, p.literal("[TOOL_CALLS]"))
                + p.tag(Tag::TOOL_ARGS, p.json())
            );

            // No repeat needed - [TOOL_CALLS] appears once with the entire array
            auto tool_calls = p.trigger_rule("tool-call-root", tool_call);

            return p.tag(Tag::CONTENT, p.until("[TOOL_CALLS]")) << tool_calls;
        }

        // Content only parser
        return p.tag(Tag::CONTENT, p.rest());
    });

    data.parser = parser.save();

    if (has_tools) {
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
                            // Nemo's template expects a 9-character alphanumeric ID.
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

        data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "[TOOL_CALLS]"});
    }

    return data;
}
