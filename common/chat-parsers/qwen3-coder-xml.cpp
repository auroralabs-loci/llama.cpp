// Qwen3 Coder XML tool call format
// Format: <tool_call><function=name><parameter=key>value</parameter></function></tool_call>

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_qwen3_coder_xml_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_QWEN3_CODER_XML;

    data.preserved_tokens = {
        "<tool_call>",
        "</tool_call>",
        "<function=",
        "</function>",
        "<parameter=",
        "</parameter>",
    };

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar = true;

    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        const auto consume_end_block = [&]() {
            auto optional_end = p.optional(p.choice({
                p.literal("<|im_end|>"),
                p.literal("<|endoftext|>")
            }));
            return p.optional(p.literal("\n")) + optional_end + p.optional(p.literal("\n"));
        };

        const auto content_until = [&](const std::string & marker, bool allow_inline) {
            std::vector<std::string> delimiters = {
                std::string("\r\n") + marker,
                std::string("\n") + marker,
            };
            if (allow_inline) {
                delimiters.push_back(marker);
            }
            return p.tag(Tag::CONTENT, p.until_one_of(delimiters));
        };

        const auto content_before_tool = p.optional(p.rule("qwen-tool-prefix",
            p.tag(Tag::CONTENT, p.until("<tool_call>"))
            + p.peek(p.literal("<tool_call>"))
        ));

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema))
                << consume_end_block();
        }

        // Tool call parser
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto parameter_name = p.choice();
            parameter_name |= p.tag(Tag::TOOL_ARG_NAME, p.until(">\r\n"));
            parameter_name |= p.tag(Tag::TOOL_ARG_NAME, p.until(">\n"));
            parameter_name |= p.tag(Tag::TOOL_ARG_NAME, p.until(">"));
            auto parameter_terminator = p.choice({
                p.literal(">\r\n"),
                p.literal(">\n"),
                p.literal(">"),
            });

            auto tool_choice = p.choice();
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");

                auto schema_info = common_schema_info();
                schema_info.resolve_refs(parameters);

                // By JSON Schema spec, missing additionalProperties defaults to true
                bool allow_additional = true;
                bool additional_has_schema = false;
                json additional_schema;
                if (parameters.contains("additionalProperties")) {
                    const auto & additional = parameters.at("additionalProperties");
                    if (additional.is_boolean()) {
                        allow_additional = additional.get<bool>();
                    } else if (additional.is_object()) {
                        allow_additional = true;
                        additional_has_schema = true;
                        additional_schema = additional;
                    }
                }

                auto args = p.sequence();
                foreach_parameter(function, [&](const std::string & param_name, const json & param_schema, bool /* is_required */) {
                    auto parameter_value = p.schema_or_raw_string_until("qwen-param-" + name + "-" + param_name, param_schema, "</parameter>",
                        schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, true);

                    auto arg_rule = p.rule("qwen-parameter-" + name + "-" + param_name,
                        p.atomic_tag(Tag::TOOL_ARG_OPEN,
                            p.literal("<parameter=")
                            + p.literal_tag(Tag::TOOL_ARG_NAME, param_name)
                            + parameter_terminator)
                        + parameter_value
                        + p.atomic_tag(Tag::TOOL_ARG_CLOSE, p.literal("</parameter>"))
                        + p.space()  // Allow whitespace after </parameter>
                    );

                    args += p.repeat(arg_rule, /* min = */ 0, /* max = */ 1);
                });

                if (allow_additional) {
                    auto additional_value = additional_has_schema
                        ? p.schema_or_raw_string_until("qwen-param-" + name + "-additional", additional_schema, "</parameter>",
                            schema_info, Tag::TOOL_ARG_STRING_VALUE, Tag::TOOL_ARG_JSON_VALUE, true)
                        : p.tag(Tag::TOOL_ARG_STRING_VALUE, p.until("</parameter>"));

                    auto additional_rule = p.rule("qwen-parameter-generic-" + name,
                        p.atomic_tag(Tag::TOOL_ARG_OPEN,
                            p.literal("<parameter=")
                            + parameter_name
                            + parameter_terminator)
                        + additional_value
                        + p.atomic_tag(Tag::TOOL_ARG_CLOSE, p.literal("</parameter>"))
                        + p.space()  // Allow whitespace after </parameter>
                    );

                    args += p.repeat(additional_rule, 0, -1);
                }

                // Format: <function=name><parameter=key>value</parameter></function>
                // Allow optional whitespace/indentation for flexibility
                tool_choice |= p.rule("tool-" + name,
                    p.atomic_tag(Tag::TOOL_OPEN, p.literal("<function=") + p.literal_tag(Tag::TOOL_NAME, name) + p.literal(">"))
                    + p.space()  // Allow whitespace after <function=name>
                    + args
                    + p.space()  // Allow whitespace before </function>
                    + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</function>"))
                );
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            // Format:<tool_call>\n<function=name>...</function>\n</tool_call>
            // Add p.space() to consume whitespace between parallel tool calls
            auto tool_call = p.rule("tool-call",
                p.space()
                + "<tool_call>"
                + p.space()
                + tool_choice
                + p.space()
                + "</tool_call>"
                + p.space()
            );
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_call, /* min = */ min_calls, /* max = */ max_calls));

            bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
            if (require_tools) {
                return tool_calls + consume_end_block();
            }
            return p.optional(content_before_tool) + tool_calls + consume_end_block();
        }

        // Content only parser
        include_grammar = false;
        return p.choice({
            content_until("<|im_end|>", /* allow_inline = */ true) << consume_end_block(),
            content_until("<|endoftext|>", /* allow_inline = */ true) << consume_end_block(),
            p.tag(Tag::CONTENT, p.rest())
        });
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        // Build grammar from PEG parser
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            foreach_function(inputs.tools, [&](const json & tool) {
                auto schema = tool.at("function").at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });

        if (data.grammar_lazy) {
            data.grammar_triggers.push_back({COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>"});
        } else {
            data.grammar_triggers.clear();
        }
    }

    return data;
}
