// Hermes 2 Pro tool call format
// Formats:
// - <tool_call>{"name":"func","arguments":{}}</tool_call>
// - <function=name>{"key":"value"}</function>
// - <function name="name">{"key":"value"}</function>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_hermes_2_pro_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;

    json extra_context = json {
        {"enable_thinking", inputs.enable_thinking},
    };
    extra_context.update(inputs.extra_context);

    data.prompt = apply(tmpl, inputs, /* messages_override =*/ std::nullopt, /* tools_override= */ std::nullopt, extra_context);

    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!extra_context["enable_thinking"]) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.format = COMMON_CHAT_FORMAT_HERMES_2_PRO;
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<function",
        "<tools>",
        "</tools>",
        "<response>",
        "</response>",
        "<function_call>",
        "</function_call>",
        "<json>",
        "</json>",
        "<JSON>",
        "</JSON>",
        "```",
        "```json",
        "```xml",
    };

    // Build PEG parser
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto consume_message_end = [&]() {
            return p.optional(p.choice({p.literal("<|im_end|>"), p.literal("<|eot_id|>"), p.literal("<|eom_id|>")}))
                + p.optional(p.space());
        };

        // Optional thinking block
        auto reasoning = p.eps();
        if (extract_reasoning) {
            if (data.thinking_forced_open) {
                reasoning = p.tag(Tag::REASONING, p.until("</think>")) + "</think>";
            } else {
                reasoning = p.optional("<think>" + p.tag(Tag::REASONING, p.until("</think>")) + "</think>");
            }
        }

        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            auto tool_choice = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");

                // <tool_call>{"name":"func","arguments":{}}</tool_call>
                tool_choice |= p.rule("tool-call-" + name, p.tag(Tag::TOOL,
                    p.atomic_tag(Tag::TOOL_OPEN, p.literal("<tool_call>"))
                    + p.space()
                    + "{" + p.space()
                    + "\"name\"" + p.space() + ":" + p.space()
                    + "\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\"" + p.space() + "," + p.space()
                    + "\"arguments\"" + p.space() + ":" + p.space()
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    + p.space() + "}"
                    + p.space()
                    + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</tool_call>"))
                ) + p.space());

                // <function=name>{...}</function>
                tool_choice |= p.rule("func-eq-" + name, p.tag(Tag::TOOL,
                    p.atomic_tag(Tag::TOOL_OPEN, "<function=" + p.literal_tag(Tag::TOOL_NAME, name) + ">")
                    + p.space()
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "func-" + name + "-args", parameters))
                    + p.space()
                    + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</function>"))
                ) + p.space());

                // <function name="name">{...}</function>
                tool_choice |= p.rule("func-name-" + name, p.tag(Tag::TOOL,
                    p.atomic_tag(Tag::TOOL_OPEN, "<function" + p.space() + "name=\"" + p.literal_tag(Tag::TOOL_NAME, name) + "\">")
                    + p.space()
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "funcn-" + name + "-args", parameters))
                    + p.space()
                    + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("</function>"))
                ) + p.space());
            });

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root", p.repeat(tool_choice, min_calls, max_calls));

            auto content_prefix = p.optional(p.tag(Tag::CONTENT, p.until_one_of({
                "<tool_call>",
                "<function",
            })));

            return reasoning << content_prefix << tool_calls << consume_message_end();
        }

        // Content only parser
        auto content_block = p.sequence({
            p.tag(Tag::CONTENT, p.until("<|im_end|>")),
            consume_message_end()
        });
        return reasoning << p.choice({content_block, p.tag(Tag::CONTENT, p.rest()), p.eps()});
    });

    data.parser = parser.save();

    if (has_tools) {
        // Build grammar manually for backward compatibility with streaming tests
        // (using regular string literals instead of token syntax)
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            std::vector<std::string> tool_call_alts;
            std::vector<std::string> escaped_names;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_schema(name + "-call", {
                    {"type", "object"},
                    {"properties", json {
                        {"name", json {{"const", name}}},
                        {"arguments", parameters},
                    }},
                    {"required", json::array({"name", "arguments"})},
                }));
                tool_call_alts.push_back(builder.add_rule(
                    name + "-function-tag",
                    "\"<function\" ( \"=" + name + "\" | \" name=\\\"" + name + "\\\"\" ) \">\" space " +
                    builder.add_schema(name + "-args", parameters) + " "
                    "\"</function>\" space"));

                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_WORD,
                    "<function=" + name + ">",
                });
                escaped_names.push_back(regex_escape(name));
                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                    "<function\\s+name\\s*=\\s*\"" + regex_escape(name) + "\"",
                });
            });
            auto any_tool_call = builder.add_rule("any_tool_call", "( " + string_join(tool_rules, " | ") + " ) space");
            std::vector<std::string> alt_tags {
                any_tool_call,
                "\"<tool_call>\" space "     + any_tool_call + " \"</tool_call>\"",
                // The rest is just to accommodate common "good bad" outputs.
                "\"<function_call>\" space " + any_tool_call + " \"</function_call>\"",
                "\"<response>\"  space "     + any_tool_call + " \"</response>\"",
                "\"<tools>\"     space "     + any_tool_call + " \"</tools>\"",
                "\"<json>\"      space "     + any_tool_call + " \"</json>\"",
                "\"<xml>\"      space "     + any_tool_call + " \"</xml>\"",
                "\"<JSON>\"      space "     + any_tool_call + " \"</JSON>\"",
            };
            auto wrappable_tool_call = builder.add_rule("wrappable_tool_call", "( " + string_join(alt_tags, " | ") + " ) space");
            tool_call_alts.push_back(wrappable_tool_call);
            tool_call_alts.push_back(
                "( \"```\\n\" | \"```json\\n\" | \"```xml\\n\" ) space " + wrappable_tool_call + " space \"```\" space ");
            auto tool_call = builder.add_rule("tool_call", string_join(tool_call_alts, " | "));
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                (inputs.parallel_tool_calls ? "(" + tool_call + ")+" : tool_call));
            // Trigger on some common known "good bad" outputs
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") + (
                    "\\s*("
                    "(?:<tool_call>"
                    "|<function"
                    "|(?:```(?:json|xml)?\n\\s*)?(?:<function_call>|<tools>|<xml><json>|<response>)?"
                    "\\s*\\{\\s*\"name\"\\s*:\\s*\"(?:" + string_join(escaped_names, "|") + ")\""
                    ")"
                    ")[\\s\\S]*"
                ),
            });
        });
    }

    return data;
}
