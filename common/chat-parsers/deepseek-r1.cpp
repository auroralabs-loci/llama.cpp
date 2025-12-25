// DeepSeek R1 tool call format
// Format: <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>name
// ```json
// {"arg":"value"}
// ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
// With optional <think>...</think> reasoning blocks

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_deepseek_r1_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    auto prompt = apply(tmpl, inputs);

    // Hacks to fix the official (broken) prompt.
    // It is advisable to use --chat-template-file models/templates/llama-cpp-deepseek-r1.jinja instead,
    // until the official template is fixed.
    if (tmpl.source().find("{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}") != std::string::npos) {
        // Don't leave the chat dangling after tool results
        if (string_ends_with(prompt, "<｜tool▁outputs▁end｜>")) {
            prompt += "<｜end▁of▁sentence｜>";
            if (inputs.add_generation_prompt) {
                prompt += "<｜Assistant｜>";
            }
        }
        // Fix up tool call delta example added by Minja
        prompt = std::regex_replace(
            prompt,
            std::regex("(<｜tool▁call▁end｜>)[\\s\\r\\n]*(<｜tool▁outputs▁begin｜>|<｜User｜>)"),
            "$1<｜tool▁calls▁end｜><｜end▁of▁sentence｜>$2");
    }
    data.prompt = prompt;

    if (string_ends_with(data.prompt, "<think>\n")) {
        if (!inputs.enable_thinking) {
            data.prompt += "</think>";
        } else {
            data.thinking_forced_open = true;
        }
    }

    bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    auto extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    data.format = COMMON_CHAT_FORMAT_DEEPSEEK_R1;
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED && inputs.json_schema.is_null();

    data.preserved_tokens = {
        "<think>",
        "</think>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>",
        "<｜tool▁sep｜>",
        "<｜tool▁call▁end｜>",
        "<｜tool▁calls▁end｜>",
    };

    // Build PEG parser
    bool require_tools = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        auto consume_eos = [&]() {
            return p.optional(p.literal("<｜end▁of▁sentence｜>")) + p.optional(p.space());
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

                // Format: function<｜tool▁sep｜>name\n```json\n{...}\n```<｜tool▁call▁end｜>
                tool_choice |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                    p.optional(p.atomic_tag(Tag::TOOL_OPEN, p.literal("<｜tool▁call▁begin｜>")))
                    + "function" + p.literal("<｜tool▁sep｜>") + p.literal_tag(Tag::TOOL_NAME, name) + "\n```json\n"
                    + p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-args", parameters))
                    + "\n```" + p.atomic_tag(Tag::TOOL_CLOSE, p.literal("<｜tool▁call▁end｜>"))
                ));
            });

            // Accept multiple variants of the tool calls begin marker
            auto tool_calls_begin = p.choice()
                | "<｜tool▁calls▁begin｜>"
                | "<｜tool_calls_begin｜>"
                | "<｜tool calls begin｜>"
                | "<｜tool\\_calls\\_begin｜>"
                | "<｜tool▁calls｜>";

            auto min_calls = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;
            auto tool_calls = p.trigger_rule("tool-call-root",
                tool_calls_begin + p.repeat(tool_choice, min_calls, max_calls) + "<｜tool▁calls▁end｜>"
            ) << consume_eos();

            // Content until tool calls marker
            auto content = p.tag(Tag::CONTENT, p.until_one_of({
                "<｜tool▁calls▁begin｜>",
                "<｜tool_calls_begin｜>",
                "<｜tool calls begin｜>",
                "<｜tool\\_calls\\_begin｜>",
                "<｜tool▁calls｜>",
            }));

            if (require_tools) {
                return reasoning << tool_calls;
            }
            return reasoning << content << tool_calls;
        }

        // Content only parser
        auto content_only = p.sequence({
            p.tag(Tag::CONTENT, p.until("<｜end▁of▁sentence｜>")),
            consume_eos()
        });
        return reasoning << p.choice({content_only, p.tag(Tag::CONTENT, p.rest())});
    });

    data.parser = parser.save();

    if (has_tools) {
        // Build grammar manually for backward compatibility
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> tool_rules;
            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);
                tool_rules.push_back(builder.add_rule(name + "-call",
                    "( \"<｜tool▁call▁begin｜>\" )? \"function<｜tool▁sep｜>" + name + "\\n"
                    "```json\\n\" " + builder.add_schema(name + "-args", parameters) + " "
                    "\"\\n```<｜tool▁call▁end｜>\""));
            });
            // Distill Qwen 7B & 32B models seem confused re/ syntax of their tool call opening tag,
            // so we accept common variants (then it's all constrained)
            builder.add_rule("root",
                std::string(data.thinking_forced_open ? "( \"</think>\" space )? " : "") +
                "( \"<｜tool▁calls▁begin｜>\" | \"<｜tool_calls_begin｜>\" | \"<｜tool calls begin｜>\" | \"<｜tool\\\\_calls\\\\_begin｜>\" | \"<｜tool▁calls｜>\" ) "
                "(" + string_join(tool_rules, " | ") + ")" + (inputs.parallel_tool_calls ? "*" : "") + " "
                "\"<｜tool▁calls▁end｜>\""
                " space");
        });

        if (data.grammar_lazy) {
            data.grammar_triggers.push_back({
                COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                std::string(data.thinking_forced_open ? "[\\s\\S]*?(</think>\\s*)" : "(?:<think>[\\s\\S]*?</think>\\s*)?") +
                    "(<｜tool▁calls▁begin｜>|<｜tool_calls_begin｜>|<｜tool calls begin｜>|<｜tool\\\\_calls\\\\_begin｜>|<｜tool▁calls｜>)[\\s\\S]*"
            });
        } else {
            data.grammar_triggers.clear();
        }
    }

    return data;
}
