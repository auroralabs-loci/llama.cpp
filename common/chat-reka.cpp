// ABOUTME: Reka Edge chat template handler with tool calling and thinking support.
// ABOUTME: Implements PEG-based parsing for <tool_call>/<tool_response> XML format.

#include "chat-reka.h"

#include "chat-auto-parser.h"
#include "chat-peg-parser.h"
#include "json-schema-to-grammar.h"

static void reka_foreach_function(const json & tools, const std::function<void(const json &)> & fn) {
    for (const auto & tool : tools) {
        if (!tool.contains("type") || tool.at("type") != "function" || !tool.contains("function")) {
            continue;
        }
        fn(tool);
    }
}

static json reka_sort_tools_for_parsing(const json & tools) {
    if (!tools.is_array()) {
        return tools;
    }

    json sorted = tools;
    std::stable_sort(sorted.begin(), sorted.end(), [](const json & a, const json & b) {
        const std::string a_name = a.contains("function") ? a.at("function").value("name", "") : "";
        const std::string b_name = b.contains("function") ? b.at("function").value("name", "") : "";
        return a_name.size() > b_name.size();
    });
    return sorted;
}

common_chat_params common_chat_params_init_reka_edge(const common_chat_template &      tmpl,
                                                     const autoparser::templates_params & inputs) {
    common_chat_params data;

    data.prompt            = common_chat_template_direct_apply(tmpl, inputs);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = true;
    data.preserved_tokens  = {
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<tool_response>",
        "</tool_response>",
        "<sep>",
        "<image>",
        "</image>",
        "<video>",
        "</video>",
        "<REKA_IMG_TOKEN>",
    };

    const bool has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    const bool extract_reasoning = inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
    const bool thinking_forced_open =
        extract_reasoning &&
        inputs.enable_thinking &&
        inputs.add_generation_prompt &&
        (inputs.messages.empty() || inputs.messages.back().value("role", "") != "assistant");
    const bool include_grammar = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    const json parser_tools = has_tools ? reka_sort_tools_for_parsing(inputs.tools) : inputs.tools;

    if (extract_reasoning) {
        data.thinking_start_tag = "<think>";
        data.thinking_end_tag   = "</think>";
    }

    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        const std::string THINK_START = "<think>";
        const std::string THINK_END   = "</think>";
        const std::string TOOL_CALL   = "<tool_call>";

        auto end = p.end();
        auto reasoning = p.eps();

        if (extract_reasoning) {
            auto reasoning_body = p.reasoning(p.until_one_of({ THINK_END, TOOL_CALL }));
            if (thinking_forced_open) {
                reasoning = reasoning_body + p.optional(p.literal(THINK_END)) + p.space();
            } else {
                reasoning = p.optional(p.literal(THINK_START) + reasoning_body + p.optional(p.literal(THINK_END)) + p.space());
            }
        }

        if (!has_tools || inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_NONE) {
            return reasoning + p.content(p.rest()) + end;
        }

        auto single_tool = p.standard_json_tools(
            "<tool_call>",
            "</tool_call>",
            parser_tools,
            /* parallel_tool_calls = */ false,
            /* force_tool_calls    = */ true);

        auto min_calls  = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED ? 1 : 0;
        auto max_calls  = inputs.parallel_tool_calls ? -1 : 1;
        auto tool_calls = p.rule("tool-calls", p.repeat(single_tool + p.space(), min_calls, max_calls));
        auto content    = p.content(p.until(TOOL_CALL));

        return reasoning + content + tool_calls + end;
    });

    data.additional_stops = { "<sep>" };
    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            reka_foreach_function(parser_tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                auto         schema   = function.at("parameters");
                builder.resolve_refs(schema);
            });
            parser.build_grammar(builder, data.grammar_lazy);
        });
        if (inputs.parallel_tool_calls) {
            string_replace_all(
                data.grammar,
                "root ::= tool-call\n",
                "root ::= tool-call (space tool-call)*\n");
        }

        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_WORD, "<tool_call>" }
        };
    }

    return data;
}
