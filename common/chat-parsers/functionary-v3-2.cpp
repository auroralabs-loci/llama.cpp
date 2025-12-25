// Functionary v3.2 tool call format
// Format: >>>all\ntext>>>fn1\n{...}>>>fn2\n{...}...
// ALL tool calls use >>> prefix (template generates >>> for every call)
// Python tool can have raw code (without opening {)

#include "chat-parsers-internal.h"

common_chat_params common_chat_params_init_functionary_v3_2_peg(const common_chat_template & tmpl, const struct templates_params & inputs) {
    common_chat_params data;
    data.prompt = apply(tmpl, inputs);
    data.format = COMMON_CHAT_FORMAT_FUNCTIONARY_V3_2;

    auto has_tools = inputs.tools.is_array() && !inputs.tools.empty();
    data.grammar_lazy = inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    // Build PEG parser for >>>function_name\n{...} format
    auto parser = build_chat_peg_parser([&](auto & p) {
        using Tag = common_chat_peg_tag;

        // Response format parser
        if (inputs.json_schema.is_object() && !inputs.json_schema.empty()) {
            return p.tag(Tag::CONTENT, p.schema(p.json(), "response-format", inputs.json_schema));
        }

        // Tool call parser: first tool call has no >>> prefix (it's in the generation prompt),
        // subsequent calls have >>> prefix
        if (has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE) {
            // First tool call: no >>> prefix (since >>> is in generation prompt)
            auto first_tool_call = p.choice();
            // Subsequent tool calls: with >>> prefix
            auto subsequent_tool_call = p.choice();

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");

                if (name == "python") {
                    // Python can have raw code or JSON
                    auto python_args = p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters))
                                     | p.tag(Tag::TOOL_ARGS, p.until(">>>"));
                    // First tool needs empty TOOL_OPEN to create tool call object (>>> is in generation prompt)
                    first_tool_call |= p.rule("tool-first-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, "") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + python_args
                    ));
                    subsequent_tool_call |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, ">>>") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + python_args
                    ));
                } else {
                    // Regular JSON tool
                    auto tool_args = p.tag(Tag::TOOL_ARGS, p.schema(p.json(), "tool-" + name + "-params", parameters));
                    // First tool needs empty TOOL_OPEN to create tool call object (>>> is in generation prompt)
                    first_tool_call |= p.rule("tool-first-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, "") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + tool_args
                    ));
                    subsequent_tool_call |= p.rule("tool-" + name, p.tag(Tag::TOOL,
                        p.literal_tag(Tag::TOOL_OPEN, ">>>") + p.literal_tag(Tag::TOOL_NAME, name) + "\n" + tool_args
                    ));
                }
            });

            // Build pattern: optional content with "all\n" marker, then tool calls
            // Format with content: all\n<content>>>>name\n{...}>>>name2\n{...}
            // Format without content: name\n{...}>>>name2\n{...}
            auto max_calls = inputs.parallel_tool_calls ? -1 : 1;

            // Content marker: "all\n" followed by text until >>>
            auto content_marker = "all\n" + p.tag(Tag::CONTENT, p.until(">>>"));

            // Subsequent tool calls (with >>> prefix)
            auto more_tool_calls = p.repeat(subsequent_tool_call, 0, max_calls > 0 ? max_calls - 1 : -1);

            // Optional trailing content, stop at end tokens
            auto trailing_content = p.optional(p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|start_header_id|>"})));

            // Pattern 1: content marker + tool calls (all with >>> since content ends at >>>)
            auto with_content = p.trigger_rule("tool-with-content", content_marker)
                << p.repeat(subsequent_tool_call, 1, max_calls) << trailing_content;
            // Pattern 2: first tool (no >>>) + subsequent tools (with >>>)
            auto without_content = p.trigger_rule("tool-without-content", first_tool_call)
                << more_tool_calls << trailing_content;

            return with_content | without_content;
        }

        // Content only parser
        // Handle optional "all\n" content marker used by Functionary v3.2
        auto content_with_all = "all\n" + p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|start_header_id|>"}));
        auto content_without_all = p.tag(Tag::CONTENT, p.until_one_of({"<|eot_id|>", "<|start_header_id|>"}));
        return content_with_all | content_without_all;
    });

    data.parser = parser.save();

    if (has_tools) {

        // Build grammar
        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            std::vector<std::string> first_tool_rules;      // Without >>> (first tool, >>> in generation prompt)
            std::vector<std::string> subsequent_tool_rules; // With >>> prefix

            foreach_function(inputs.tools, [&](const json & tool) {
                const auto & function = tool.at("function");
                std::string name = function.at("name");
                auto parameters = function.at("parameters");
                builder.resolve_refs(parameters);

                std::string args_pattern = "[\\s\\S]*";
                auto args_rule = builder.add_schema(name + "-args", parameters);
                if (name == "python") {
                    args_rule = builder.add_rule(name + "-maybe-raw-args", args_rule + " | [^{] .*");
                } else {
                    args_pattern = "\\{" + args_pattern;
                }

                // First tool call: no >>> (it's in the generation prompt)
                auto first_call_rule = builder.add_rule(name + "-first-call", "\"" + name + "\\n\" " + args_rule);
                first_tool_rules.push_back(first_call_rule);

                // Subsequent tool calls: with >>> prefix
                auto call_rule = builder.add_rule(name + "-call", "\">>>\" \"" + name + "\\n\" " + args_rule);
                subsequent_tool_rules.push_back(call_rule);

                data.grammar_triggers.push_back({
                    COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
                    "(?:[\\s\\S]+?>>>)?" + regex_escape(name) + "\n" + args_pattern,
                });
            });

            data.preserved_tokens = {
                "<|end_header_id|>",
            };

            if (!first_tool_rules.empty()) {
                auto first_tool_choice = builder.add_rule("first_tool_call", string_join(first_tool_rules, " | "));
                auto subsequent_tool_choice = builder.add_rule("subsequent_tool_call", string_join(subsequent_tool_rules, " | "));
                if (inputs.parallel_tool_calls) {
                    // First tool (no >>>) + optional subsequent tools (with >>>)
                    builder.add_rule("root", first_tool_choice + " (" + subsequent_tool_choice + " space)*");
                } else {
                    // Single tool only (no >>>)
                    builder.add_rule("root", first_tool_choice + " space");
                }
            }
        });
    }

    return data;
}
