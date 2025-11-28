// run-chat.cpp - Console/chat mode functionality for llama-run
//
// This file contains the implementation of interactive chat mode and signal handling.

#include "run-chat.h"
#include "server-context.h"
#include "readline/readline.h"

#include <atomic>
#include <csignal>

#if defined(_WIN32)
#include <windows.h>
#endif

// Static globals for signal handling
static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, force terminating...\n");
        exit(1);
    }

    shutdown_handler(signal);
}

void setup_signal_handlers(std::function<void(int)> handler) {
    shutdown_handler = handler;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

void run_chat_mode(const common_params & params, server_context & ctx_server) {
    // Initialize readline
    readline::Prompt prompt_config;
    prompt_config.prompt = "> ";
    prompt_config.alt_prompt = ". ";
    prompt_config.placeholder = "Send a message";
    readline::Readline rl(prompt_config);
    rl.history_enable();

    // Message history
    std::vector<json> messages;
    while (true) {
        // Read user input
        std::string user_input;
        try {
            user_input = rl.readline();
        } catch (const readline::eof_error&) {
            printf("\n");
            break;
        } catch (const readline::interrupt_error&) {
            printf("\nUse Ctrl + d or /bye to exit.\n");
            continue;
        }

        if (user_input.empty()) {
            continue;
        }

        if (user_input == "/bye") {
            break;
        }

        // Add user message to history
        json user_msg = {
            {"role", "user"},
            {"content", user_input}
        };
        messages.push_back(user_msg);

        // Apply chat template to format the prompt
        // Convert json messages to llama_chat_message array
        std::vector<llama_chat_message> chat_msgs;
        std::vector<std::string> role_strs;
        std::vector<std::string> content_strs;

        for (const auto & msg : messages) {
            role_strs.push_back(msg["role"].get<std::string>());
            content_strs.push_back(msg["content"].get<std::string>());
        }

        for (size_t i = 0; i < messages.size(); i++) {
            chat_msgs.push_back({role_strs[i].c_str(), content_strs[i].c_str()});
        }

        // Get the chat template
        const char * tmpl = llama_model_chat_template(ctx_server.model, nullptr);

        // First call to get the required buffer size
        std::string formatted_prompt;
        int32_t res = llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true, nullptr, 0);
        if (res < 0) {
            LOG_ERR("%s: failed to apply chat template\n", __func__);
            messages.pop_back(); // Remove the user message we just added
            continue;
        }

        // Allocate buffer and apply template
        formatted_prompt.resize(res);
        res = llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true, &formatted_prompt[0], res + 1);
        if (res < 0) {
            LOG_ERR("%s: failed to apply chat template\n", __func__);
            messages.pop_back(); // Remove the user message we just added
            continue;
        }

        // Tokenize the formatted prompt
        std::vector<llama_token> tokens = common_tokenize(ctx_server.ctx, formatted_prompt, true, false);

        // Create completion task
        server_task task;
        task.type = SERVER_TASK_TYPE_COMPLETION;
        task.params.stream = true;
        task.params.cache_prompt = true;
        task.params.n_predict = params.n_predict;
        task.params.sampling = params.sampling;
        task.tokens = server_tokens(tokens, false);

        // Submit task
        const int task_id = ctx_server.queue_tasks.get_new_id();
        task.id = task_id;
        ctx_server.queue_results.add_waiting_task_id(task_id);
        ctx_server.queue_tasks.post(std::move(task), true);

        // Receive and stream results
        std::string assistant_response;
        bool has_error = false;
        bool interrupted = false;

        while (true) {
            // Check for interrupt (Ctrl-C) during streaming
            if (rl.check_interrupt()) {
                printf("\n");
                interrupted = true;
                break;
            }

            // Use timeout to allow periodic interrupt checks
            server_task_result_ptr result = ctx_server.queue_results.recv_with_timeout({task_id}, 1);

            if (result == nullptr) {
                // Timeout or no result - continue checking for interrupt
                continue;
            }

            if (result->is_error()) {
                LOG_ERR("%s: error in response\n", __func__);
                has_error = true;
                break;
            }

            // Check if this is a partial result (streaming chunk)
            auto * cmpl_partial = dynamic_cast<server_task_result_cmpl_partial*>(result.get());
            if (cmpl_partial) {
                // Print the partial content without newline
                printf("%s", cmpl_partial->content.c_str());
                fflush(stdout);
                assistant_response += cmpl_partial->content;
                continue;
            }

            // Check if this is the final result
            auto * cmpl_final = dynamic_cast<server_task_result_cmpl_final*>(result.get());
            if (cmpl_final) {
                // Print any remaining content and add newline
                if (!cmpl_final->content.empty()) {
                    printf("%s", cmpl_final->content.c_str());
                    fflush(stdout);
                    assistant_response += cmpl_final->content;
                }
                printf("\n");
                break;
            }
        }

        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (interrupted) {
            messages.pop_back(); // Remove the user message since generation was interrupted
        } else if (!has_error && !assistant_response.empty()) {
            // Add assistant message to history
            json assistant_msg = {
                {"role", "assistant"},
                {"content", assistant_response}
            };
            messages.push_back(assistant_msg);
        } else if (has_error) {
            messages.pop_back(); // Remove the user message
        }
    }

    LOG_INF("%s: exiting chat mode\n", __func__);
}
