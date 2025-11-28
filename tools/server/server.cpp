#include "server-common.h"
#include "server-http.h"
#include "server-task.h"
#include "server-queue.h"
#include "server-context.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"
#include "sampling.h"
#include "speculative.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <atomic>
#include <cstddef>
#include <cinttypes>
#include <memory>
#include <signal.h>
#include <thread>
#include <unordered_set>

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

using json = nlohmann::ordered_json;

constexpr int HTTP_POLLING_SECONDS = 1;

// Signal handler globals
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

static void setup_signal_handlers(std::function<void(int)> handler) {
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

// generator-like API for server responses, support pooling connection state and aggregating results
struct server_response_reader {
    std::unordered_set<int> id_tasks;
    server_context & ctx_server;
    size_t received_count = 0;
    bool cancelled = false;

    server_response_reader(server_context & ctx_server) : ctx_server(ctx_server) {}
    ~server_response_reader() {
        stop();
    }

    void post_tasks(std::vector<server_task> && tasks) {
        id_tasks = server_task::get_list_id(tasks);
        ctx_server.queue_results.add_waiting_tasks(tasks);
        ctx_server.queue_tasks.post(std::move(tasks));
    }

    bool has_next() const {
        return !cancelled && received_count < id_tasks.size();
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            server_task_result_ptr result = ctx_server.queue_results.recv_with_timeout(id_tasks, HTTP_POLLING_SECONDS);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    SRV_DBG("%s", "stopping wait for next result due to should_stop condition\n");
                    return nullptr;
                }
            } else {
                if (result->is_error()) {
                    stop(); // cancel remaining tasks
                    SRV_DBG("%s", "received error result, stopping further processing\n");
                    return result;
                }
                if (result->is_stop()) {
                    received_count++;
                }
                return result;
            }
        }

        // should not reach here
    }

    struct batch_response {
        bool is_terminated = false; // if true, indicates that processing was stopped before all results were received
        std::vector<server_task_result_ptr> results;
        server_task_result_ptr error; // nullptr if no error
    };

    batch_response wait_for_all(const std::function<bool()> & should_stop) {
        batch_response batch_res;
        batch_res.results.resize(id_tasks.size());
        while (has_next()) {
            auto res = next(should_stop);
            if (res == nullptr) {
                batch_res.is_terminated = true;
                return batch_res;
            }
            if (res->is_error()) {
                batch_res.error = std::move(res);
                return batch_res;
            }
            const size_t idx = res->get_index();
            GGML_ASSERT(idx < batch_res.results.size() && "index out of range");
            GGML_ASSERT(batch_res.results[idx] == nullptr && "duplicate result received");
            batch_res.results[idx] = std::move(res);
        }
        return batch_res;
    }

    void stop() {
        ctx_server.queue_results.remove_waiting_task_ids(id_tasks);
        if (has_next() && !cancelled) {
            // if tasks is not finished yet, cancel them
            cancelled = true;
            std::vector<server_task> cancel_tasks;
            cancel_tasks.reserve(id_tasks.size());
            for (const auto & id_task : id_tasks) {
                SRV_WRN("cancel task, id_task = %d\n", id_task);
                server_task task(SERVER_TASK_TYPE_CANCEL);
                task.id_target = id_task;
                ctx_server.queue_results.remove_waiting_task_id(id_task);
                cancel_tasks.push_back(std::move(task));
            }
            // push to beginning of the queue, so it has highest priority
            ctx_server.queue_tasks.post(std::move(cancel_tasks), true);
        } else {
            SRV_DBG("%s", "all tasks already finished, no need to cancel\n");
        }
    }
};

// generator-like API for HTTP response generation
struct server_res_generator : server_http_res {
    server_response_reader rd;
    server_res_generator(server_context & ctx_server_) : rd(ctx_server_) {}
    void ok(const json & response_data) {
        status = 200;
        data = safe_json_to_str(response_data);
    }
    void error(const json & error_data) {
        status = json_value(error_data, "code", 500);
        data = safe_json_to_str({{ "error", error_data }});
    }
};

struct server_routes {
    const common_params & params;
    server_context & ctx_server;
    server_http_context & ctx_http; // for reading is_ready
    server_routes(const common_params & params, server_context & ctx_server, server_http_context & ctx_http)
        : params(params), ctx_server(ctx_server), ctx_http(ctx_http) {}

public:
    // handlers using lambda function, so that they can capture `this` without `std::bind`

    server_http_context::handler_t get_health = [this](const server_http_req &) {
        // error and loading states are handled by middleware
        auto res = std::make_unique<server_res_generator>(ctx_server);
        res->ok({{"status", "ok"}});
        return res;
    };

    server_http_context::handler_t get_metrics = [this](const server_http_req &) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        if (!params.endpoint_metrics) {
            res->error(format_error_response("This server does not support metrics endpoint. Start it with `--metrics`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        // TODO: use server_response_reader
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = task_id;
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task), true); // high-priority task
        }

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
        json all_metrics_def = json {
            {"counter", {{
                    {"name",  "prompt_tokens_total"},
                    {"help",  "Number of prompt tokens processed."},
                    {"value",  (uint64_t) res_task->n_prompt_tokens_processed_total}
            }, {
                    {"name",  "prompt_seconds_total"},
                    {"help",  "Prompt process time"},
                    {"value",  (uint64_t) res_task->t_prompt_processing_total / 1.e3}
            }, {
                    {"name",  "tokens_predicted_total"},
                    {"help",  "Number of generation tokens processed."},
                    {"value",  (uint64_t) res_task->n_tokens_predicted_total}
            }, {
                    {"name",  "tokens_predicted_seconds_total"},
                    {"help",  "Predict process time"},
                    {"value",  (uint64_t) res_task->t_tokens_generation_total / 1.e3}
            }, {
                    {"name",  "n_decode_total"},
                    {"help",  "Total number of llama_decode() calls"},
                    {"value",  res_task->n_decode_total}
            }, {
                    {"name",  "n_tokens_max"},
                    {"help",  "Largest observed n_tokens."},
                    {"value",  res_task->n_tokens_max}
            }, {
                    {"name",  "n_busy_slots_per_decode"},
                    {"help",  "Average number of busy slots per llama_decode() call"},
                    {"value",  (float) res_task->n_busy_slots_total / (std::max)((float) res_task->n_decode_total, 1.f)}
            }}},
            {"gauge", {{
                    {"name",  "prompt_tokens_seconds"},
                    {"help",  "Average prompt throughput in tokens/s."},
                    {"value",  res_task->n_prompt_tokens_processed ? 1.e3 / res_task->t_prompt_processing * res_task->n_prompt_tokens_processed : 0.}
            },{
                    {"name",  "predicted_tokens_seconds"},
                    {"help",  "Average generation throughput in tokens/s."},
                    {"value",  res_task->n_tokens_predicted ? 1.e3 / res_task->t_tokens_generation * res_task->n_tokens_predicted : 0.}
            },{
                    {"name",  "requests_processing"},
                    {"help",  "Number of requests processing."},
                    {"value",  (uint64_t) res_task->n_processing_slots}
            },{
                    {"name",  "requests_deferred"},
                    {"help",  "Number of requests deferred."},
                    {"value",  (uint64_t) res_task->n_tasks_deferred}
            }}}
        };

        std::stringstream prometheus;

        for (const auto & el : all_metrics_def.items()) {
            const auto & type        = el.key();
            const auto & metrics_def = el.value();

            for (const auto & metric_def : metrics_def) {
                const std::string name = metric_def.at("name");
                const std::string help = metric_def.at("help");

                auto value = json_value(metric_def, "value", 0.);
                prometheus << "# HELP llamacpp:" << name << " " << help  << "\n"
                            << "# TYPE llamacpp:" << name << " " << type  << "\n"
                            << "llamacpp:"        << name << " " << value << "\n";
            }
        }

        res->headers["Process-Start-Time-Unix"] = std::to_string(res_task->t_start);
        res->content_type = "text/plain; version=0.0.4";
        res->ok(prometheus.str());
        return res;
    };

    server_http_context::handler_t get_slots = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        if (!params.endpoint_slots) {
            res->error(format_error_response("This server does not support slots endpoint. Start it with `--slots`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // request slots data using task queue
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_METRICS);
            task.id = task_id;
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task), true); // high-priority task
        }

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        // TODO: get rid of this dynamic_cast
        auto res_task = dynamic_cast<server_task_result_metrics*>(result.get());
        GGML_ASSERT(res_task != nullptr);

        // optionally return "fail_on_no_slot" error
        if (!req.get_param("fail_on_no_slot").empty()) {
            if (res_task->n_idle_slots == 0) {
                res->error(format_error_response("no slot available", ERROR_TYPE_UNAVAILABLE));
                return res;
            }
        }

        res->ok(res_task->slots_data);
        return res;
    };

    server_http_context::handler_t post_slots = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        if (params.slot_save_path.empty()) {
            res->error(format_error_response("This server does not support slots action. Start it with `--slot-save-path`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        std::string id_slot_str = req.get_param("id_slot");
        int id_slot;

        try {
            id_slot = std::stoi(id_slot_str);
        } catch (const std::exception &) {
            res->error(format_error_response("Invalid slot ID", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::string action = req.get_param("action");

        if (action == "save") {
            return handle_slots_save(req, id_slot);
        } else if (action == "restore") {
            return handle_slots_restore(req, id_slot);
        } else if (action == "erase") {
            return handle_slots_erase(req, id_slot);
        } else {
            res->error(format_error_response("Invalid action", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
    };

    server_http_context::handler_t get_props = [this](const server_http_req &) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        json default_generation_settings_for_props;

        {
            task_params params;

            params.sampling = ctx_server.params_base.sampling;

            default_generation_settings_for_props = json {
                {"params", params.to_json(true)},
                {"n_ctx",  ctx_server.slots[0].n_ctx},
            };
        }

        // this endpoint is publicly available, please only return what is safe to be exposed
        json data = {
            { "default_generation_settings", default_generation_settings_for_props },
            { "total_slots",                 ctx_server.params_base.n_parallel },
            { "model_alias",                 ctx_server.params_base.model_alias },
            { "model_path",                  ctx_server.params_base.model.path },
            { "modalities",                  json {
                {"vision", ctx_server.oai_parser_opt.allow_image},
                {"audio",  ctx_server.oai_parser_opt.allow_audio},
            } },
            { "endpoint_slots",              params.endpoint_slots },
            { "endpoint_props",              params.endpoint_props },
            { "endpoint_metrics",            params.endpoint_metrics },
            { "webui",                       params.webui },
            { "chat_template",               common_chat_templates_source(ctx_server.chat_templates.get()) },
            { "bos_token",                   common_token_to_piece(ctx_server.ctx, llama_vocab_bos(ctx_server.vocab), /* special= */ true)},
            { "eos_token",                   common_token_to_piece(ctx_server.ctx, llama_vocab_eos(ctx_server.vocab), /* special= */ true)},
            { "build_info",                  build_info },
        };
        if (ctx_server.params_base.use_jinja) {
            if (auto tool_use_src = common_chat_templates_source(ctx_server.chat_templates.get(), "tool_use")) {
                data["chat_template_tool_use"] = tool_use_src;
            }
        }

        res->ok(data);
        return res;
    };

    server_http_context::handler_t post_props = [this](const server_http_req &) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        if (!params.endpoint_props) {
            res->error(format_error_response("This server does not support changing global properties. Start it with `--props`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }
        // update any props here

        res->ok({{ "success", true }});
        return res;
    };

    server_http_context::handler_t get_api_show = [this](const server_http_req &) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        bool has_mtmd = ctx_server.mctx != nullptr;
        json data = {
            {
                "template", common_chat_templates_source(ctx_server.chat_templates.get()),
            },
            {
                "model_info", {
                    { "llama.context_length", ctx_server.slots.back().n_ctx, },
                }
            },
            {"modelfile", ""},
            {"parameters", ""},
            {"template", common_chat_templates_source(ctx_server.chat_templates.get())},
            {"details", {
                {"parent_model", ""},
                {"format", "gguf"},
                {"family", ""},
                {"families", {""}},
                {"parameter_size", ""},
                {"quantization_level", ""}
            }},
            {"model_info", ""},
            {"capabilities", has_mtmd ? json({"completion","multimodal"}) : json({"completion"})}
        };

        res->ok(data);
        return res;
    };

    server_http_context::handler_t post_infill = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        // check model compatibility
        std::string err;
        if (llama_vocab_fim_pre(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "prefix token is missing. ";
        }
        if (llama_vocab_fim_suf(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "suffix token is missing. ";
        }
        if (llama_vocab_fim_mid(ctx_server.vocab) == LLAMA_TOKEN_NULL) {
            err += "middle token is missing. ";
        }
        if (!err.empty()) {
            res->error(format_error_response(string_format("Infill is not supported by this model: %s", err.c_str()), ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        // validate input
        json data = json::parse(req.body);
        if (data.contains("prompt") && !data.at("prompt").is_string()) {
            // prompt is optional
            res->error(format_error_response("\"prompt\" must be a string", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_prefix")) {
            res->error(format_error_response("\"input_prefix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (!data.contains("input_suffix")) {
            res->error(format_error_response("\"input_suffix\" is required", ERROR_TYPE_INVALID_REQUEST));
        }

        if (data.contains("input_extra") && !data.at("input_extra").is_array()) {
            // input_extra is optional
            res->error(format_error_response("\"input_extra\" must be an array of {\"filename\": string, \"text\": string}", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        json input_extra = json_value(data, "input_extra", json::array());
        for (const auto & chunk : input_extra) {
            // { "text": string, "filename": string }
            if (!chunk.contains("text") || !chunk.at("text").is_string()) {
                res->error(format_error_response("extra_context chunk must contain a \"text\" field with a string value", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
            // filename is optional
            if (chunk.contains("filename") && !chunk.at("filename").is_string()) {
                res->error(format_error_response("extra_context chunk's \"filename\" field must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }
        data["input_extra"] = input_extra; // default to empty array if it's not exist

        std::string prompt = json_value(data, "prompt", std::string());
        std::vector<server_tokens> tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, false, true);
        SRV_DBG("creating infill tasks, n_prompts = %d\n", (int) tokenized_prompts.size());
        data["prompt"] = format_prompt_infill(
            ctx_server.vocab,
            data.at("input_prefix"),
            data.at("input_suffix"),
            data.at("input_extra"),
            ctx_server.params_base.n_batch,
            ctx_server.params_base.n_predict,
            ctx_server.slots[0].n_ctx, // TODO: there should be a better way
            ctx_server.params_base.spm_infill,
            tokenized_prompts[0].get_text_tokens() // TODO: this could maybe be multimodal.
        );

        std::vector<raw_buffer> files; // dummy
        return handle_completions_impl(
            SERVER_TASK_TYPE_INFILL,
            data,
            files,
            req.should_stop,
            TASK_RESPONSE_TYPE_NONE); // infill is not OAI compatible
    };

    server_http_context::handler_t post_completions = [this](const server_http_req & req) {
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            req.should_stop,
            TASK_RESPONSE_TYPE_NONE);
    };

    server_http_context::handler_t post_completions_oai = [this](const server_http_req & req) {
        std::vector<raw_buffer> files; // dummy
        const json body = json::parse(req.body);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            body,
            files,
            req.should_stop,
            TASK_RESPONSE_TYPE_OAI_CMPL);
    };

    server_http_context::handler_t post_chat_completions = [this](const server_http_req & req) {
        std::vector<raw_buffer> files;
        json body = json::parse(req.body);
        json body_parsed = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            req.should_stop,
            TASK_RESPONSE_TYPE_OAI_CHAT);
    };

    server_http_context::handler_t post_anthropic_messages = [this](const server_http_req & req) {
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);
        return handle_completions_impl(
            SERVER_TASK_TYPE_COMPLETION,
            body_parsed,
            files,
            req.should_stop,
            TASK_RESPONSE_TYPE_ANTHROPIC);
    };

    server_http_context::handler_t post_anthropic_count_tokens = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        std::vector<raw_buffer> files;
        json body = convert_anthropic_to_oai(json::parse(req.body));
        json body_parsed = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);

        json prompt = body_parsed.at("prompt");
        llama_tokens tokens = tokenize_mixed(ctx_server.vocab, prompt, true, true);

        res->ok({{"input_tokens", static_cast<int>(tokens.size())}});
        return res;
    };

    // same with handle_chat_completions, but without inference part
    server_http_context::handler_t post_apply_template = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        std::vector<raw_buffer> files; // dummy, unused
        json body = json::parse(req.body);
        json data = oaicompat_chat_params_parse(
            body,
            ctx_server.oai_parser_opt,
            files);
        res->ok({{ "prompt", std::move(data.at("prompt")) }});
        return res;
    };

    server_http_context::handler_t get_models = [this](const server_http_req &) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        bool is_model_ready = ctx_http.is_ready.load();
        json model_meta = nullptr;
        if (is_model_ready) {
            model_meta = ctx_server.model_meta();
        }
        bool has_mtmd = ctx_server.mctx != nullptr;
        json models = {
            {"models", {
                {
                    {"name", params.model_alias.empty() ? params.model.path : params.model_alias},
                    {"model", params.model_alias.empty() ? params.model.path : params.model_alias},
                    {"modified_at", ""},
                    {"size", ""},
                    {"digest", ""}, // dummy value, llama.cpp does not support managing model file's hash
                    {"type", "model"},
                    {"description", ""},
                    {"tags", {""}},
                    {"capabilities", has_mtmd ? json({"completion","multimodal"}) : json({"completion"})},
                    {"parameters", ""},
                    {"details", {
                        {"parent_model", ""},
                        {"format", "gguf"},
                        {"family", ""},
                        {"families", {""}},
                        {"parameter_size", ""},
                        {"quantization_level", ""}
                    }}
                }
            }},
            {"object", "list"},
            {"data", {
                {
                    {"id",       params.model_alias.empty() ? params.model.path : params.model_alias},
                    {"object",   "model"},
                    {"created",  std::time(0)},
                    {"owned_by", "llamacpp"},
                    {"meta",     model_meta},
                },
            }}
        };

        res->ok(models);
        return res;
    };

    server_http_context::handler_t post_tokenize = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        const json body = json::parse(req.body);
        json tokens_response = json::array();
        if (body.count("content") != 0) {
            const bool add_special = json_value(body, "add_special", false);
            const bool parse_special = json_value(body, "parse_special", true);
            const bool with_pieces = json_value(body, "with_pieces", false);

            llama_tokens tokens = tokenize_mixed(ctx_server.vocab, body.at("content"), add_special, parse_special);

            if (with_pieces) {
                for (const auto& token : tokens) {
                    std::string piece = common_token_to_piece(ctx_server.ctx, token);
                    json piece_json;

                    // Check if the piece is valid UTF-8
                    if (is_valid_utf8(piece)) {
                        piece_json = piece;
                    } else {
                        // If not valid UTF-8, store as array of byte values
                        piece_json = json::array();
                        for (unsigned char c : piece) {
                            piece_json.push_back(static_cast<int>(c));
                        }
                    }

                    tokens_response.push_back({
                        {"id", token},
                        {"piece", piece_json}
                    });
                }
            } else {
                tokens_response = tokens;
            }
        }

        res->ok(json{{"tokens", std::move(tokens_response)}});
        return res;
    };

    server_http_context::handler_t post_detokenize = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        const json body = json::parse(req.body);

        std::string content;
        if (body.count("tokens") != 0) {
            const llama_tokens tokens = body.at("tokens");
            content = tokens_to_str(ctx_server.ctx, tokens);
        }

        res->ok(json{{"content", std::move(content)}});
        return res;
    };

    server_http_context::handler_t post_embeddings = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_NONE);
    };

    server_http_context::handler_t post_embeddings_oai = [this](const server_http_req & req) {
        return handle_embeddings_impl(req, TASK_RESPONSE_TYPE_OAI_EMBD);
    };

    server_http_context::handler_t post_rerank = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        if (!ctx_server.params_base.embedding || ctx_server.params_base.pooling_type != LLAMA_POOLING_TYPE_RANK) {
            res->error(format_error_response("This server does not support reranking. Start it with `--reranking`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        const json body = json::parse(req.body);

        // if true, use TEI API format, otherwise use Jina API format
        // Jina: https://jina.ai/reranker/
        // TEI: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
        bool is_tei_format = body.contains("texts");

        json query;
        if (body.count("query") == 1) {
            query = body.at("query");
            if (!query.is_string()) {
                res->error(format_error_response("\"query\" must be a string", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        } else {
            res->error(format_error_response("\"query\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        std::vector<std::string> documents = json_value(body, "documents",
                                             json_value(body, "texts", std::vector<std::string>()));
        if (documents.empty()) {
            res->error(format_error_response("\"documents\" must be a non-empty string array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int top_n = json_value(body, "top_n", (int)documents.size());

        // create and queue the task
        json responses = json::array();
        server_response_reader rd(ctx_server);
        {
            std::vector<server_task> tasks;
            tasks.reserve(documents.size());
            for (size_t i = 0; i < documents.size(); i++) {
                auto tmp = format_prompt_rerank(ctx_server.model, ctx_server.vocab, ctx_server.mctx, query, documents[i]);
                server_task task = server_task(SERVER_TASK_TYPE_RERANK);
                task.id     = ctx_server.queue_tasks.get_new_id();
                task.index  = i;
                task.tokens = std::move(tmp);
                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_rerank*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = format_response_rerank(
            body,
            responses,
            is_tei_format,
            documents,
            top_n);

        res->ok(root);
        return res;
    };

    server_http_context::handler_t get_lora_adapters = [this](const server_http_req &) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        json result = json::array();
        const auto & loras = ctx_server.params_base.lora_adapters;
        for (size_t i = 0; i < loras.size(); ++i) {
            auto & lora = loras[i];
            json entry = {
                {"id", i},
                {"path", lora.path},
                {"scale", lora.scale},
                {"task_name", lora.task_name},
                {"prompt_prefix", lora.prompt_prefix},
            };
            std::string alora_invocation_string = "";
            const uint64_t n_alora_tokens = llama_adapter_get_alora_n_invocation_tokens(lora.ptr);
            std::vector<llama_token> alora_invocation_tokens;
            if (n_alora_tokens) {
                const llama_token * alora_tokens = llama_adapter_get_alora_invocation_tokens(lora.ptr);
                for (uint64_t i = 0; i < n_alora_tokens; ++i) {
                    alora_invocation_string += common_token_to_piece(ctx_server.ctx, alora_tokens[i]);
                    alora_invocation_tokens.push_back(alora_tokens[i]);
                }
                entry["alora_invocation_string"] = alora_invocation_string;
                entry["alora_invocation_tokens"] = alora_invocation_tokens;
            }
            result.push_back(std::move(entry));
        }
        res->ok(result);
        return res;
    };

    server_http_context::handler_t post_lora_adapters = [this](const server_http_req & req) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        const json body = json::parse(req.body);
        if (!body.is_array()) {
            res->error(format_error_response("Request body must be an array", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SET_LORA);
            task.id = task_id;
            task.set_lora = parse_lora_request(ctx_server.params_base.lora_adapters, body);
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        // get the result
        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_apply_lora*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    };

private:
    std::unique_ptr<server_res_generator> handle_completions_impl(
                server_task_type type,
                const json & data,
                const std::vector<raw_buffer> & files,
                const std::function<bool()> & should_stop,
                task_response_type res_type) {
        GGML_ASSERT(type == SERVER_TASK_TYPE_COMPLETION || type == SERVER_TASK_TYPE_INFILL);

        auto res = std::make_unique<server_res_generator>(ctx_server);
        auto completion_id = gen_chatcmplid();
        auto & rd = res->rd;

        try {
            std::vector<server_task> tasks;

            const auto & prompt = data.at("prompt");
            // TODO: this log can become very long, put it behind a flag or think about a more compact format
            //SRV_DBG("Prompt: %s\n", prompt.is_string() ? prompt.get<std::string>().c_str() : prompt.dump(2).c_str());

            // process prompt
            std::vector<server_tokens> inputs;

            if (res_type != TASK_RESPONSE_TYPE_NONE && ctx_server.mctx != nullptr) {
                // This is the case used by OAI compatible chat path with MTMD. TODO It can be moved to the path below.
                inputs.push_back(process_mtmd_prompt(ctx_server.mctx, prompt.get<std::string>(), files));
            } else {
                // Everything else, including multimodal completions.
                inputs = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
            }
            tasks.reserve(inputs.size());
            for (size_t i = 0; i < inputs.size(); i++) {
                server_task task = server_task(type);

                task.id    = ctx_server.queue_tasks.get_new_id();
                task.index = i;

                task.tokens = std::move(inputs[i]);
                task.params = server_task::params_from_json_cmpl(
                        ctx_server.ctx,
                        ctx_server.params_base,
                        data);
                task.id_slot = json_value(data, "id_slot", -1);

                // OAI-compat
                task.params.res_type          = res_type;
                task.params.oaicompat_cmpl_id = completion_id;
                // oaicompat_model is already populated by params_from_json_cmpl

                tasks.push_back(std::move(task));
            }

            rd.post_tasks(std::move(tasks));
        } catch (const std::exception & e) {
            res->error(format_error_response(e.what(), ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        bool stream = json_value(data, "stream", false);

        if (!stream) {
            // non-stream, wait for the results
            auto all_results = rd.wait_for_all(should_stop);
            if (all_results.is_terminated) {
                return res; // connection is closed
            } else if (all_results.error) {
                res->error(all_results.error->to_json());
                return res;
            } else {
                json arr = json::array();
                for (auto & res : all_results.results) {
                    GGML_ASSERT(dynamic_cast<server_task_result_cmpl_final*>(res.get()) != nullptr);
                    arr.push_back(res->to_json());
                }
                // if single request, return single object instead of array
                res->ok(arr.size() == 1 ? arr[0] : arr);
            }

        } else {
            // in streaming mode, the first error must be treated as non-stream response
            // this is to match the OAI API behavior
            // ref: https://github.com/ggml-org/llama.cpp/pull/16486#discussion_r2419657309
            server_task_result_ptr first_result = rd.next(should_stop);
            if (first_result == nullptr) {
                return res; // connection is closed
            } else if (first_result->is_error()) {
                res->error(first_result->to_json());
                return res;
            } else {
                GGML_ASSERT(
                    dynamic_cast<server_task_result_cmpl_partial*>(first_result.get()) != nullptr
                    || dynamic_cast<server_task_result_cmpl_final*>(first_result.get()) != nullptr
                );
            }

            // next responses are streamed
            if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                res->data = format_anthropic_sse(first_result->to_json());
            } else {
                res->data = format_oai_sse(first_result->to_json()); // to be sent immediately
            }
            res->status = 200;
            res->content_type = "text/event-stream";
            res->next = [res_this = res.get(), res_type, &should_stop](std::string & output) -> bool {
                if (should_stop()) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                if (!res_this->data.empty()) {
                    // flush the first chunk
                    output = std::move(res_this->data);
                    res_this->data.clear();
                    return true;
                }

                server_response_reader & rd = res_this->rd;

                // check if there is more data
                if (!rd.has_next()) {
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        // Anthropic doesn't send [DONE], message_stop was already sent
                        output = "";
                    } else if (res_type != TASK_RESPONSE_TYPE_NONE) {
                        output = "data: [DONE]\n\n";
                    } else {
                        output = "";
                    }
                    SRV_DBG("%s", "all results received, terminating stream\n");
                    return false; // no more data, terminate
                }

                // receive subsequent results
                auto result = rd.next(should_stop);
                if (result == nullptr) {
                    SRV_DBG("%s", "stopping streaming due to should_stop condition\n");
                    return false; // should_stop condition met
                }

                // send the results
                json res_json = result->to_json();
                if (result->is_error()) {
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse({
                            {"event", "error"},
                            {"data", res_json},
                        });
                    } else {
                        output = format_oai_sse(json {{ "error", res_json }});
                    }
                    SRV_DBG("%s", "error received during streaming, terminating stream\n");
                    return false; // terminate on error
                } else {
                    GGML_ASSERT(
                        dynamic_cast<server_task_result_cmpl_partial*>(result.get()) != nullptr
                        || dynamic_cast<server_task_result_cmpl_final*>(result.get()) != nullptr
                    );
                    if (res_type == TASK_RESPONSE_TYPE_ANTHROPIC) {
                        output = format_anthropic_sse(res_json);
                    } else {
                        output = format_oai_sse(res_json);
                    }
                }

                // has next data, continue
                return true;
            };
        }

        return res;
    }

    std::unique_ptr<server_res_generator> handle_slots_save(const server_http_req & req, int id_slot) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        const json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string filepath = params.slot_save_path + filename;

        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_SAVE);
            task.id = task_id;
            task.slot_action.slot_id  = id_slot;
            task.slot_action.filename = filename;
            task.slot_action.filepath = filepath;

            // TODO: use server_response_reader
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        res->ok(result->to_json());
        return res;
    }

    std::unique_ptr<server_res_generator> handle_slots_restore(const server_http_req & req, int id_slot) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        const json request_data = json::parse(req.body);
        std::string filename = request_data.at("filename");
        if (!fs_validate_filename(filename)) {
            res->error(format_error_response("Invalid filename", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string filepath = params.slot_save_path + filename;

        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_RESTORE);
            task.id = task_id;
            task.slot_action.slot_id  = id_slot;
            task.slot_action.filename = filename;
            task.slot_action.filepath = filepath;

            // TODO: use server_response_reader
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_slot_save_load*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    }

    std::unique_ptr<server_res_generator> handle_slots_erase(const server_http_req &, int id_slot) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        int task_id = ctx_server.queue_tasks.get_new_id();
        {
            server_task task(SERVER_TASK_TYPE_SLOT_ERASE);
            task.id = task_id;
            task.slot_action.slot_id = id_slot;

            // TODO: use server_response_reader
            ctx_server.queue_results.add_waiting_task_id(task_id);
            ctx_server.queue_tasks.post(std::move(task));
        }

        server_task_result_ptr result = ctx_server.queue_results.recv(task_id);
        ctx_server.queue_results.remove_waiting_task_id(task_id);

        if (result->is_error()) {
            res->error(result->to_json());
            return res;
        }

        GGML_ASSERT(dynamic_cast<server_task_result_slot_erase*>(result.get()) != nullptr);
        res->ok(result->to_json());
        return res;
    }

    std::unique_ptr<server_res_generator> handle_embeddings_impl(const server_http_req & req, task_response_type res_type) {
        auto res = std::make_unique<server_res_generator>(ctx_server);
        if (!ctx_server.params_base.embedding) {
            res->error(format_error_response("This server does not support embeddings. Start it with `--embeddings`", ERROR_TYPE_NOT_SUPPORTED));
            return res;
        }

        if (res_type != TASK_RESPONSE_TYPE_NONE && llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
            res->error(format_error_response("Pooling type 'none' is not OAI compatible. Please use a different pooling type", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        const json body = json::parse(req.body);

        // for the shape of input/content, see tokenize_input_prompts()
        json prompt;
        if (body.count("input") != 0) {
            prompt = body.at("input");
        } else if (body.contains("content")) {
            res_type = TASK_RESPONSE_TYPE_NONE; // "content" field is not OAI compatible
            prompt = body.at("content");
        } else {
            res->error(format_error_response("\"input\" or \"content\" must be provided", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }

        bool use_base64 = false;
        if (body.count("encoding_format") != 0) {
            const std::string& format = body.at("encoding_format");
            if (format == "base64") {
                use_base64 = true;
            } else if (format != "float") {
                res->error(format_error_response("The format to return the embeddings in. Can be either float or base64", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }

        auto tokenized_prompts = tokenize_input_prompts(ctx_server.vocab, ctx_server.mctx, prompt, true, true);
        for (const auto & tokens : tokenized_prompts) {
            // this check is necessary for models that do not add BOS token to the input
            if (tokens.empty()) {
                res->error(format_error_response("Input content cannot be empty", ERROR_TYPE_INVALID_REQUEST));
                return res;
            }
        }

        int embd_normalize = 2; // default to Euclidean/L2 norm
        if (body.count("embd_normalize") != 0) {
            embd_normalize = body.at("embd_normalize");
            if (llama_pooling_type(ctx_server.ctx) == LLAMA_POOLING_TYPE_NONE) {
                SRV_DBG("embd_normalize is not supported by pooling type %d, ignoring it\n", llama_pooling_type(ctx_server.ctx));
            }
        }

        // create and queue the task
        json responses = json::array();
        server_response_reader rd(ctx_server);
        {
            std::vector<server_task> tasks;
            for (size_t i = 0; i < tokenized_prompts.size(); i++) {
                server_task task = server_task(SERVER_TASK_TYPE_EMBEDDING);

                task.id     = ctx_server.queue_tasks.get_new_id();
                task.index  = i;
                task.tokens = std::move(tokenized_prompts[i]);

                // OAI-compat
                task.params.res_type = res_type;
                task.params.embd_normalize = embd_normalize;

                tasks.push_back(std::move(task));
            }
            rd.post_tasks(std::move(tasks));
        }

        // wait for the results
        auto all_results = rd.wait_for_all(req.should_stop);

        // collect results
        if (all_results.is_terminated) {
            return res; // connection is closed
        } else if (all_results.error) {
            res->error(all_results.error->to_json());
            return res;
        } else {
            for (auto & res : all_results.results) {
                GGML_ASSERT(dynamic_cast<server_task_result_embd*>(res.get()) != nullptr);
                responses.push_back(res->to_json());
            }
        }

        // write JSON response
        json root = res_type == TASK_RESPONSE_TYPE_OAI_EMBD
            ? format_embeddings_response_oaicompat(body, responses, use_base64)
            : json(responses);
        res->ok(root);
        return res;
    }
};

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        try {
            return func(req);
        } catch (const std::exception & e) {
            message = e.what();
        } catch (...) {
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, ERROR_TYPE_SERVER);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            LOG_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            LOG_ERR("got another exception: %s | while hanlding exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
    // own arguments required by this example
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // TODO: should we have a separate n_parallel parameter for the server?
    //       https://github.com/ggml-org/llama.cpp/pull/16736#discussion_r2483763177
    // TODO: this is a common configuration that is suitable for most local use cases
    //       however, overriding the parameters is a bit confusing - figure out something more intuitive
    if (params.n_parallel == 1 && params.kv_unified == false && !params.has_speculative()) {
        LOG_WRN("%s: setting n_parallel = 4 and kv_unified = true (add -kvu to disable this)\n", __func__);

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    common_init();

    // struct that contains llama context and inference
    server_context ctx_server;

    // Necessary similarity of prompt for slot selection
    ctx_server.slot_prompt_similarity = params.slot_prompt_similarity;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server, ctx_http);

    ctx_http.get ("/health",              ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));
    ctx_http.post("/props",               ex_wrapper(routes.post_props));
    ctx_http.post("/api/show",            ex_wrapper(routes.get_api_show));
    ctx_http.get ("/models",              ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/api/tags",            ex_wrapper(routes.get_models)); // ollama specific endpoint. public endpoint (no API key check)
    ctx_http.post("/completion",          ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",         ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat",            ex_wrapper(routes.post_chat_completions)); // ollama specific endpoint
    ctx_http.post("/v1/messages",         ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    ctx_http.post("/infill",              ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",           ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",          ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",       ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",              ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",        ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",            ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",          ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",      ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",       ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",       ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",               ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",      ex_wrapper(routes.post_slots));

    //
    // Start the server
    //

    // setup clean up function, to be called before exit
    auto clean_up = [&ctx_http, &ctx_server]() {
        SRV_INF("%s: cleaning up before exit...\n", __func__);
        ctx_http.stop();
        ctx_server.queue_results.terminate();
        llama_backend_free();
    };

    // start the HTTP server before loading the model to be able to serve /health requests
    if (!ctx_http.start()) {
        clean_up();
        LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
        return 1;
    }

    // load the model
    LOG_INF("%s: loading model\n", __func__);

    if (!ctx_server.load_model(params)) {
        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        LOG_ERR("%s: exiting due to model loading error\n", __func__);
        return 1;
    }

    ctx_server.init();
    ctx_http.is_ready.store(true);

    LOG_INF("%s: model loaded\n", __func__);

    ctx_server.queue_tasks.on_new_task([&ctx_server](server_task && task) {
        ctx_server.process_single_task(std::move(task));
    });

    ctx_server.queue_tasks.on_update_slots([&ctx_server]() {
        ctx_server.update_slots();
    });

    // Setup signal handlers
    setup_signal_handlers([&](int) {
        // this will unblock start_loop()
        ctx_server.queue_tasks.terminate();
    });

    LOG_INF("%s: server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
    LOG_INF("%s: starting the main loop...\n", __func__);
    // this call blocks the main thread until queue_tasks.terminate() is called
    ctx_server.queue_tasks.start_loop();

    clean_up();
    if (ctx_http.thread.joinable()) {
        ctx_http.thread.join();
    }
    llama_memory_breakdown_print(ctx_server.ctx);

    return 0;
}
