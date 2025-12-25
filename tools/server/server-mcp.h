#pragma once

#include "server-common.h"
#include <map>
#include <string>
#include <vector>
#include <optional>
#include <functional>
#include <fstream>

// MCP JSON-RPC 2.0 base types
struct mcp_request_id {
    std::string str;
    std::optional<int64_t> num;

    bool is_valid() const { return !str.empty() || num.has_value(); }
    std::string to_string() const {
        if (num.has_value()) return std::to_string(*num);
        return str;
    }
};

// JSON-RPC 2.0 Request
struct mcp_jsonrpc_request {
    std::string jsonrpc = "2.0";
    mcp_request_id id;
    std::string method;
    json params;  // Can be null, object, or array

    mcp_jsonrpc_request() = default;
    mcp_jsonrpc_request(const json & j) {
        if (j.contains("jsonrpc")) jsonrpc = j["jsonrpc"].get<std::string>();
        if (j.contains("method")) method = j["method"].get<std::string>();
        if (j.contains("params")) params = j["params"];
        if (j.contains("id")) {
            const auto & id_val = j["id"];
            if (id_val.is_string()) {
                id.str = id_val.get<std::string>();
            } else if (id_val.is_number_integer()) {
                id.num = id_val.get<int64_t>();
            }
        }
    }

    json to_json() const {
        json j;
        j["jsonrpc"] = jsonrpc;
        j["method"] = method;
        if (!params.is_null()) j["params"] = params;
        if (id.str.empty() && id.num.has_value()) {
            j["id"] = *id.num;
        } else if (!id.str.empty()) {
            j["id"] = id.str;
        }
        return j;
    }
};

// JSON-RPC 2.0 Response
struct mcp_jsonrpc_response {
    std::string jsonrpc = "2.0";
    mcp_request_id id;
    std::optional<json> result;
    std::optional<json> error;  // {code, message, data?}

    json to_json() const {
        json j;
        j["jsonrpc"] = jsonrpc;
        if (id.str.empty() && id.num.has_value()) {
            j["id"] = *id.num;
        } else if (!id.str.empty()) {
            j["id"] = id.str;
        }
        if (error.has_value()) {
            j["error"] = *error;
        } else if (result.has_value()) {
            j["result"] = *result;
        }
        return j;
    }

    static mcp_jsonrpc_response make_error(mcp_request_id id, int code, const std::string & message, const std::optional<json> & data = std::nullopt) {
        mcp_jsonrpc_response resp;
        resp.id = id;
        json err;
        err["code"] = code;
        err["message"] = message;
        if (data.has_value()) err["data"] = *data;
        resp.error = err;
        return resp;
    }

    static mcp_jsonrpc_response make_result(mcp_request_id id, const json & result) {
        mcp_jsonrpc_response resp;
        resp.id = id;
        resp.result = result;
        return resp;
    }
};

// JSON-RPC 2.0 Notification (no id field)
struct mcp_jsonrpc_notification {
    std::string jsonrpc = "2.0";
    std::string method;
    json params;

    json to_json() const {
        json j;
        j["jsonrpc"] = jsonrpc;
        j["method"] = method;
        if (!params.is_null()) j["params"] = params;
        return j;
    }
};

// MCP Tool types
struct mcp_tool {
    std::string name;
    std::string description;
    json input_schema;  // JSON Schema
};

// MCP Tool call
struct mcp_tool_call {
    std::string name;
    json arguments;  // Arguments map
};

// MCP Server configuration (from JSON config file)
struct mcp_server_config {
    std::string name;
    std::string command;              // Command to spawn
    std::vector<std::string> args;    // Command arguments
    std::map<std::string, std::string> env;  // Environment variables
    std::string cwd;                  // Working directory (optional)

    mcp_server_config() = default;
    mcp_server_config(const std::string & name, const json & j) : name(name) {
        if (j.contains("command")) command = j["command"].get<std::string>();
        if (j.contains("args")) {
            const auto & args_arr = j["args"];
            if (args_arr.is_array()) {
                for (const auto & arg : args_arr) {
                    args.push_back(arg.get<std::string>());
                }
            }
        }
        if (j.contains("env")) {
            const auto & env_obj = j["env"];
            if (env_obj.is_object()) {
                for (auto it = env_obj.begin(); it != env_obj.end(); ++it) {
                    env[it.key()] = it.value().get<std::string>();
                }
            }
        }
        if (j.contains("cwd")) cwd = j["cwd"].get<std::string>();
    }

    json to_json() const {
        json j;
        j["command"] = command;
        j["args"] = args;
        if (!env.empty()) j["env"] = env;
        if (!cwd.empty()) j["cwd"] = cwd;
        return j;
    }
};

// MCP config file structure
// Expected JSON format:
// {
//   "mcpServers": {
//     "filesystem": {
//       "command": "npx",
//       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
//     },
//     "brave-search": {
//       "command": "npx",
//       "args": ["-y", "@modelcontextprotocol/server-brave-search"],
//       "env": { "BRAVE_API_KEY": "..." }
//     }
//   }
// }

struct mcp_config {
    std::map<std::string, mcp_server_config> mcp_servers;

    mcp_config() = default;

    // Load from JSON file
    static std::optional<mcp_config> from_file(const std::string & path) {
        std::ifstream f(path);
        if (!f.is_open()) {
            SRV_WRN("%s: failed to open MCP config file: %s\n", __func__, path.c_str());
            return std::nullopt;
        }

        try {
            json j;
            f >> j;

            mcp_config config;
            if (j.contains("mcpServers")) {
                const auto & servers = j["mcpServers"];
                if (servers.is_object()) {
                    for (auto it = servers.begin(); it != servers.end(); ++it) {
                        config.mcp_servers[it.key()] = mcp_server_config(it.key(), it.value());
                    }
                }
            }

            SRV_INF("%s: loaded %zu MCP server configurations from %s\n",
                    __func__, config.mcp_servers.size(), path.c_str());
            return config;
        } catch (const std::exception & e) {
            SRV_ERR("%s: failed to parse MCP config file: %s: %s\n",
                    __func__, path.c_str(), e.what());
            return std::nullopt;
        }
    }

    // Get server config by name
    std::optional<mcp_server_config> get_server(const std::string & name) const {
        auto it = mcp_servers.find(name);
        if (it != mcp_servers.end()) {
            return it->second;
        }
        return std::nullopt;
    }

    json to_json() const {
        json j;
        j["mcpServers"] = json::object();
        for (const auto & [name, config] : mcp_servers) {
            j["mcpServers"][name] = config.to_json();
        }
        return j;
    }
};

// MCP protocol methods (from MCP spec)
namespace mcp_methods {
    constexpr const char * INITIALIZE        = "initialize";
    constexpr const char * INITIALIZED       = "notifications/initialized";
    constexpr const char * LIST_TOOLS        = "tools/list";
    constexpr const char * CALL_TOOL         = "tools/call";
    constexpr const char * LIST_RESOURCES    = "resources/list";
    constexpr const char * READ_RESOURCE     = "resources/read";
    constexpr const char * LIST_PROMPTS      = "prompts/list";
    constexpr const char * GET_PROMPT        = "prompts/get";
    constexpr const char * SET_LEVEL         = "logging/set_level";
    constexpr const char * TOOLS_CHANGED     = "notifications/tools/list_changed";
    constexpr const char * RESOURCES_CHANGED = "notifications/resources/list_changed";
    constexpr const char * PROMPTS_CHANGED   = "notifications/prompts/list_changed";
    constexpr const char * CANCEL_REQUEST    = "requests/cancel";
    constexpr const char * PING              = "ping";
}
