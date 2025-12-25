#include "server-mcp-bridge.h"
#include "log.h"
#include <filesystem>
#include <sys/stat.h>

server_mcp_bridge::server_mcp_bridge() {
    SRV_INF("%s: MCP bridge initialized\n", __func__);
}

server_mcp_bridge::~server_mcp_bridge() {
    std::unique_lock lock(mutex_);
    connections_.clear();
    SRV_INF("%s: MCP bridge destroyed\n", __func__);
}

bool server_mcp_bridge::load_config(const std::string & config_path) {
    auto config = mcp_config::from_file(config_path);
    if (!config) {
        SRV_WRN("%s: failed to load MCP config from: %s\n",
                __func__, config_path.c_str());
        // Use empty config - servers will be looked up dynamically
        mcp_config_ = mcp_config{};
        config_path_.clear();
        last_modified_ = {};
        return false;
    }

    mcp_config_ = std::move(*config);
    config_path_ = config_path;
    last_modified_ = get_file_mtime(config_path);
    SRV_INF("%s: loaded %zu MCP server configurations from: %s\n",
            __func__, mcp_config_.mcp_servers.size(), config_path.c_str());
    return true;
}

std::filesystem::file_time_type server_mcp_bridge::get_file_mtime(const std::string & path) const {
    try {
        return std::filesystem::last_write_time(path);
    } catch (const std::filesystem::filesystem_error &) {
        return {};
    }
}

void server_mcp_bridge::on_connection_opened(std::shared_ptr<server_ws_connection> conn) {
    std::string server_name = get_server_name(conn);

    if (server_name.empty()) {
        SRV_WRN("%s: WebSocket connection missing 'server' query parameter\n", __func__);
        conn->close(1008, "Missing 'server' query parameter");
        return;
    }

    SRV_INF("%s: WebSocket connection opened for MCP server: %s\n",
            __func__, server_name.c_str());

    auto state = std::make_unique<connection_state>();
    state->conn = conn;
    state->server_name = server_name;
    state->initialized = false;

    void * conn_ptr = conn.get();
    {
        std::unique_lock lock(mutex_);
        connections_[conn_ptr] = std::move(state);
    }
}

void server_mcp_bridge::on_connection_message(std::shared_ptr<server_ws_connection> conn,
                                              const std::string & message) {
    void * conn_ptr = conn.get();

    std::unique_lock lock(mutex_);
    auto it = connections_.find(conn_ptr);
    if (it == connections_.end()) {
        SRV_WRN("%s: message from unknown connection\n", __func__);
        return;
    }

    connection_state * state = it->second.get();

    SRV_DBG("%s: message from %s: %s\n", __func__, state->server_name.c_str(), message.c_str());

    // Parse JSON-RPC message
    try {
        json j = json::parse(message);

        // Check if it's a request (has id) or notification (no id)
        if (j.contains("id")) {
            mcp_jsonrpc_request req(j);

            // Handle initialize request
            if (req.method == mcp_methods::INITIALIZE) {
                handle_initialize(state, req);
                return;
            }

            // Forward to MCP process
            forward_to_mcp(state, message);
        } else {
            // It's a notification - forward to MCP process
            forward_to_mcp(state, message);
        }
    } catch (const std::exception & e) {
        SRV_ERR("%s: failed to parse JSON-RPC message: %s\n",
                __func__, e.what());

        // Send error response
        json error_resp;
        error_resp["jsonrpc"] = "2.0";
        error_resp["error"]["code"] = -32700;
        error_resp["error"]["message"] = "Parse error";
        conn->send(error_resp.dump());
    }
}

void server_mcp_bridge::on_connection_closed(std::shared_ptr<server_ws_connection> conn) {
    void * conn_ptr = conn.get();

    std::unique_lock lock(mutex_);
    auto it = connections_.find(conn_ptr);
    if (it != connections_.end()) {
        SRV_INF("%s: WebSocket connection closed for MCP server: %s\n",
                __func__, it->second->server_name.c_str());

        // Process will be automatically stopped when connection_state is destroyed
        connections_.erase(it);
    }
}

std::string server_mcp_bridge::get_server_name(std::shared_ptr<server_ws_connection> conn) {
    return conn->get_query_param("server");
}

std::optional<mcp_server_config> server_mcp_bridge::get_server_config(const std::string & name) {
    // First check loaded config
    auto config = mcp_config_.get_server(name);
    if (config) {
        return config;
    }

    // If not found, return nullopt
    // In the future, we could support dynamic server discovery
    return std::nullopt;
}

mcp_process * server_mcp_bridge::get_or_create_process(connection_state * state) {
    if (state->process && state->process->is_running()) {
        SRV_INF("%s: reusing existing MCP process: %s\n",
                __func__, state->server_name.c_str());
        return state->process.get();
    }

    // Get server configuration
    auto config = get_server_config(state->server_name);
    if (!config) {
        SRV_ERR("%s: no configuration found for MCP server: %s\n",
                __func__, state->server_name.c_str());
        return nullptr;
    }

    // Create process
    state->process = mcp_process_factory::create(*config);

    // Set up message callback BEFORE starting
    state->process->set_on_message([this, state](const std::string & msg) {
        SRV_INF("%s: received from %s: %s\n", __func__, state->server_name.c_str(), msg.c_str());
        forward_to_ws(state, msg);
    });

    // Start process
    SRV_INF("%s: starting MCP process: %s\n",
            __func__, state->server_name.c_str());
    if (!state->process->start()) {
        SRV_ERR("%s: failed to start MCP process: %s\n",
                __func__, state->server_name.c_str());
        state->process.reset();
        return nullptr;
    }

    SRV_INF("%s: successfully started MCP process: %s\n",
            __func__, state->server_name.c_str());

    return state->process.get();
}

void server_mcp_bridge::forward_to_mcp(connection_state * state, const std::string & message) {
    mcp_process * proc = get_or_create_process(state);
    if (!proc) {
        SRV_ERR("%s: no MCP process available for: %s\n",
                __func__, state->server_name.c_str());

        // Send error response
        try {
            json j = json::parse(message);
            if (j.contains("id")) {
                mcp_request_id id;
                if (j["id"].is_string()) {
                    id.str = j["id"].get<std::string>();
                } else {
                    id.num = j["id"].get<int64_t>();
                }
                auto resp = mcp_jsonrpc_response::make_error(
                    id, -32000, "MCP process not available");
                send_response(state, resp);
            }
        } catch (...) {
            // Ignore parse errors here
        }
        return;
    }

    // Write to MCP process
    SRV_INF("%s: writing to %s: %s\n", __func__, state->server_name.c_str(), message.c_str());
    if (!proc->write(message)) {
        SRV_ERR("%s: failed to write to MCP process: %s\n",
                __func__, state->server_name.c_str());
    } else {
        SRV_INF("%s: successfully wrote to %s\n", __func__, state->server_name.c_str());
    }
}

void server_mcp_bridge::forward_to_ws(connection_state * state, const std::string & message) {
    if (!state->conn) {
        SRV_WRN("%s: no connection for %s, cannot forward message\n",
                __func__, state->server_name.c_str());
        return;
    }

    SRV_INF("%s: to %s: %s\n", __func__, state->server_name.c_str(), message.c_str());

    state->conn->send(message);
}

void server_mcp_bridge::handle_initialize(connection_state * state, const mcp_jsonrpc_request & req) {
    // Process initialize request
    // We need to forward this to the MCP process
    forward_to_mcp(state, req.to_json().dump());

    // Mark as initialized (the actual response will come from the MCP process)
    state->initialized = true;
}

void server_mcp_bridge::send_response(connection_state * state, const mcp_jsonrpc_response & resp) {
    if (!state->conn) {
        return;
    }

    state->conn->send(resp.to_json().dump());
}

void server_mcp_bridge::send_notification(connection_state * state, const mcp_jsonrpc_notification & notif) {
    if (!state->conn) {
        return;
    }

    state->conn->send(notif.to_json().dump());
}

std::vector<std::string> server_mcp_bridge::get_available_servers() {
    // Check if config file has been modified and reload if needed
    if (!config_path_.empty()) {
        auto current_mtime = get_file_mtime(config_path_);
        if (current_mtime != last_modified_) {
            SRV_INF("%s: config file changed, reloading from: %s\n",
                    __func__, config_path_.c_str());
            load_config(config_path_);
        }
    }

    std::vector<std::string> servers;
    servers.reserve(mcp_config_.mcp_servers.size());

    for (const auto & [name, config] : mcp_config_.mcp_servers) {
        servers.push_back(name);
    }

    return servers;
}
