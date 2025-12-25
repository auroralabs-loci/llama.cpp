#pragma once

#include "server-ws.h"
#include "server-mcp.h"
#include "server-mproc.h"
#include <unordered_map>
#include <mutex>
#include <memory>

// Manages MCP server instances per WebSocket connection
// Handles message routing between WebSocket clients and MCP processes
class server_mcp_bridge {
public:
    server_mcp_bridge();
    ~server_mcp_bridge();

    // Load MCP configuration from JSON file
    bool load_config(const std::string & config_path);

    // Get available MCP server names from config (auto-reloads if file changed)
    std::vector<std::string> get_available_servers();

    // Handle WebSocket connection opened
    void on_connection_opened(std::shared_ptr<server_ws_connection> conn);

    // Handle WebSocket message (from UI)
    void on_connection_message(std::shared_ptr<server_ws_connection> conn,
                               const std::string & message);

    // Handle WebSocket connection closed
    void on_connection_closed(std::shared_ptr<server_ws_connection> conn);

    // Get number of active connections
    size_t active_connections() const {
        std::unique_lock lock(mutex_);
        return connections_.size();
    }

private:
    // Per-connection state
    struct connection_state {
        std::shared_ptr<server_ws_connection> conn;
        std::unique_ptr<mcp_process> process;
        std::string server_name;
        bool initialized = false;  // true after initialize handshake
    };

    mutable std::mutex mutex_;
    std::unordered_map<void*, std::unique_ptr<connection_state>> connections_;

    // MCP server configurations
    mcp_config mcp_config_;

    // Path to the loaded config file (for reloading)
    std::string config_path_;

    // Last modification time of the config file
    std::filesystem::file_time_type last_modified_;

    // Get the file modification time
    std::filesystem::file_time_type get_file_mtime(const std::string & path) const;

    // Parse server name from query parameter
    std::string get_server_name(std::shared_ptr<server_ws_connection> conn);

    // Get MCP server config by name
    std::optional<mcp_server_config> get_server_config(const std::string & name);

    // Get or create MCP process for connection
    mcp_process * get_or_create_process(connection_state * state);

    // Forward message from WebSocket to MCP process
    void forward_to_mcp(connection_state * state, const std::string & message);

    // Forward message from MCP process to WebSocket
    void forward_to_ws(connection_state * state, const std::string & message);

    // Handle MCP initialize request
    void handle_initialize(connection_state * state, const mcp_jsonrpc_request & req);

    // Send JSON-RPC response
    void send_response(connection_state * state, const mcp_jsonrpc_response & resp);

    // Send JSON-RPC notification
    void send_notification(connection_state * state, const mcp_jsonrpc_notification & notif);
};
