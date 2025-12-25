#pragma once

#include "server-common.h"
#include <functional>
#include <string>
#include <memory>
#include <map>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

struct common_params;

// Forward declarations
struct sockaddr_in;
class ws_connection_impl;

// WebSocket connection interface
// Abstracts the underlying WebSocket implementation
struct server_ws_connection {
    virtual ~server_ws_connection() = default;

    // Send a message to the client
    virtual void send(const std::string & message) = 0;

    // Close the connection
    virtual void close(int code = 1000, const std::string & reason = "") = 0;

    // Get query parameter by key
    virtual std::string get_query_param(const std::string & key) const = 0;

    // Get the remote address
    virtual std::string get_remote_address() const = 0;
};

// Forward declarations
class ws_connection_impl;

// WebSocket context - manages the WebSocket server
// Runs on a separate thread and handles WebSocket connections
struct server_ws_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread;
    std::atomic<bool> is_ready = false;

    std::string path_prefix;  // e.g., "/mcp"
    int port;

    server_ws_context();
    ~server_ws_context();

    // Initialize the WebSocket server
    bool init(const common_params & params);

    // Start the WebSocket server (runs in background thread)
    bool start();

    // Stop the WebSocket server
    void stop();

    // Get the actual port the WebSocket server is listening on
    int get_actual_port() const;

    // Set the port for the WebSocket server (note: actual port may differ if set to 0)
    void set_port(int port) { this->port = port; }

    // Called when new connection is established
    using on_open_t = std::function<void(std::shared_ptr<server_ws_connection>)>;
    void on_open(on_open_t handler);

    // Called when message is received from a connection
    using on_message_t = std::function<void(std::shared_ptr<server_ws_connection>, const std::string &)>;
    void on_message(on_message_t handler);

    // Called when connection is closed
    using on_close_t = std::function<void(std::shared_ptr<server_ws_connection>)>;
    void on_close(on_close_t handler);

    // For debugging
    std::string listening_address;
};
