#pragma once

#include "server-mcp.h"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

// Manages stdio process for MCP server
// Spawns a process and provides bidirectional JSON-RPC communication
class mcp_process {
public:
    mcp_process(const mcp_server_config & config);
    ~mcp_process();

    // Non-copyable, movable
    mcp_process(const mcp_process &) = delete;
    mcp_process & operator=(const mcp_process &) = delete;
    mcp_process(mcp_process &&) noexcept;
    mcp_process & operator=(mcp_process &&) noexcept;

    // Start the process
    bool start();

    // Stop the process gracefully
    void stop();

    // Write JSON-RPC request to stdin (as a JSON line)
    // Returns true if write succeeded
    bool write(const std::string & json_line);

    // Register callback for stdout messages (JSON lines)
    using on_message_t = std::function<void(const std::string &)>;
    void set_on_message(on_message_t callback);

    // Check if process is running
    bool is_running() const { return running_; }

    // Get process name (for logging)
    const std::string & name() const { return config_.name; }

private:
    mcp_server_config config_;
    std::atomic<bool> running_;
    std::atomic<bool> should_stop_;

    // Process handles (platform-specific)
#ifdef _WIN32
    void * process_handle_;  // HANDLE
    void * stdin_write_;     // HANDLE
    void * stdout_read_;     // HANDLE
#else
    pid_t pid_;
    int stdin_fd_;   // Write end of pipe to child's stdin
    int stdout_fd_;  // Read end of pipe from child's stdout
#endif

    // Read thread
    std::thread read_thread_;
    std::string read_buffer_;  // Buffer for incomplete lines

    // Message callback
    on_message_t on_message_;
    std::mutex callback_mutex_;

    // Platform-specific implementation
    bool spawn_process();
    void terminate_process();
    void read_loop();
    bool platform_write(const std::string & data);
};

// Factory for creating MCP processes from config
class mcp_process_factory {
public:
    // Create process from configuration
    static std::unique_ptr<mcp_process> create(const mcp_server_config & config);
};
