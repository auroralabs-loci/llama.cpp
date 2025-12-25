#include "server-mproc.h"
#include "log.h"

#include <cstring>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <poll.h>
#include <errno.h>
#endif

// Environment variables to disable output buffering in child processes
// This is CRITICAL for stdio-based communication - without it, output may be
// delayed indefinitely until the buffer fills up (typically 4-8KB)
static const std::vector<std::pair<std::string, std::string>> UNBUFFER_ENV_VARS = {
    {"PYTHONUNBUFFERED", "1"},           // Python: disable stdout/stderr buffering
    {"PYTHONDONTWRITEBYTECODE", "1"},    // Python: don't write .pyc files
    {"NODE_OPTIONS", "--no-warnings"},   // Node.js: reduce noise (already line-buffered)
    {"RUST_BACKTRACE", "1"},             // Rust: show backtraces on panic
    {"STDBUF_O", "L"},                   // stdbuf hint for line buffering
};

mcp_process::mcp_process(const mcp_server_config & config)
    : config_(config)
    , running_(false)
    , should_stop_(false)
#ifdef _WIN32
    , process_handle_(nullptr)
    , stdin_write_(nullptr)
    , stdout_read_(nullptr)
#else
    , pid_(-1)
    , stdin_fd_(-1)
    , stdout_fd_(-1)
#endif
{
}

mcp_process::~mcp_process() {
    stop();
}

mcp_process::mcp_process(mcp_process && other) noexcept
    : config_(std::move(other.config_))
    , running_(other.running_.load())
    , should_stop_(other.should_stop_.load())
#ifdef _WIN32
    , process_handle_(other.process_handle_)
    , stdin_write_(other.stdin_write_)
    , stdout_read_(other.stdout_read_)
#else
    , pid_(other.pid_)
    , stdin_fd_(other.stdin_fd_)
    , stdout_fd_(other.stdout_fd_)
#endif
    , on_message_(std::move(other.on_message_)) {
    other.running_ = false;
#ifdef _WIN32
    other.process_handle_ = nullptr;
    other.stdin_write_ = nullptr;
    other.stdout_read_ = nullptr;
#else
    other.pid_ = -1;
    other.stdin_fd_ = -1;
    other.stdout_fd_ = -1;
#endif
}

mcp_process & mcp_process::operator=(mcp_process && other) noexcept {
    if (this != &other) {
        stop();

        config_ = std::move(other.config_);
        running_ = other.running_.load();
        should_stop_ = other.should_stop_.load();

#ifdef _WIN32
        process_handle_ = other.process_handle_;
        stdin_write_ = other.stdin_write_;
        stdout_read_ = other.stdout_read_;
        other.process_handle_ = nullptr;
        other.stdin_write_ = nullptr;
        other.stdout_read_ = nullptr;
#else
        pid_ = other.pid_;
        stdin_fd_ = other.stdin_fd_;
        stdout_fd_ = other.stdout_fd_;
        other.pid_ = -1;
        other.stdin_fd_ = -1;
        other.stdout_fd_ = -1;
#endif

        on_message_ = std::move(other.on_message_);
        other.running_ = false;
    }
    return *this;
}

bool mcp_process::start() {
    if (running_) {
        SRV_WRN("%s: process already running: %s\n", __func__, config_.name.c_str());
        return false;
    }

    if (!spawn_process()) {
        return false;
    }

    running_ = true;
    should_stop_ = false;

    // Start read thread
    read_thread_ = std::thread(&mcp_process::read_loop, this);

    SRV_INF("%s: started MCP process: %s (cmd: %s)\n",
            __func__, config_.name.c_str(), config_.command.c_str());

    return true;
}

void mcp_process::stop() {
    if (!running_) {
        return;
    }

    should_stop_ = true;

    // Wait for read thread to finish
    if (read_thread_.joinable()) {
        read_thread_.join();
    }

    terminate_process();

    running_ = false;

    SRV_INF("%s: stopped MCP process: %s\n", __func__, config_.name.c_str());
}

bool mcp_process::write(const std::string & json_line) {
    if (!running_) {
        SRV_WRN("%s: process not running: %s\n", __func__, config_.name.c_str());
        return false;
    }

    SRV_INF("%s: writing to %s: %.200s%s\n",
            __func__, config_.name.c_str(), json_line.c_str(),
            json_line.length() > 200 ? "..." : "");

    bool result = platform_write(json_line + "\n");

    if (!result) {
        SRV_ERR("%s: write failed for %s\n", __func__, config_.name.c_str());
    }

    return result;
}

void mcp_process::set_on_message(on_message_t callback) {
    std::lock_guard<std::mutex> lock(callback_mutex_);
    on_message_ = std::move(callback);
}

void mcp_process::read_loop() {
    std::vector<char> buffer(4096);
    read_buffer_.clear();

    SRV_INF("%s: read loop started for %s\n", __func__, config_.name.c_str());

    while (!should_stop_ && running_) {
#ifdef _WIN32
        DWORD bytes_read = 0;
        BOOL success = ReadFile(
            static_cast<HANDLE>(stdout_read_),
            buffer.data(),
            static_cast<DWORD>(buffer.size() - 1),
            &bytes_read,
            nullptr
        );
        if (!success || bytes_read == 0) {
            if (!should_stop_) {
                DWORD err = GetLastError();
                SRV_WRN("%s: read error or EOF for %s: error=%lu\n", __func__, config_.name.c_str(), err);
            }
            break;
        }
#else
        // Use poll() to implement timeout-based reading
        // This allows us to detect stuck processes and log periodic status
        struct pollfd pfd;
        pfd.fd = stdout_fd_;
        pfd.events = POLLIN;

        int poll_result = poll(&pfd, 1, 5000);  // 5 second timeout

        if (poll_result < 0) {
            if (errno == EINTR) {
                continue;  // Interrupted by signal, retry
            }
            SRV_ERR("%s: poll error for %s: %s\n", __func__, config_.name.c_str(), strerror(errno));
            break;
        }

        if (poll_result == 0) {
            // Timeout - no data available, but process might still be alive
            // This is normal for idle connections, just continue waiting
            continue;
        }

        if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
            if (!should_stop_) {
                SRV_WRN("%s: pipe error for %s: revents=0x%x\n", __func__, config_.name.c_str(), pfd.revents);
            }
            break;
        }

        ssize_t bytes_read = read(stdout_fd_, buffer.data(), buffer.size() - 1);
        if (bytes_read < 0) {
            if (errno == EINTR || errno == EAGAIN) {
                continue;  // Retry
            }
            if (!should_stop_) {
                SRV_ERR("%s: read error for %s: %s\n", __func__, config_.name.c_str(), strerror(errno));
            }
            break;
        }
        if (bytes_read == 0) {
            if (!should_stop_) {
                SRV_INF("%s: EOF from %s (process likely exited)\n", __func__, config_.name.c_str());
            }
            break;
        }
#endif

        buffer[bytes_read] = '\0';

        // Log raw bytes received for debugging
        SRV_DBG("%s: raw read from %s (%zd bytes): %.*s\n",
                __func__, config_.name.c_str(), (size_t)bytes_read,
                (int)std::min((size_t)bytes_read, (size_t)200), buffer.data());

        // Process lines
        std::string data(buffer.data(), bytes_read);
        read_buffer_ += data;

        // Extract complete lines
        size_t pos = 0;
        while ((pos = read_buffer_.find('\n')) != std::string::npos) {
            std::string line = read_buffer_.substr(0, pos);
            read_buffer_.erase(0, pos + 1);

            // Skip empty lines
            if (line.empty() || line == "\r") {
                continue;
            }

            // Trim trailing \r
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }

            // Check if this looks like JSON (MCP message) or stderr output
            bool is_json = !line.empty() && (line[0] == '{' || line[0] == '[');

            if (is_json) {
                SRV_INF("%s: JSON from %s: %.200s%s\n",
                        __func__, config_.name.c_str(), line.c_str(),
                        line.length() > 200 ? "..." : "");
            } else {
                // This is likely stderr output or non-JSON - log it prominently
                SRV_WRN("%s: stderr/non-JSON from %s: %s\n",
                        __func__, config_.name.c_str(), line.c_str());
            }

            // Invoke callback (only for JSON-like messages)
            if (is_json) {
                std::lock_guard<std::mutex> lock(callback_mutex_);
                if (on_message_) {
                    on_message_(line);
                }
            }
        }
    }

    SRV_INF("%s: read loop ended for %s (should_stop=%d, running=%d, buffer_remaining=%zu)\n",
            __func__, config_.name.c_str(), should_stop_.load(), running_.load(), read_buffer_.size());
}

#ifdef _WIN32

bool mcp_process::spawn_process() {
    SECURITY_ATTRIBUTES sa = {};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;

    HANDLE stdin_read = nullptr, stdin_write = nullptr;
    HANDLE stdout_read = nullptr, stdout_write = nullptr;

    // Create pipes for stdin
    if (!CreatePipe(&stdin_read, &stdin_write, &sa, 0)) {
        SRV_ERR("%s: CreatePipe stdin failed: %s\n", __func__, config_.name.c_str());
        return false;
    }
    SetHandleInformation(stdin_write, HANDLE_FLAG_INHERIT, 0);

    // Create pipes for stdout
    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        SRV_ERR("%s: CreatePipe stdout failed: %s\n", __func__, config_.name.c_str());
        CloseHandle(stdin_read);
        CloseHandle(stdin_write);
        return false;
    }
    SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0);

    // Build command line
    std::string cmdline = config_.command;
    for (const auto & arg : config_.args) {
        cmdline += " ";
        // Quote arguments that contain spaces
        if (arg.find(' ') != std::string::npos) {
            cmdline += "\"" + arg + "\"";
        } else {
            cmdline += arg;
        }
    }

    STARTUPINFOA si = {};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = stdin_read;
    si.hStdOutput = stdout_write;
    si.hStdError = stdout_write;  // Redirect stderr to stdout for capture

    PROCESS_INFORMATION pi = {};

    // Create environment block with unbuffering vars first
    std::string env_block;

    // Add unbuffering environment variables (critical for stdio communication)
    for (const auto & [key, value] : UNBUFFER_ENV_VARS) {
        env_block += key + "=" + value;
        env_block.push_back('\0');
    }

    // Add user-specified environment variables
    for (const auto & [key, value] : config_.env) {
        env_block += key + "=" + value;
        env_block.push_back('\0');
    }
    env_block.push_back('\0');  // Double-null terminated

    SRV_INF("%s: spawning process for %s: %s\n", __func__, config_.name.c_str(), cmdline.c_str());

    // Set working directory if specified, otherwise use nullptr (inherits from parent)
    LPCSTR current_directory = config_.cwd.empty() ? nullptr : config_.cwd.c_str();

    BOOL success = CreateProcessA(
        nullptr,
        const_cast<char *>(cmdline.c_str()),
        nullptr,
        nullptr,
        TRUE,  // Inherit handles
        CREATE_NO_WINDOW,
        const_cast<char *>(env_block.c_str()),  // Always use env block with unbuffer vars
        current_directory,
        &si,
        &pi
    );

    // Close unused pipe ends
    CloseHandle(stdin_read);
    CloseHandle(stdout_write);

    if (!success) {
        DWORD err = GetLastError();
        SRV_ERR("%s: CreateProcess failed for %s: error %lu\n",
                __func__, config_.name.c_str(), err);
        CloseHandle(stdin_write);
        CloseHandle(stdout_read);
        return false;
    }

    process_handle_ = pi.hProcess;
    CloseHandle(pi.hThread);  // We don't need the thread handle

    stdin_write_ = stdin_write;
    stdout_read_ = stdout_read;

    return true;
}

void mcp_process::terminate_process() {
    if (stdin_write_) {
        CloseHandle(static_cast<HANDLE>(stdin_write_));
        stdin_write_ = nullptr;
    }
    if (stdout_read_) {
        CloseHandle(static_cast<HANDLE>(stdout_read_));
        stdout_read_ = nullptr;
    }
    if (process_handle_) {
        TerminateProcess(static_cast<HANDLE>(process_handle_), 1);
        CloseHandle(static_cast<HANDLE>(process_handle_));
        process_handle_ = nullptr;
    }
}

bool mcp_process::platform_write(const std::string & data) {
    DWORD bytes_written = 0;
    BOOL success = WriteFile(
        static_cast<HANDLE>(stdin_write_),
        data.data(),
        static_cast<DWORD>(data.size()),
        &bytes_written,
        nullptr
    );
    if (!success) {
        SRV_ERR("%s: WriteFile failed: %s\n", __func__, config_.name.c_str());
        running_ = false;
        return false;
    }
    FlushFileBuffers(static_cast<HANDLE>(stdin_write_));
    return true;
}

#else // Unix

bool mcp_process::spawn_process() {
    // Build command line for logging
    std::string cmdline = config_.command;
    for (const auto & arg : config_.args) {
        cmdline += " " + arg;
    }
    SRV_INF("%s: spawning process for %s: %s\n", __func__, config_.name.c_str(), cmdline.c_str());

    int stdin_pipe[2] = {-1, -1};
    int stdout_pipe[2] = {-1, -1};

    if (pipe(stdin_pipe) != 0) {
        SRV_ERR("%s: pipe stdin failed: %s\n", __func__, strerror(errno));
        return false;
    }
    if (pipe(stdout_pipe) != 0) {
        SRV_ERR("%s: pipe stdout failed: %s\n", __func__, strerror(errno));
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        return false;
    }

    pid_t pid = fork();

    if (pid < 0) {
        SRV_ERR("%s: fork failed: %s\n", __func__, strerror(errno));
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);
        return false;
    }

    if (pid == 0) {
        // Child process

        // Set up pipes - redirect both stdout AND stderr to the pipe
        // This ensures we capture error messages from the MCP server
        dup2(stdin_pipe[0], STDIN_FILENO);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stdout_pipe[1], STDERR_FILENO);  // Capture stderr too!

        // Close unused pipe ends
        close(stdin_pipe[0]);
        close(stdin_pipe[1]);
        close(stdout_pipe[0]);
        close(stdout_pipe[1]);

        // Change working directory if specified
        if (!config_.cwd.empty()) {
            if (chdir(config_.cwd.c_str()) != 0) {
                fprintf(stderr, "chdir to '%s' failed: %s\n", config_.cwd.c_str(), strerror(errno));
                _exit(126);
            }
        }

        // Set unbuffering environment variables FIRST
        // This is CRITICAL - without these, Python and other interpreters
        // will buffer stdout, causing communication to hang
        for (const auto & [key, value] : UNBUFFER_ENV_VARS) {
            setenv(key.c_str(), value.c_str(), 0);  // 0 = don't overwrite if already set
        }

        // Set up user-specified environment (can override unbuffer vars if needed)
        for (const auto & [key, value] : config_.env) {
            setenv(key.c_str(), value.c_str(), 1);  // 1 = overwrite
        }

        // Build argv
        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(config_.command.c_str()));
        for (const auto & arg : config_.args) {
            argv.push_back(const_cast<char *>(arg.c_str()));
        }
        argv.push_back(nullptr);

        // Execute
        execvp(config_.command.c_str(), argv.data());

        // If we get here, exec failed - write error to stderr (which goes to parent)
        fprintf(stderr, "execvp failed for '%s': %s\n", config_.command.c_str(), strerror(errno));
        _exit(127);
    }

    // Parent process

    // Close unused ends
    close(stdin_pipe[0]);   // Close read end of stdin
    close(stdout_pipe[1]);  // Close write end of stdout

    // Wait briefly to check if child process failed to exec
    // If execvp fails, the child exits immediately with code 127
    // Keep this short - we just need to detect immediate exec failures
    // Docker containers will take longer to start but that's handled by request timeouts
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    int status;
    pid_t result = waitpid(pid, &status, WNOHANG);
    if (result > 0) {
        // Child process already exited
        if (WIFEXITED(status)) {
            int exit_code = WEXITSTATUS(status);
            SRV_ERR("%s: child process %s exited with code %d (command: %s)\n",
                    __func__, config_.name.c_str(), exit_code, config_.command.c_str());

            // Clean up pipes
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            return false;
        }
    }

    pid_ = pid;
    stdin_fd_ = stdin_pipe[1];
    stdout_fd_ = stdout_pipe[0];

    return true;
}

void mcp_process::terminate_process() {
    if (stdin_fd_ >= 0) {
        close(stdin_fd_);
        stdin_fd_ = -1;
    }
    if (stdout_fd_ >= 0) {
        close(stdout_fd_);
        stdout_fd_ = -1;
    }
    if (pid_ > 0) {
        // Send SIGTERM first
        kill(pid_, SIGTERM);

        // Wait up to 5 seconds
        int status;
        int attempts = 50;
        while (attempts-- > 0) {
            pid_t result = waitpid(pid_, &status, WNOHANG);
            if (result == pid_) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Force kill if still running
        if (attempts == 0) {
            kill(pid_, SIGKILL);
            waitpid(pid_, &status, 0);
        }

        pid_ = -1;
    }
}

bool mcp_process::platform_write(const std::string & data) {
    ssize_t written = ::write(stdin_fd_, data.data(), data.size());
    if (written < 0) {
        SRV_ERR("%s: write failed: %s\n", __func__, strerror(errno));
        running_ = false;
        return false;
    }
    if (static_cast<size_t>(written) != data.size()) {
        SRV_WRN("%s: partial write: %zd of %zu\n", __func__, written, data.size());
    }
    ::fsync(stdin_fd_);
    return true;
}

#endif

std::unique_ptr<mcp_process> mcp_process_factory::create(const mcp_server_config & config) {
    return std::make_unique<mcp_process>(config);
}
