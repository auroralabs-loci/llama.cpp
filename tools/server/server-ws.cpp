#include "server-ws.h"
#include "common.h"
#include "log.h"
#include "arg.h"

#include <cstring>
#include <random>
#include <iomanip>
#include <sstream>
#include <algorithm>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET socket_t;
#define INVALID_SOCKET_VALUE INVALID_SOCKET
#define SOCKET_ERROR_VALUE SOCKET_ERROR
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
typedef int socket_t;
#define INVALID_SOCKET_VALUE -1
#define SOCKET_ERROR_VALUE -1
#endif

// WebSocket frame constants
namespace ws_frame {
    constexpr uint8_t FIN_BIT = 0x80;
    constexpr uint8_t MASK_BIT = 0x80;

    enum opcode : uint8_t {
        CONTINUATION = 0x0,
        TEXT = 0x1,
        BINARY = 0x2,
        CLOSE = 0x8,
        PING = 0x9,
        PONG = 0xa
    };
}

// Rotate left helper (must be defined before sha1)
static inline uint32_t rotl(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// Forward declarations
static std::string base64_encode(const unsigned char * data, size_t len);

// Simple SHA-1 implementation
static std::vector<unsigned char> sha1(const std::string & input) {
    // SHA-1 constants
    static const uint32_t k[4] = {
        0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6
    };

    // Pad the message
    uint64_t bit_len = input.size() * 8;
    std::vector<uint8_t> padded(input.begin(), input.end());
    padded.push_back(0x80);

    while ((padded.size() % 64) != 56) {
        padded.push_back(0x00);
    }

    // Add length as 64-bit big-endian
    for (int i = 7; i >= 0; i--) {
        padded.push_back((bit_len >> (i * 8)) & 0xff);
    }

    // Process in 64-byte chunks
    std::vector<uint32_t> h = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0};

    for (size_t chunk = 0; chunk < padded.size(); chunk += 64) {
        uint32_t w[80] = {};

        // Break chunk into 16 words
        for (int i = 0; i < 16; i++) {
            w[i] = (padded[chunk + i * 4] << 24) |
                   (padded[chunk + i * 4 + 1] << 16) |
                   (padded[chunk + i * 4 + 2] << 8) |
                   (padded[chunk + i * 4 + 3]);
        }

        // Extend to 80 words
        for (int i = 16; i < 80; i++) {
            w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        }

        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];

        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5a827999;
            } else if (i < 40) {
                f = b ^ c ^ d;
                k = 0x6ed9eba1;
            } else if (i < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8f1bbcdc;
            } else {
                f = b ^ c ^ d;
                k = 0xca62c1d6;
            }

            uint32_t temp = rotl(a, 5) + f + e + k + w[i];
            e = d;
            d = c;
            c = rotl(b, 30);
            b = a;
            a = temp;
        }

        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }

    // Convert to big-endian bytes
    std::vector<unsigned char> result(20);
    for (int i = 0; i < 5; i++) {
        result[i * 4] = (h[i] >> 24) & 0xff;
        result[i * 4 + 1] = (h[i] >> 16) & 0xff;
        result[i * 4 + 2] = (h[i] >> 8) & 0xff;
        result[i * 4 + 3] = h[i] & 0xff;
    }

    return result;
}

// Simple WebSocket implementation using raw sockets
class ws_connection_impl : public server_ws_connection {
public:
    ws_connection_impl(socket_t sock, const std::string & path, const std::string & query)
        : sock_(sock)
        , path_(path)
        , query_(query)
        , closed_(false) {
        parse_query_params();
    }

    ~ws_connection_impl() override {
        close(1000, "");
    }

    void send(const std::string & message) override {
        if (closed_) {
            SRV_WRN("%s: cannot send, connection closed: %s\n", __func__, get_remote_address().c_str());
            return;
        }

        // Create WebSocket text frame
        std::vector<uint8_t> frame;

        uint8_t first_byte = ws_frame::FIN_BIT | ws_frame::TEXT;
        frame.push_back(first_byte);

        size_t len = message.size();
        if (len < 126) {
            frame.push_back(static_cast<uint8_t>(len));
        } else if (len < 65536) {
            frame.push_back(126);
            frame.push_back(static_cast<uint8_t>((len >> 8) & 0xff));
            frame.push_back(static_cast<uint8_t>(len & 0xff));
        } else {
            frame.push_back(127);
            for (int i = 7; i >= 0; i--) {
                frame.push_back(static_cast<uint8_t>((len >> (i * 8)) & 0xff));
            }
        }

        frame.insert(frame.end(), message.begin(), message.end());

        // Log frame header for debugging
        SRV_INF("%s: %s: frame_header=[%02x %02x %02x %02x ...] payload_size=%zd\n",
                __func__, get_remote_address().c_str(),
                frame.size() > 0 ? frame[0] : 0,
                frame.size() > 1 ? frame[1] : 0,
                frame.size() > 2 ? frame[2] : 0,
                frame.size() > 3 ? frame[3] : 0,
                len);

        // Send frame
#ifdef _WIN32
        int sent = ::send(sock_, reinterpret_cast<const char *>(frame.data()), static_cast<int>(frame.size()), 0);
#else
        ssize_t sent = ::send(sock_, frame.data(), frame.size(), 0);
#endif
        SRV_INF("%s: %s: frame_size=%zd sent=%zd\n", __func__, get_remote_address().c_str(), frame.size(), sent);
        if (sent < 0) {
            SRV_ERR("%s: send failed: %s (errno=%d)\n", __func__, get_remote_address().c_str(), errno);
            closed_ = true;
        } else if (static_cast<size_t>(sent) != frame.size()) {
            SRV_WRN("%s: partial send: %zd/%zd bytes\n", __func__, sent, frame.size());
        }
    }

    void close(int code, const std::string & reason) override {
        if (closed_) {
            return;
        }

        // Send close frame
        std::vector<uint8_t> close_frame;
        close_frame.push_back(ws_frame::FIN_BIT | ws_frame::CLOSE);
        close_frame.push_back(2 + reason.size());
        close_frame.push_back(static_cast<uint8_t>((code >> 8) & 0xff));
        close_frame.push_back(static_cast<uint8_t>(code & 0xff));
        close_frame.insert(close_frame.end(), reason.begin(), reason.end());

#ifdef _WIN32
        ::send(sock_, reinterpret_cast<const char *>(close_frame.data()), static_cast<int>(close_frame.size()), 0);
        closesocket(sock_);
#else
        ::send(sock_, close_frame.data(), close_frame.size(), 0);
        ::close(sock_);
#endif

        closed_ = true;
    }

    std::string get_query_param(const std::string & key) const override {
        auto it = query_params_.find(key);
        if (it != query_params_.end()) {
            return it->second;
        }
        return "";
    }

    std::string get_remote_address() const override {
        return remote_address_;
    }

    socket_t socket() const { return sock_; }
    bool is_closed() const { return closed_; }

    // Handle incoming data
    void handle_data(const std::vector<uint8_t> & data, std::function<void(const std::string &)> on_message) {
        receive_buffer_.insert(receive_buffer_.end(), data.begin(), data.end());

        while (true) {
            if (receive_buffer_.size() < 2) {
                break;  // Need at least 2 bytes for header
            }

            uint8_t first_byte = receive_buffer_[0];
            uint8_t second_byte = receive_buffer_[1];

            bool fin = (first_byte & 0x80) != 0;
            ws_frame::opcode opcode = static_cast<ws_frame::opcode>(first_byte & 0x0f);
            bool masked = (second_byte & 0x80) != 0;
            uint64_t payload_len = second_byte & 0x7f;

            size_t header_len = 2;

            if (payload_len == 126) {
                if (receive_buffer_.size() < 4) break;
                payload_len = (static_cast<uint64_t>(receive_buffer_[2]) << 8) |
                              static_cast<uint64_t>(receive_buffer_[3]);
                header_len = 4;
            } else if (payload_len == 127) {
                if (receive_buffer_.size() < 10) break;
                payload_len = 0;
                for (int i = 0; i < 8; i++) {
                    payload_len = (payload_len << 8) | receive_buffer_[2 + i];
                }
                header_len = 10;
            }

            size_t total_len = header_len + payload_len + (masked ? 4 : 0);
            if (receive_buffer_.size() < total_len) {
                break;  // Incomplete frame
            }

            // Extract payload (skip the mask if present)
            size_t payload_offset = header_len + (masked ? 4 : 0);
            std::vector<uint8_t> payload(receive_buffer_.begin() + payload_offset,
                                         receive_buffer_.begin() + payload_offset + payload_len);

            // Unmask if needed
            if (masked) {
                uint8_t mask[4];
                std::memcpy(mask, &receive_buffer_[header_len], 4);
                SRV_DBG("%s: unmasking payload with mask: [0x%02x, 0x%02x, 0x%02x, 0x%02x]\n",
                        __func__, mask[0], mask[1], mask[2], mask[3]);
                for (size_t i = 0; i < payload_len; i++) {
                    uint8_t masked = payload[i];
                    payload[i] ^= mask[i % 4];
                    if (i < 20) {
                        SRV_DBG("%s:   [%zu] masked=0x%02x unmasked=0x%02x ('%c')\n",
                                __func__, i, masked, payload[i],
                                isprint(payload[i]) ? payload[i] : '.');
                    }
                }
                SRV_DBG("%s: first 20 chars of payload: '%.*s'\n",
                        __func__, (int)std::min(size_t(20), payload.size()),
                        payload.data());
            }

            // Remove processed frame from buffer
            receive_buffer_.erase(receive_buffer_.begin(),
                                  receive_buffer_.begin() + total_len);

            // Handle frame
            if (opcode == ws_frame::TEXT || opcode == ws_frame::CONTINUATION) {
                if (fin) {
                    // Complete message
                    message_buffer_.insert(message_buffer_.end(), payload.begin(), payload.end());
                    std::string msg(message_buffer_.begin(), message_buffer_.end());
                    on_message(msg);
                    message_buffer_.clear();
                } else {
                    // Fragmented message
                    message_buffer_.insert(message_buffer_.end(), payload.begin(), payload.end());
                }
            } else if (opcode == ws_frame::PING) {
                // Respond with pong
                send_pong(payload);
            } else if (opcode == ws_frame::CLOSE) {
                close(1000, "Normal closure");
                break;
            }
        }
    }

    void set_remote_address(const std::string & addr) {
        remote_address_ = addr;
    }

private:
    socket_t sock_;
    std::string path_;
    std::string query_;
    std::string remote_address_;
    std::map<std::string, std::string> query_params_;
    bool closed_;

    std::vector<uint8_t> receive_buffer_;
    std::vector<uint8_t> message_buffer_;

    void parse_query_params() {
        std::string remaining = query_;
        while (!remaining.empty()) {
            size_t amp_pos = remaining.find('&');
            std::string pair;
            if (amp_pos == std::string::npos) {
                pair = remaining;
                remaining.clear();
            } else {
                pair = remaining.substr(0, amp_pos);
                remaining = remaining.substr(amp_pos + 1);
            }

            size_t eq_pos = pair.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = pair.substr(0, eq_pos);
                std::string value = pair.substr(eq_pos + 1);
                query_params_[key] = value;
            }
        }
    }

    void send_pong(const std::vector<uint8_t> & payload) {
        std::vector<uint8_t> frame;
        frame.push_back(ws_frame::FIN_BIT | ws_frame::PONG);
        frame.push_back(static_cast<uint8_t>(payload.size()));
        frame.insert(frame.end(), payload.begin(), payload.end());

#ifdef _WIN32
        ::send(sock_, reinterpret_cast<const char *>(frame.data()), static_cast<int>(frame.size()), 0);
#else
        ::send(sock_, frame.data(), frame.size(), 0);
#endif
    }
};

struct server_ws_context::Impl {
    socket_t listen_sock = INVALID_SOCKET_VALUE;
    std::atomic<bool> running{false};
    std::thread accept_thread;
    std::mutex connections_mutex;
    std::map<void*, std::shared_ptr<ws_connection_impl>> connections;

    server_ws_context::on_open_t on_open_cb;
    server_ws_context::on_message_t on_message_cb;
    server_ws_context::on_close_t on_close_cb;

    int port = 0;
    std::string path_prefix = "/mcp";

    // Methods
    void accept_loop();
    void handle_connection(socket_t sock, const struct sockaddr_in & addr);
};

server_ws_context::server_ws_context()
    : pimpl(std::make_unique<Impl>()) {
}

server_ws_context::~server_ws_context() {
    stop();
}

bool server_ws_context::init(const common_params & params) {
    // Use port + 1 from the HTTP server to avoid conflicts
    // This provides a predictable port for the frontend
    pimpl->port = params.port + 1;
    pimpl->path_prefix = "/mcp";

    SRV_INF("%s: WebSocket context initialized\n", __func__);
    return true;
}

bool server_ws_context::start() {
    if (pimpl->running) {
        SRV_WRN("%s: WebSocket server already running\n", __func__);
        return true;
    }

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        SRV_ERR("%s: WSAStartup failed\n", __func__);
        return false;
    }
#endif

    // Create listening socket
    pimpl->listen_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (pimpl->listen_sock == INVALID_SOCKET_VALUE) {
        SRV_ERR("%s: socket() failed\n", __func__);
        return false;
    }

    // Set SO_REUSEADDR
    int opt = 1;
#ifdef _WIN32
    setsockopt(pimpl->listen_sock, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));
#else
    setsockopt(pimpl->listen_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    // Bind to address
    struct sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(pimpl->port);

    if (bind(pimpl->listen_sock, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
        SRV_ERR("%s: bind() failed on port %d\n", __func__, pimpl->port);
#ifdef _WIN32
        closesocket(pimpl->listen_sock);
#else
        close(pimpl->listen_sock);
#endif
        return false;
    }

    // Listen
    if (listen(pimpl->listen_sock, SOMAXCONN) < 0) {
        SRV_ERR("%s: listen() failed\n", __func__);
#ifdef _WIN32
        closesocket(pimpl->listen_sock);
#else
        close(pimpl->listen_sock);
#endif
        return false;
    }

    // Get actual port
    struct sockaddr_in actual_addr;
    socklen_t len = sizeof(actual_addr);
    getsockname(pimpl->listen_sock, reinterpret_cast<struct sockaddr *>(&actual_addr), &len);
    pimpl->port = ntohs(actual_addr.sin_port);

    std::ostringstream oss;
    oss << "ws://" << actual_addr.sin_addr.s_addr << ":" << pimpl->port;
    listening_address = oss.str();

    pimpl->running = true;

    // Start accept thread
    pimpl->accept_thread = std::thread([this]() {
        pimpl->accept_loop();
    });

    is_ready = true;

    SRV_INF("%s: WebSocket server started on port %d\n", __func__, pimpl->port);
    return true;
}

void server_ws_context::stop() {
    if (!pimpl->running) {
        return;
    }

    pimpl->running = false;

    // Close all connections
    {
        std::lock_guard<std::mutex> lock(pimpl->connections_mutex);
        for (auto & [key, conn] : pimpl->connections) {
            conn->close(1001, "Server shutdown");
        }
        pimpl->connections.clear();
    }

    // Close listening socket
    if (pimpl->listen_sock != INVALID_SOCKET_VALUE) {
#ifdef _WIN32
        closesocket(pimpl->listen_sock);
#else
        close(pimpl->listen_sock);
#endif
        pimpl->listen_sock = INVALID_SOCKET_VALUE;
    }

    // Wait for accept thread
    if (pimpl->accept_thread.joinable()) {
        pimpl->accept_thread.join();
    }

    is_ready.store(false);

    SRV_INF("%s: WebSocket server stopped\n", __func__);
}

int server_ws_context::get_actual_port() const {
    return pimpl->port;
}

void server_ws_context::on_open(on_open_t handler) {
    pimpl->on_open_cb = std::move(handler);
}

void server_ws_context::on_message(on_message_t handler) {
    pimpl->on_message_cb = std::move(handler);
}

void server_ws_context::on_close(on_close_t handler) {
    pimpl->on_close_cb = std::move(handler);
}

void server_ws_context::Impl::accept_loop() {
    while (running) {
        struct sockaddr_in client_addr = {};
        socklen_t client_len = sizeof(client_addr);

        socket_t client_sock = accept(listen_sock,
                                      reinterpret_cast<struct sockaddr *>(&client_addr),
                                      &client_len);

        if (client_sock == INVALID_SOCKET_VALUE) {
            if (running) {
                SRV_ERR("%s: accept() failed\n", __func__);
            }
            continue;
        }

        // Handle connection in a thread
        std::thread([this, client_sock, client_addr]() {
            this->handle_connection(client_sock, client_addr);
        }).detach();
    }
}

void server_ws_context::Impl::handle_connection(socket_t sock, const struct sockaddr_in & addr) {
    // Set socket options
    int flag = 1;
#ifdef _WIN32
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<const char *>(&flag), sizeof(flag));
#else
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));
#endif

    // Read HTTP request (WebSocket handshake)
    std::vector<uint8_t> buffer(4096);
#ifdef _WIN32
    int recv_len = recv(sock, reinterpret_cast<char *>(buffer.data()), buffer.size() - 1, 0);
#else
    ssize_t recv_len = recv(sock, buffer.data(), buffer.size() - 1, 0);
#endif

    if (recv_len <= 0) {
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return;
    }

    buffer[recv_len] = '\0';
    std::string request(reinterpret_cast<char *>(buffer.data()));

    // Parse HTTP request
    std::istringstream iss(request);
    std::string method, path, version;
    iss >> method >> path >> version;

    if (method != "GET") {
        // Bad request
        const char * response = "HTTP/1.1 400 Bad Request\r\n\r\n";
        send(sock, response, strlen(response), 0);
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return;
    }

    // Parse path and query
    std::string query;
    size_t qpos = path.find('?');
    if (qpos != std::string::npos) {
        query = path.substr(qpos + 1);
        path = path.substr(0, qpos);
    }

    // Check if path matches our prefix
    if (path != path_prefix) {
        // Not found
        const char * response = "HTTP/1.1 404 Not Found\r\n\r\n";
        send(sock, response, strlen(response), 0);
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return;
    }

    // After iss >> method >> path >> version, consume the remaining empty line
    // The stream is positioned right after "HTTP/1.1", the next line is just "\r\n"
    std::string line;  // Declare line for use below
    std::getline(iss, line);

    // Extract headers (case-insensitive matching)
    std::string websocket_key;
    while (std::getline(iss, line)) {
        // Trim trailing \r (common in HTTP headers)
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        // Empty line marks end of headers
        if (line.empty()) {
            break;
        }
        // Case-insensitive header matching
        // Convert line to lowercase for comparison, but preserve original for value extraction
        std::string line_lower = line;
        for (char & c : line_lower) {
            if (c >= 'A' && c <= 'Z') {
                c = c + 32; // tolower
            }
        }
        if (line_lower.substr(0, 18) == "sec-websocket-key:") {
            if (line.length() > 19) {
                websocket_key = line.substr(19);
                // Trim leading spaces
                while (!websocket_key.empty() && websocket_key[0] == ' ') {
                    websocket_key.erase(0, 1);
                }
            }
        }
    }

    if (websocket_key.empty()) {
        const char * response = "HTTP/1.1 400 Bad Request\r\n\r\n";
        send(sock, response, strlen(response), 0);
#ifdef _WIN32
        closesocket(sock);
#else
        close(sock);
#endif
        return;
    }

    // Compute accept key
    std::string magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    std::string combined = websocket_key + magic;

    // SHA-1 hash
    std::vector<unsigned char> hash = sha1(combined);

    // Base64 encode
    std::string accept_key = base64_encode(hash.data(), hash.size());

    std::ostringstream full_response_debug;
    full_response_debug << "HTTP/1.1 101 Switching Protocols\r\n";
    full_response_debug << "Upgrade: websocket\r\n";
    full_response_debug << "Connection: Upgrade\r\n";
    full_response_debug << "Sec-WebSocket-Accept: " << accept_key << "\r\n\r\n";
    std::string full_response_str = full_response_debug.str();

    SRV_INF("%s: hash_size=%zu, accept_key_len=%zu, last_4_chars=",
            __func__, hash.size(), accept_key.length());
    if (accept_key.length() >= 4) {
        for (size_t i = accept_key.length() - 4; i < accept_key.length(); i++) {
            SRV_INF("    [%zu]='%c' (0x%02x)\n", i, accept_key[i], (unsigned char)accept_key[i]);
        }
    }

    // Send handshake response
    std::ostringstream response;
    response << "HTTP/1.1 101 Switching Protocols\r\n";
    response << "Upgrade: websocket\r\n";
    response << "Connection: Upgrade\r\n";
    response << "Sec-WebSocket-Accept: " << accept_key << "\r\n\r\n";

    std::string response_str = response.str();
    SRV_DBG("%s: Sending 101 response, %zu bytes\n", __func__, response_str.size());
    size_t total_sent = 0;
    size_t to_send = response_str.size();
    while (total_sent < to_send) {
#ifdef _WIN32
        int sent = ::send(sock, response_str.c_str() + total_sent, static_cast<int>(to_send - total_sent), 0);
#else
        ssize_t sent = ::send(sock, response_str.c_str() + total_sent, to_send - total_sent, 0);
#endif
        if (sent <= 0) {
            SRV_ERR("%s: send() failed during handshake\n", __func__);
            break;
        }
        total_sent += sent;
    }

    // Create connection object
    auto conn = std::make_shared<ws_connection_impl>(sock, path, query);

    // Set remote address
    char addr_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &addr.sin_addr, addr_str, sizeof(addr_str));
    conn->set_remote_address(addr_str);

    // Store connection
    {
        std::lock_guard<std::mutex> lock(connections_mutex);
        connections[conn.get()] = conn;
    }

    // Call on_open callback
    if (on_open_cb) {
        SRV_INF("%s: About to call on_open_cb for server: %s\n", __func__, query.c_str());
        on_open_cb(conn);
        SRV_INF("%s: Returned from on_open_cb for server: %s\n", __func__, query.c_str());
    }

    // Read loop
    SRV_INF("%s: Entering read loop for server: %s\n", __func__, query.c_str());
    std::vector<uint8_t> recv_buf(4096);
    int recv_count = 0;
    while (!conn->is_closed()) {
#ifdef _WIN32
        int n = recv(sock, reinterpret_cast<char *>(recv_buf.data()), recv_buf.size(), 0);
#else
        ssize_t n = recv(sock, recv_buf.data(), recv_buf.size(), 0);
#endif

        if (n <= 0) {
            SRV_INF("%s: recv returned %zd (count=%d) for server: %s\n", __func__, n, recv_count, query.c_str());
            break;
        }
        SRV_INF("%s: recv returned %zd bytes (count=%d) for server: %s\n", __func__, n, ++recv_count, query.c_str());

        recv_buf.resize(n);
        SRV_DBG("%s: received %zd bytes from WebSocket\n", __func__, n);
        // Log first 100 bytes in hex
        for (ssize_t i = 0; i < std::min(ssize_t(100), n); i++) {
            SRV_DBG("%s:   [%zd] = 0x%02x (%c)\n", __func__, i, recv_buf[i],
                    isprint(recv_buf[i]) ? recv_buf[i] : '.');
        }

        conn->handle_data(recv_buf, [this, conn](const std::string & msg) {
            SRV_DBG("%s: handle_data callback: msg length=%zu\n", __func__, msg.length());
            if (on_message_cb) {
                on_message_cb(conn, msg);
            }
        });
    }

    // Call on_close callback
    if (on_close_cb) {
        on_close_cb(conn);
    }

    // Remove connection
    {
        std::lock_guard<std::mutex> lock(connections_mutex);
        connections.erase(conn.get());
    }

    // Close socket
    conn->close(1000, "");
}

// Base64 encoding helper function (static, not part of any class)
static std::string base64_encode(const unsigned char * data, size_t len) {
    static const char * table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t triple = (data[i] << 16) |
                         (i + 1 < len ? data[i + 1] << 8 : 0) |
                         (i + 2 < len ? data[i + 2] : 0);

        result.push_back(table[(triple >> 18) & 0x3f]);
        result.push_back(table[(triple >> 12) & 0x3f]);

        // For the third character, check if we have at least 2 bytes
        if (i + 1 < len) {
            result.push_back(table[(triple >> 6) & 0x3f]);
        } else {
            result.push_back('=');
        }

        // For the fourth character, check if we have at least 3 bytes
        if (i + 2 < len) {
            result.push_back(table[triple & 0x3f]);
        } else {
            result.push_back('=');
        }
    }

    return result;
}
