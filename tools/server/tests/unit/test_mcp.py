#!/usr/bin/env python
"""
Tests for MCP (Model Context Protocol) server functionality.

Tests cover:
- HTTP endpoints: /mcp/servers, /mcp/ws-port
- WebSocket connection lifecycle
- MCP JSON-RPC protocol
"""

import pytest
import json
import os
import tempfile
from pathlib import Path
import sys

# Ensure parent path is in sys.path
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import ServerProcess, ServerPreset

import websocket

server: ServerProcess

TIMEOUT_SERVER_START = 60
TIMEOUT_HTTP_REQUEST = 10
TIMEOUT_WS = 5


@pytest.fixture(autouse=True)
def create_server():
    """Create a basic server for MCP testing."""
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 8082  # Use different port to avoid conflicts
    server.n_slots = 1
    server.n_ctx = 2048
    yield
    server.stop()


class TestMcpHttpEndpoints:
    """Test MCP HTTP endpoints."""

    def test_mcp_servers_endpoint_no_config(self):
        """Test /mcp/servers returns empty list when no config is set."""
        server.start(timeout_seconds=TIMEOUT_SERVER_START)

        res = server.make_request("GET", "/mcp/servers", timeout=TIMEOUT_HTTP_REQUEST)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"

        body = res.body
        assert "servers" in body, f"Expected 'servers' key in response: {body}"
        # Without config, should return empty list or available servers
        assert isinstance(body["servers"], list), f"Expected list, got {type(body['servers'])}"

    def test_mcp_ws_port_endpoint(self):
        """Test /mcp/ws-port returns the WebSocket port."""
        server.start(timeout_seconds=TIMEOUT_SERVER_START)

        res = server.make_request("GET", "/mcp/ws-port", timeout=TIMEOUT_HTTP_REQUEST)
        assert res.status_code == 200, f"Expected 200, got {res.status_code}"

        body = res.body
        assert "port" in body, f"Expected 'port' key in response: {body}"

        # WebSocket port should be HTTP port + 1
        expected_ws_port = server.server_port + 1
        assert body["port"] == expected_ws_port, \
            f"Expected WebSocket port {expected_ws_port}, got {body['port']}"


class TestMcpWithConfig:
    """Test MCP with a configuration file."""

    @pytest.fixture
    def mcp_config_file(self):
        """Create a temporary MCP config file."""
        config = {
            "mcpServers": {
                "echo-test": {
                    "command": "echo",
                    "args": ["test"],
                    "env": {}
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as f:
            json.dump(config, f)
            config_path = f.name

        yield config_path

        # Cleanup
        try:
            os.unlink(config_path)
        except:
            pass

    def test_mcp_servers_with_config(self, mcp_config_file):
        """Test /mcp/servers returns configured servers."""
        # Set environment variable for MCP config
        old_env = os.environ.get('LLAMA_MCP_CONFIG')
        os.environ['LLAMA_MCP_CONFIG'] = mcp_config_file

        try:
            server.start(timeout_seconds=TIMEOUT_SERVER_START)

            res = server.make_request("GET", "/mcp/servers", timeout=TIMEOUT_HTTP_REQUEST)
            assert res.status_code == 200, f"Expected 200, got {res.status_code}"

            body = res.body
            assert "servers" in body, f"Expected 'servers' key in response: {body}"
            servers = body["servers"]

            # Should have the echo-test server
            assert "echo-test" in servers, f"Expected 'echo-test' in servers: {servers}"
        finally:
            # Restore environment
            if old_env is not None:
                os.environ['LLAMA_MCP_CONFIG'] = old_env
            elif 'LLAMA_MCP_CONFIG' in os.environ:
                del os.environ['LLAMA_MCP_CONFIG']


class TestMcpWebSocket:
    """Test MCP WebSocket functionality."""

    def test_websocket_connection_without_server_param(self):
        """Test WebSocket connection fails without server parameter."""
        server.start(timeout_seconds=TIMEOUT_SERVER_START)

        ws_port = server.server_port + 1
        ws_url = f"ws://{server.server_host}:{ws_port}/mcp"

        # Should fail or close quickly without server parameter
        ws = websocket.create_connection(ws_url, timeout=TIMEOUT_WS)
        try:
            # Expect the server to close the connection or send an error
            result = ws.recv()
            data = json.loads(result)
            assert "error" in data or data.get("error") is not None, \
                f"Expected error response without server param: {data}"
        except websocket.WebSocketConnectionClosedException:
            # This is expected - server closes connection without valid server param
            pass
        finally:
            ws.close()

    def test_websocket_connection_invalid_server(self):
        """Test WebSocket connection with invalid server name."""
        server.start(timeout_seconds=TIMEOUT_SERVER_START)

        ws_port = server.server_port + 1
        ws_url = f"ws://{server.server_host}:{ws_port}/mcp?server=nonexistent"

        ws = websocket.create_connection(ws_url, timeout=TIMEOUT_WS)
        try:
            # Expect error message about unknown server
            result = ws.recv()
            data = json.loads(result)
            assert "error" in data or "error" in str(data).lower(), \
                f"Expected error for invalid server: {data}"
        except websocket.WebSocketConnectionClosedException:
            # Also acceptable - server may just close connection
            pass
        finally:
            ws.close()


class TestMcpJsonRpcProtocol:
    """Test MCP JSON-RPC protocol handling."""

    @pytest.fixture
    def python_mcp_config(self):
        """Create config with a Python-based echo MCP server for testing."""
        # Create a simple Python script that acts as an MCP server
        script_content = '''#!/usr/bin/env python3
import sys
import json

def main():
    # Read JSON-RPC messages from stdin, respond on stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            method = request.get("method", "")
            req_id = request.get("id")

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "test-echo", "version": "1.0.0"}
                    }
                }
            elif method == "tools/list":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "tools": [{
                            "name": "echo",
                            "description": "Echo back the input",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {"type": "string"}
                                },
                                "required": ["message"]
                            }
                        }]
                    }
                }
            elif method == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name", "")
                args = params.get("arguments", {})
                if tool_name == "echo":
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": args.get("message", "")}]
                        }
                    }
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"}
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                }

            print(json.dumps(response), flush=True)
        except Exception as e:
            if req_id:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32603, "message": str(e)}
                }
                print(json.dumps(error_response), flush=True)

if __name__ == "__main__":
    main()
'''

        # Create temp script file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as script_file:
            script_file.write(script_content)
            script_path = script_file.name

        # Make it executable
        os.chmod(script_path, 0o755)

        # Create config pointing to the script
        config = {
            "mcpServers": {
                "test-echo": {
                    "command": sys.executable,
                    "args": [script_path],
                    "env": {"PYTHONUNBUFFERED": "1"}
                }
            }
        }

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        ) as config_file:
            json.dump(config, config_file)
            config_path = config_file.name

        yield config_path, script_path

        # Cleanup
        try:
            os.unlink(config_path)
            os.unlink(script_path)
        except:
            pass

    def test_mcp_initialize_handshake(self, python_mcp_config):
        """Test MCP initialize handshake via WebSocket."""
        config_path, _ = python_mcp_config

        old_env = os.environ.get('LLAMA_MCP_CONFIG')
        os.environ['LLAMA_MCP_CONFIG'] = config_path

        try:
            server.start(timeout_seconds=TIMEOUT_SERVER_START)

            ws_port = server.server_port + 1
            ws_url = f"ws://{server.server_host}:{ws_port}/mcp?server=test-echo"

            ws = websocket.create_connection(ws_url, timeout=TIMEOUT_WS)
            try:
                # Send initialize request
                init_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"}
                    }
                }
                ws.send(json.dumps(init_request))

                # Receive response
                result = ws.recv()
                response = json.loads(result)

                assert response.get("id") == 1, f"Expected id=1: {response}"
                assert "result" in response, f"Expected result in response: {response}"
                assert "protocolVersion" in response["result"], \
                    f"Expected protocolVersion in result: {response}"

            finally:
                ws.close()
        finally:
            if old_env is not None:
                os.environ['LLAMA_MCP_CONFIG'] = old_env
            elif 'LLAMA_MCP_CONFIG' in os.environ:
                del os.environ['LLAMA_MCP_CONFIG']

    def test_mcp_tools_list(self, python_mcp_config):
        """Test MCP tools/list method."""
        config_path, _ = python_mcp_config

        old_env = os.environ.get('LLAMA_MCP_CONFIG')
        os.environ['LLAMA_MCP_CONFIG'] = config_path

        try:
            server.start(timeout_seconds=TIMEOUT_SERVER_START)

            ws_port = server.server_port + 1
            ws_url = f"ws://{server.server_host}:{ws_port}/mcp?server=test-echo"

            ws = websocket.create_connection(ws_url, timeout=TIMEOUT_WS)
            try:
                # Initialize first
                init_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"}
                    }
                }
                ws.send(json.dumps(init_request))
                ws.recv()  # Consume init response

                # Send initialized notification
                ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }))

                # Request tools list
                tools_request = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
                ws.send(json.dumps(tools_request))

                result = ws.recv()
                response = json.loads(result)

                assert response.get("id") == 2, f"Expected id=2: {response}"
                assert "result" in response, f"Expected result: {response}"
                assert "tools" in response["result"], f"Expected tools: {response}"

                tools = response["result"]["tools"]
                assert len(tools) > 0, f"Expected at least one tool: {tools}"
                assert tools[0]["name"] == "echo", f"Expected echo tool: {tools}"

            finally:
                ws.close()
        finally:
            if old_env is not None:
                os.environ['LLAMA_MCP_CONFIG'] = old_env
            elif 'LLAMA_MCP_CONFIG' in os.environ:
                del os.environ['LLAMA_MCP_CONFIG']

    def test_mcp_tool_call(self, python_mcp_config):
        """Test MCP tools/call method."""
        config_path, _ = python_mcp_config

        old_env = os.environ.get('LLAMA_MCP_CONFIG')
        os.environ['LLAMA_MCP_CONFIG'] = config_path

        try:
            server.start(timeout_seconds=TIMEOUT_SERVER_START)

            ws_port = server.server_port + 1
            ws_url = f"ws://{server.server_host}:{ws_port}/mcp?server=test-echo"

            ws = websocket.create_connection(ws_url, timeout=TIMEOUT_WS)
            try:
                # Initialize
                ws.send(json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0.0"}
                    }
                }))
                ws.recv()

                # Call echo tool
                call_request = {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "echo",
                        "arguments": {"message": "Hello, MCP!"}
                    }
                }
                ws.send(json.dumps(call_request))

                result = ws.recv()
                response = json.loads(result)

                assert response.get("id") == 3, f"Expected id=3: {response}"
                assert "result" in response, f"Expected result: {response}"
                assert "content" in response["result"], f"Expected content: {response}"

                content = response["result"]["content"]
                assert len(content) > 0, f"Expected content items: {content}"
                assert content[0]["text"] == "Hello, MCP!", \
                    f"Expected echoed message: {content}"

            finally:
                ws.close()
        finally:
            if old_env is not None:
                os.environ['LLAMA_MCP_CONFIG'] = old_env
            elif 'LLAMA_MCP_CONFIG' in os.environ:
                del os.environ['LLAMA_MCP_CONFIG']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
