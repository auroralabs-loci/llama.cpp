/**
 * MCP Service - Manages WebSocket connections to MCP servers
 *
 * Handles communication with Model Context Protocol servers via WebSocket.
 * Each MCP server gets its own WebSocket connection.
 */

import type {
	McpJsonRpcRequest,
	McpJsonRpcResponse,
	McpJsonRpcNotification,
	McpTool
} from '$lib/types/mcp';
import { MCP_METHODS } from '$lib/types/mcp';

// Timeout constants - increased for Docker containers that may take time to start
const REQUEST_TIMEOUT_MS = 120_000; // 2 minutes for slow-starting containers
const MAX_RECONNECT_ATTEMPTS = 10; // More attempts for reliability
const MAX_RECONNECT_DELAY_MS = 60_000; // Max 60s between reconnect attempts
const INITIAL_RECONNECT_DELAY_MS = 1000; // Start with 1s delay

export class McpService {
	private ws: WebSocket | null = null;
	private pendingRequests = new Map<
		string,
		{
			resolve: (value: McpJsonRpcResponse) => void;
			reject: (error: Error) => void;
		}
	>();
	private requestId = 0;
	private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
	private reconnectAttempts = 0;
	private manualDisconnect = false;

	// Event callbacks
	onToolsChanged?: (tools: McpTool[]) => void;
	onNotification?: (notification: McpJsonRpcNotification) => void;
	onClose?: () => void;
	onError?: (error: Error) => void;
	onOpen?: () => void;

	constructor(
		public readonly serverName: string,
		private readonly wsUrl: string
	) {}

	/**
	 * Connect to the MCP server via WebSocket
	 */
	async connect(): Promise<void> {
		console.log(`[MCP] connect() called for: ${this.serverName}`);
		return new Promise((resolve, reject) => {
			try {
				this.ws = new WebSocket(this.wsUrl);
				console.log(`[MCP] WebSocket object created for: ${this.serverName}`);

				this.ws.onopen = async () => {
					console.log(`[MCP] Connected to server: ${this.serverName}`);
					this.reconnectAttempts = 0;
					try {
						await this.onOpen?.();
					} catch (err) {
						console.error(`[MCP] Error in onOpen callback for ${this.serverName}:`, err);
						this.onError?.(err instanceof Error ? err : new Error(String(err)));
					}
					resolve();
				};

				this.ws.onmessage = (event) => {
					this.handleMessage(event.data);
				};

				this.ws.onclose = (event) => {
					console.log(
						`[MCP] Disconnected from: ${this.serverName}, code: ${event.code}, reason: ${event.reason}, wasClean: ${event.wasClean}`
					);
					this.onClose?.();
					// Only auto-reconnect if not manually disconnected
					if (!this.manualDisconnect) {
						this.scheduleReconnect();
					}
					this.manualDisconnect = false; // Reset for next connection
				};

				this.ws.onerror = () => {
					console.error(`[MCP] WebSocket error for: ${this.serverName}`);
					reject(new Error(`Failed to connect to MCP server: ${this.serverName}`));
				};
			} catch (error) {
				console.error(`[MCP] Exception in connect() for ${this.serverName}:`, error);
				reject(error);
			}
		});
	}

	/**
	 * Disconnect from the MCP server
	 */
	disconnect() {
		console.log(`[MCP] disconnect() called for: ${this.serverName}, ws exists: ${!!this.ws}`);
		// Set flag to prevent auto-reconnect when onclose fires
		this.manualDisconnect = true;
		if (this.reconnectTimer) {
			clearTimeout(this.reconnectTimer);
			this.reconnectTimer = null;
		}
		this.ws?.close();
		this.ws = null;
		this.pendingRequests.clear();
	}

	/**
	 * Call a tool on the MCP server
	 */
	async callTool(name: string, args?: Record<string, unknown>): Promise<unknown> {
		const response = await this.sendRequest({
			method: MCP_METHODS.CALL_TOOL,
			params: { name, arguments: args }
		});

		if (response.error) {
			throw new Error(response.error.message);
		}

		return response.result;
	}

	/**
	 * List available tools from the MCP server
	 */
	async listTools(): Promise<McpTool[]> {
		const response = await this.sendRequest({
			method: MCP_METHODS.LIST_TOOLS,
			params: {}
		});

		if (response.error) {
			throw new Error(response.error.message);
		}

		const result = response.result as { tools?: McpTool[] } | undefined;
		return result?.tools || [];
	}

	/**
	 * Initialize the MCP session
	 */
	async initialize(capabilities: Record<string, unknown> = {}): Promise<void> {
		console.log(`[MCP] initialize() called for ${this.serverName}`);
		const response = await this.sendRequest({
			method: MCP_METHODS.INITIALIZE,
			params: {
				protocolVersion: '2024-11-05',
				capabilities,
				clientInfo: {
					name: 'llama.cpp-webui',
					version: '1.0.0'
				}
			}
		});
		console.log(`[MCP] initialize() got response for ${this.serverName}`);

		if (response.error) {
			throw new Error(response.error.message);
		}

		// Send initialized notification
		this.sendNotification({
			method: MCP_METHODS.INITIALIZED,
			params: {}
		});
		console.log(`[MCP] initialize() completed for ${this.serverName}`);
	}

	/**
	 * Send a ping to the MCP server
	 */
	async ping(): Promise<void> {
		const response = await this.sendRequest({
			method: MCP_METHODS.PING,
			params: {}
		});

		if (response.error) {
			throw new Error(response.error.message);
		}
	}

	/**
	 * Check if the connection is currently open
	 */
	isConnected(): boolean {
		return this.ws?.readyState === WebSocket.OPEN;
	}

	/**
	 * Send a JSON-RPC request and wait for response
	 */
	private async sendRequest(params: {
		method: string;
		params?: unknown;
	}): Promise<McpJsonRpcResponse> {
		const id = String(++this.requestId);
		const request: McpJsonRpcRequest = {
			jsonrpc: '2.0',
			id,
			method: params.method,
			params: params.params
		};

		return new Promise((resolve, reject) => {
			// Timeout for slow-starting containers (e.g., Docker)
			const timeout = setTimeout(() => {
				if (this.pendingRequests.delete(id)) {
					reject(new Error(`Request timeout: ${params.method}`));
				}
			}, REQUEST_TIMEOUT_MS);

			// Wrap resolve/reject to clear timeout
			this.pendingRequests.set(id, {
				resolve: (value) => {
					clearTimeout(timeout);
					resolve(value);
				},
				reject: (error) => {
					clearTimeout(timeout);
					reject(error);
				}
			});

			this.send(request);
		});
	}

	/**
	 * Send a JSON-RPC notification (no response expected)
	 */
	private sendNotification(params: { method: string; params?: unknown }): void {
		const notification: McpJsonRpcNotification = {
			jsonrpc: '2.0',
			method: params.method,
			params: params.params
		};
		this.send(notification);
	}

	/**
	 * Send a JSON-RPC message
	 */
	private send(message: McpJsonRpcRequest | McpJsonRpcNotification): void {
		console.log(`[MCP] Sending to ${this.serverName}:`, message);
		if (!this.isConnected()) {
			throw new Error(
				`WebSocket not connected for MCP server: ${this.serverName} (readyState: ${this.ws?.readyState})`
			);
		}

		const data = JSON.stringify(message);
		console.log(`[MCP] Sending JSON to ${this.serverName}:`, data);
		this.ws?.send(data);
		console.log(`[MCP] Sent to ${this.serverName}`);
	}

	/**
	 * Handle incoming WebSocket message
	 */
	private handleMessage(data: string) {
		console.log(`[MCP] Received from ${this.serverName}:`, data);
		let message: McpJsonRpcResponse | McpJsonRpcNotification;

		try {
			message = JSON.parse(data);
			console.log(`[MCP] Parsed message from ${this.serverName}:`, message);
		} catch (e) {
			console.error('[MCP] Failed to parse message:', data);
			console.error('[MCP] Parse error:', e);
			return;
		}

		// Response to a request (has id field)
		if ('id' in message && message.id !== undefined) {
			console.log(`[MCP] Response for id=${message.id}`);
			const pending = this.pendingRequests.get(String(message.id));
			if (pending) {
				this.pendingRequests.delete(String(message.id));
				if (message.error) {
					console.error(`[MCP] Error response:`, message.error);
					pending.reject(new Error(message.error.message));
				} else {
					console.log(`[MCP] Resolving pending request ${message.id}`);
					pending.resolve(message);
				}
			} else {
				console.warn(`[MCP] ${this.serverName}: No pending request for id=${message.id}`);
			}
			return;
		}

		// Notification (no id field)
		if ('method' in message) {
			console.log(`[MCP] Notification:`, message.method);
			this.onNotification?.(message as McpJsonRpcNotification);

			// Handle specific notifications
			if (message.method === MCP_METHODS.TOOLS_CHANGED) {
				this.listTools()
					.then((tools) => this.onToolsChanged?.(tools))
					.catch(console.error);
			}
		}
	}

	/**
	 * Schedule reconnection attempt
	 */
	private scheduleReconnect() {
		if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
			console.error(`[MCP] Max reconnect attempts reached for: ${this.serverName}`);
			this.onError?.(new Error('Max reconnect attempts reached'));
			return;
		}

		const delay = Math.min(
			INITIAL_RECONNECT_DELAY_MS * Math.pow(2, this.reconnectAttempts),
			MAX_RECONNECT_DELAY_MS
		);
		this.reconnectAttempts++;

		console.log(
			`[MCP] Scheduling reconnect attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${delay}ms`
		);

		this.reconnectTimer = setTimeout(() => {
			console.log(`[MCP] Attempting to reconnect to: ${this.serverName}`);
			this.connect().catch((err) => this.onError?.(err));
		}, delay);
	}
}

/**
 * Factory for creating McpService instances
 */
export const mcpServiceFactory = {
	/**
	 * Get the actual WebSocket port from the server
	 */
	async getWebSocketPort(): Promise<number> {
		try {
			const response = await fetch('/mcp/ws-port');
			const data = await response.json();
			return data.port;
		} catch {
			// Fallback: assume WebSocket is on HTTP port + 1
			const url = new URL(window.location.href);
			return parseInt(url.port) + 1;
		}
	},

	/**
	 * Create an MCP service for a given server name
	 */
	create: async (serverName: string): Promise<McpService> => {
		// Fetch the actual WebSocket port from the server
		const wsPort = await mcpServiceFactory.getWebSocketPort();
		const url = new URL(window.location.href);
		const wsUrl = `ws://${url.hostname}:${wsPort}/mcp?server=${encodeURIComponent(serverName)}`;
		return new McpService(serverName, wsUrl);
	}
};
