/**
 * MCP Store - Manages MCP server connections and tools
 *
 * This store manages connections to Model Context Protocol (MCP) servers.
 * It provides reactive state for connected MCP servers and their available tools.
 *
 * **Architecture & Relationships:**
 * - **McpService**: Stateless service for WebSocket communication
 * - **mcpStore** (this class): Reactive store for MCP connection state
 * - **Chat Components**: Can use MCP tools during generation
 *
 * **Key Features:**
 * - **Connection Management**: Connect/disconnect to MCP servers
 * - **Tool Discovery**: Automatically fetches available tools
 * - **Multiple Servers**: Support for multiple simultaneous MCP server connections
 */

import type { McpConnectionState, McpTool } from '$lib/types/mcp';
import { McpService } from '$lib/services/mcp';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

class McpStore {
	// ─────────────────────────────────────────────────────────────────────────────
	// State
	// ─────────────────────────────────────────────────────────────────────────────

	// Map of server name -> connection state
	connections = $state<SvelteMap<string, McpConnectionState>>(new SvelteMap());
	// Map of server name -> service instance
	private services = $state<SvelteMap<string, McpService>>(new SvelteMap());
	// Map of server name -> connection generation (to prevent stale callbacks)
	private connectionGenerations = new SvelteMap<string, number>();
	// Map of server name -> connection promise (to prevent concurrent connection attempts)
	private connectionPromises = new SvelteMap<string, Promise<void>>();
	// Loading state
	connecting = $state<SvelteSet<string>>(new SvelteSet());

	// ─────────────────────────────────────────────────────────────────────────────
	// Private Reactive Helpers
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Add to connecting set (creates new SvelteSet for Svelte 5 reactivity)
	 */
	private addConnecting(serverName: string): void {
		this.connecting = new SvelteSet([...this.connecting, serverName]);
	}

	/**
	 * Remove from connecting set (creates new SvelteSet for Svelte 5 reactivity)
	 */
	private removeConnecting(serverName: string): void {
		const newSet = new SvelteSet(this.connecting);
		newSet.delete(serverName);
		this.connecting = newSet;
	}
	// Provisional conversation ID (used before a real conversation exists)
	private provisionalConvId = $state<string | null>(null);
	// Real conversation ID that will replace provisional ID
	private realConvId = $state<string | null>(null);

	// ─────────────────────────────────────────────────────────────────────────────
	// Getters
	// ─────────────────────────────────────────────────────────────────────────────

	get connectedServers(): McpConnectionState[] {
		return Array.from(this.connections.values()).filter((s) => s.connected);
	}

	get serverNames(): string[] {
		return Array.from(this.connections.keys());
	}

	get toolsByServer(): Map<string, McpTool[]> {
		const map = new SvelteMap<string, McpTool[]>();
		for (const [name, state] of this.connections) {
			if (state.connected) {
				map.set(name, state.tools);
			}
		}
		return map;
	}

	get allTools(): Array<{ server: string; tool: McpTool }> {
		const result: Array<{ server: string; tool: McpTool }> = [];
		for (const [name, state] of this.connections) {
			if (state.connected) {
				for (const tool of state.tools) {
					result.push({ server: name, tool });
				}
			}
		}
		return result;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Connection Management
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Connect to an MCP server by name
	 */
	async connect(serverName: string): Promise<void> {
		// Check if there's already a connection in progress
		const existingPromise = this.connectionPromises.get(serverName);
		if (existingPromise) {
			console.log(`[MCP] Connection already in progress for: ${serverName}, waiting...`);
			return existingPromise;
		}

		// Check if already connected
		if (this.connections.has(serverName)) {
			const state = this.connections.get(serverName);
			if (state?.connected) {
				console.log(`[MCP] Already connected to: ${serverName}`);
				return;
			}
		}

		// Create connection promise
		const connectionPromise = (async () => {
			try {
				console.log(`[MCP] connect() starting for: ${serverName}`);

				// Check if there's an existing service
				const existingService = this.services.get(serverName);
				console.log(`[MCP] Existing service found: ${!!existingService}`);

				if (existingService) {
					// If the service exists and is already connected, we're done
					if (existingService.isConnected()) {
						const existingState = this.connections.get(serverName);
						console.log(
							`[MCP] Existing service connected: ${existingService.isConnected()}, state connected: ${existingState?.connected}`
						);
						if (existingState?.connected) {
							console.log(`[MCP] Service already connected for: ${serverName}, reusing connection`);
							this.removeConnecting(serverName);
							return;
						}
					}
					// Service exists but is disconnected - clean it up first
					console.log(`[MCP] Cleaning up existing disconnected service for: ${serverName}`);
					existingService.disconnect();
					// Wait for the WebSocket to finish closing before creating a new one
					console.log(`[MCP] Waiting for WebSocket to close before reconnecting to: ${serverName}`);
					await new Promise((resolve) => setTimeout(resolve, 500));
				}

				// Increment connection generation for this server
				const currentGen = (this.connectionGenerations.get(serverName) ?? 0) + 1;
				this.connectionGenerations.set(serverName, currentGen);

				// Add to connecting set (reactive)
				this.addConnecting(serverName);

				// Update connection state
				this.updateConnectionState(serverName, {
					name: serverName,
					connected: false,
					tools: [],
					error: undefined
				});

				try {
					// Fetch the actual WebSocket port from the server
					let wsPort: number;
					try {
						const response = await fetch('/mcp/ws-port');
						const data = await response.json();
						wsPort = data.port;
					} catch {
						// Fallback: assume WebSocket is on HTTP port + 1
						const url = new URL(window.location.href);
						wsPort = parseInt(url.port) + 1;
					}

					const wsUrl = `ws://${window.location.hostname}:${wsPort}/mcp?server=${encodeURIComponent(serverName)}`;
					const service = new McpService(serverName, wsUrl);
					this.services.set(serverName, service);

					// Set up event handlers with generation checking
					service.onToolsChanged = (tools) => {
						if (this.connectionGenerations.get(serverName) !== currentGen) {
							console.log(
								`[MCP] Ignoring tools changed callback from old connection for ${serverName}`
							);
							return;
						}
						console.log(`[MCP] Tools changed for ${serverName}:`, tools);
						this.updateConnectionState(serverName, { tools });
					};

					service.onError = (error) => {
						if (this.connectionGenerations.get(serverName) !== currentGen) {
							console.log(`[MCP] Ignoring error callback from old connection for ${serverName}`);
							return;
						}
						console.error(`[MCP] Error for ${serverName}:`, error);
						this.updateConnectionState(serverName, {
							error: error.message,
							connected: false
						});
					};

					service.onClose = () => {
						if (this.connectionGenerations.get(serverName) !== currentGen) {
							console.log(`[MCP] Ignoring close callback from old connection for ${serverName}`);
							return;
						}
						console.log(`[MCP] Connection closed for: ${serverName}`);
						this.updateConnectionState(serverName, { connected: false });
					};

					service.onOpen = async () => {
						if (this.connectionGenerations.get(serverName) !== currentGen) {
							console.log(`[MCP] Ignoring open callback from old connection for ${serverName}`);
							return;
						}
						console.log(`[MCP] Connection opened for: ${serverName}`);

						// Initialize MCP session
						try {
							await service.initialize({
								roots: { listChanged: true },
								sampling: {}
							});

							// Fetch initial tools
							const tools = await service.listTools();
							console.log(`[MCP] Got ${tools.length} tools for ${serverName}`);
							if (this.connectionGenerations.get(serverName) !== currentGen) {
								console.log(
									`[MCP] Connection generation changed during init for ${serverName}, ignoring result`
								);
								return;
							}
							this.updateConnectionState(serverName, {
								connected: true,
								tools,
								error: undefined
							});
						} catch (error) {
							console.error(`[MCP] Failed to initialize ${serverName}:`, error);
							console.error(`[MCP] Error details:`, {
								type: typeof error,
								name: error instanceof Error ? error.name : 'N/A',
								message: error instanceof Error ? error.message : String(error),
								stack: error instanceof Error ? error.stack : 'N/A',
								keys: error instanceof Error ? Object.keys(error) : Object.keys(error || {}),
								full: error
							});
							if (this.connectionGenerations.get(serverName) !== currentGen) {
								console.log(
									`[MCP] Connection generation changed during init error for ${serverName}, ignoring error`
								);
								return;
							}
							this.updateConnectionState(serverName, {
								error: error instanceof Error ? error.message : String(error),
								connected: false
							});
						}
					};

					// Connect
					await service.connect();
				} catch (error) {
					console.error(`[MCP] Failed to connect to ${serverName}:`, error);
					this.updateConnectionState(serverName, {
						error: error instanceof Error ? error.message : 'Connection failed',
						connected: false
					});
					this.services.delete(serverName);
					this.connections.delete(serverName);
				} finally {
					this.removeConnecting(serverName);
				}
			} finally {
				// Clear the connection promise when done (whether success or failure)
				this.connectionPromises.delete(serverName);
			}
		})();

		// Store the promise
		this.connectionPromises.set(serverName, connectionPromise);

		// Return the promise
		return connectionPromise;
	}

	/**
	 * Disconnect from an MCP server
	 */
	disconnect(serverName: string): void {
		// Clear connection promise to allow reconnect
		this.connectionPromises.delete(serverName);
		// Clear connecting state (reactive)
		this.removeConnecting(serverName);

		const service = this.services.get(serverName);
		if (service) {
			service.disconnect();
			// DO NOT remove from services map yet - the WebSocket close is async
			// The service will be removed when a new connection is created or explicitly cleaned up
		}
		// Create a new SvelteMap to trigger Svelte 5 reactivity
		const newConnections = new SvelteMap(this.connections);
		newConnections.delete(serverName);
		this.connections = newConnections;
		console.log(`[MCP] Disconnected from: ${serverName}`);
	}

	/**
	 * Disconnect from all MCP servers
	 */
	disconnectAll(): void {
		for (const serverName of this.services.keys()) {
			this.disconnect(serverName);
		}
	}

	/**
	 * Check if a server is connected
	 */
	isConnected(serverName: string): boolean {
		return this.connections.get(serverName)?.connected ?? false;
	}

	/**
	 * Check if currently connecting to a server
	 */
	isConnecting(serverName: string): boolean {
		return this.connecting.has(serverName);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Tool Calling
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Call a tool on an MCP server
	 */
	async callTool(
		serverName: string,
		toolName: string,
		args?: Record<string, unknown>
	): Promise<unknown> {
		const service = this.services.get(serverName);
		if (!service) {
			throw new Error(`MCP server not connected: ${serverName}`);
		}

		if (!service.isConnected()) {
			throw new Error(`MCP server not ready: ${serverName}`);
		}

		return await service.callTool(toolName, args);
	}

	/**
	 * Get tools for a specific server
	 */
	getTools(serverName: string): McpTool[] {
		return this.connections.get(serverName)?.tools ?? [];
	}

	/**
	 * Get a specific tool from a server
	 */
	getTool(serverName: string, toolName: string): McpTool | undefined {
		const tools = this.getTools(serverName);
		return tools.find((t) => t.name === toolName);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Utilities
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Update connection state for a server
	 *
	 * IMPORTANT: For Svelte 5 reactivity, we create a new Map instead of calling .set().
	 * This ensures derived states are recalculated when the connection state changes.
	 */
	private updateConnectionState(serverName: string, updates: Partial<McpConnectionState>): void {
		const current = this.connections.get(serverName);

		let newState: McpConnectionState;
		if (current) {
			newState = { ...current, ...updates };
		} else {
			newState = {
				name: serverName,
				connected: false,
				tools: [],
				error: undefined,
				...updates
			};
		}

		// Create a new SvelteMap to trigger Svelte 5 reactivity
		// Simply calling .set() on the existing Map doesn't trigger reactivity for derived states
		const newConnections = new SvelteMap(this.connections);
		newConnections.set(serverName, newState);
		this.connections = newConnections;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Provisional Conversation ID Management
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Get or create a provisional conversation ID
	 * This allows connecting to MCP servers before a real conversation exists
	 */
	getProvisionalConvId(): string {
		if (!this.provisionalConvId) {
			this.provisionalConvId = `provisional-${Date.now()}-${Math.random().toString(36).slice(2)}`;
		}
		return this.provisionalConvId;
	}

	/**
	 * Migrate provisional ID to real conversation ID
	 * Called when a real conversation is created
	 */
	migrateProvisionalConvId(realConvId: string): void {
		if (this.provisionalConvId) {
			// The connections are already established, just update the tracking
			console.log(
				`[MCP] Migrating provisional ID ${this.provisionalConvId} to real ID ${realConvId}`
			);
			this.realConvId = realConvId;
			this.provisionalConvId = null;
		}
	}

	/**
	 * Clear the provisional conversation ID
	 */
	clearProvisionalConvId(): void {
		this.provisionalConvId = null;
	}

	/**
	 * Check if currently using a provisional ID
	 */
	isProvisional(): boolean {
		return this.provisionalConvId !== null;
	}
}

// Export singleton instance
export const mcpStore = new McpStore();

// Export derived state getters
export const connectedServers = () => mcpStore.connectedServers;
export const mcpServerNames = () => mcpStore.serverNames;
export const toolsByServer = () => mcpStore.toolsByServer;
export const allMcpTools = () => mcpStore.allTools;

// Export connection state getters
export const isMcpConnected = (serverName: string) => mcpStore.isConnected(serverName);
export const isMcpConnecting = (serverName: string) => mcpStore.isConnecting(serverName);

// Export tools getter
export const getMcpTools = (serverName: string) => mcpStore.getTools(serverName);
export const getMcpTool = (serverName: string, toolName: string) =>
	mcpStore.getTool(serverName, toolName);

// Export provisional conversation ID methods
export const getProvisionalConvId = () => mcpStore.getProvisionalConvId();
export const migrateProvisionalConvId = (realConvId: string) =>
	mcpStore.migrateProvisionalConvId(realConvId);
export const clearProvisionalConvId = () => mcpStore.clearProvisionalConvId();
export const isProvisionalConvId = () => mcpStore.isProvisional();
