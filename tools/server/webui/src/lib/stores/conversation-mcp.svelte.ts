/**
 * Conversation MCP Store - Per-conversation MCP state management
 *
 * Manages MCP server state per conversation including:
 * - Disabled servers list (persisted in conversation metadata)
 * - Per-conversation MCP connections (each conversation has isolated connections)
 *
 * **Design Decisions:**
 * - Store "disabled" list (not enabled) so new servers from config are auto-enabled
 * - Each conversation has its own MCP connections for stdio isolation
 * - When switching conversations, disconnect old connections and connect new ones
 *
 * **Architecture & Relationships:**
 * - **conversationMcpStore** (this class): Per-conversation MCP state
 * - **mcpStore**: Global MCP store for server discovery
 * - **conversationsStore**: For persisting disabled servers list
 */

import { mcpStore } from './mcp.svelte';
import type { McpTool } from '$lib/types/mcp';

interface ConversationMcpState {
	disabledServers: Set<string>;
}

class ConversationMcpStore {
	// Map: conversationId -> Set of disabled server names
	private states = new Map<string, ConversationMcpState>();
	// Map: conversationId -> Map of serverName -> tools
	private conversationTools = new Map<string, Map<string, McpTool[]>>();
	// Current active conversation ID
	private activeConversationId = $state<string | null>(null);

	// ─────────────────────────────────────────────────────────────────────────────
	// State Management
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Set the active conversation (called when switching conversations)
	 */
	setActiveConversation(convId: string | null): void {
		this.activeConversationId = convId;
	}

	/**
	 * Get or create state for a conversation
	 */
	private getState(convId: string): ConversationMcpState {
		if (!this.states.has(convId)) {
			this.states.set(convId, { disabledServers: new Set() });
		}
		return this.states.get(convId)!;
	}

	/**
	 * Get disabled servers list for a conversation (for persistence)
	 */
	getDisabledServersList(convId: string): string[] {
		return Array.from(this.getState(convId).disabledServers);
	}

	/**
	 * Set disabled servers list for a conversation (when loading from persistence)
	 */
	setDisabledServersList(convId: string, servers: string[]): void {
		const state = this.getState(convId);
		state.disabledServers = new Set(servers);
	}

	/**
	 * Clean up state for a conversation
	 */
	cleanupConversation(convId: string): void {
		this.states.delete(convId);
		this.conversationTools.delete(convId);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Server Enable/Disable
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Check if a server is enabled for a conversation
	 */
	isServerEnabled(convId: string, serverName: string): boolean {
		return !this.getState(convId).disabledServers.has(serverName);
	}

	/**
	 * Toggle a server's enabled state for a conversation
	 */
	toggleServer(convId: string, serverName: string): void {
		const state = this.getState(convId);
		if (state.disabledServers.has(serverName)) {
			state.disabledServers.delete(serverName);
		} else {
			state.disabledServers.add(serverName);
		}
	}

	/**
	 * Enable a server for a conversation
	 */
	enableServer(convId: string, serverName: string): void {
		this.getState(convId).disabledServers.delete(serverName);
	}

	/**
	 * Disable a server for a conversation
	 */
	disableServer(convId: string, serverName: string): void {
		this.getState(convId).disabledServers.add(serverName);
	}

	/**
	 * Get all enabled server names for a conversation
	 */
	getEnabledServers(convId: string): string[] {
		const allServers = mcpStore.serverNames;
		const disabled = this.getState(convId).disabledServers;
		return allServers.filter((s) => !disabled.has(s));
	}

	/**
	 * Get all disabled server names for a conversation
	 */
	getDisabledServers(convId: string): Set<string> {
		return new Set(this.getState(convId).disabledServers);
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Tools
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Cache tools for a conversation (fetched from MCP servers)
	 */
	setToolsForConversation(convId: string, serverName: string, tools: McpTool[]): void {
		if (!this.conversationTools.has(convId)) {
			this.conversationTools.set(convId, new Map());
		}
		this.conversationTools.get(convId)!.set(serverName, tools);
	}

	/**
	 * Get tools for a specific server in a conversation
	 */
	getToolsForServer(convId: string, serverName: string): McpTool[] {
		return this.conversationTools.get(convId)?.get(serverName) ?? mcpStore.getTools(serverName);
	}

	/**
	 * Get all tools for enabled servers in a conversation
	 * Returns array of { serverName, tool } with mcp__{server}__{tool} naming
	 * NOTE: Uses connected servers from mcpStore, not the enabled/disabled state
	 */
	async getAllToolsForConversation(
		_convId: string // Reserved for future per-conversation filtering
	): Promise<Array<{ serverName: string; tool: McpTool; qualifiedName: string }>> {
		// Get all connected servers (regardless of conversation-level enable/disable)
		const connectedServers = Array.from(mcpStore.connections.keys()).filter((s) =>
			mcpStore.isConnected(s)
		);
		const result: Array<{ serverName: string; tool: McpTool; qualifiedName: string }> = [];

		for (const serverName of connectedServers) {
			const tools = mcpStore.getTools(serverName);
			for (const tool of tools) {
				result.push({
					serverName,
					tool,
					qualifiedName: `mcp__${serverName}__${tool.name}`
				});
			}
		}

		return result;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Tool Execution
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Execute a tool call
	 * Parses qualified name (mcp__server__tool) and routes to appropriate MCP server
	 */
	async executeToolCall(qualifiedName: string, args: Record<string, unknown>): Promise<unknown> {
		// Parse mcp__serverName__toolName
		if (!qualifiedName.startsWith('mcp__')) {
			throw new Error(`Invalid MCP tool name format: ${qualifiedName}`);
		}

		const parts = qualifiedName.split('__');
		if (parts.length < 3) {
			throw new Error(`Invalid MCP tool name format: ${qualifiedName}`);
		}

		// parts[0] = 'mcp', parts[1] = serverName, parts[2...] = toolName (may contain __)
		const serverName = parts[1];
		const toolName = parts.slice(2).join('__');

		return await mcpStore.callTool(serverName, toolName, args);
	}

	/**
	 * Parse tool call name to extract server and tool name
	 */
	parseToolName(qualifiedName: string): { serverName: string; toolName: string } | null {
		if (!qualifiedName.startsWith('mcp__')) {
			return null;
		}

		const parts = qualifiedName.split('__');
		if (parts.length < 3) {
			return null;
		}

		return {
			serverName: parts[1],
			toolName: parts.slice(2).join('__')
		};
	}
}

export const conversationMcpStore = new ConversationMcpStore();

// Export convenience functions
export const getEnabledServers = (convId: string) => conversationMcpStore.getEnabledServers(convId);
export const isServerEnabled = (convId: string, serverName: string) =>
	conversationMcpStore.isServerEnabled(convId, serverName);
export const toggleServer = (convId: string, serverName: string) =>
	conversationMcpStore.toggleServer(convId, serverName);
export const getAllToolsForConversation = (convId: string) =>
	conversationMcpStore.getAllToolsForConversation(convId);
export const executeMcpToolCall = (qualifiedName: string, args: Record<string, unknown>) =>
	conversationMcpStore.executeToolCall(qualifiedName, args);
