// MCP (Model Context Protocol) Types

// JSON-RPC 2.0 Request ID
export type McpRequestId = string | number;

// JSON-RPC 2.0 Request
export interface McpJsonRpcRequest {
	jsonrpc: '2.0';
	id: McpRequestId;
	method: string;
	params?: unknown;
}

// JSON-RPC 2.0 Response
export interface McpJsonRpcResponse {
	jsonrpc: '2.0';
	id: McpRequestId;
	result?: unknown;
	error?: {
		code: number;
		message: string;
		data?: unknown;
	};
}

// JSON-RPC 2.0 Notification (no id field)
export interface McpJsonRpcNotification {
	jsonrpc: '2.0';
	method: string;
	params?: unknown;
}

// MCP Tool types
export interface McpTool {
	name: string;
	description?: string;
	inputSchema: unknown; // JSON Schema
}

// MCP Tool call
export interface McpToolCall {
	name: string;
	arguments?: Record<string, unknown>;
}

// MCP Resource types
export interface McpResource {
	uri: string;
	name: string;
	description?: string;
	mimeType?: string;
}

// MCP Prompt types
export interface McpPrompt {
	name: string;
	description?: string;
	arguments?: McpPromptArgument[];
}

export interface McpPromptArgument {
	name: string;
	description?: string;
	required?: boolean;
}

// MCP Server info
export interface McpServerInfo {
	name: string;
	version?: string;
	protocolVersion: string;
	capabilities: unknown;
	serverConfig: McpServerConfig;
}

// MCP Server configuration (matches server-side)
export interface McpServerConfig {
	name: string;
	command: string;
	args: string[];
	env: Record<string, string>;
}

// MCP Connection state
export interface McpConnectionState {
	name: string;
	connected: boolean;
	tools: McpTool[];
	error?: string;
}

// MCP Methods (from MCP spec)
export const MCP_METHODS = {
	INITIALIZE: 'initialize',
	INITIALIZED: 'notifications/initialized',
	LIST_TOOLS: 'tools/list',
	CALL_TOOL: 'tools/call',
	LIST_RESOURCES: 'resources/list',
	READ_RESOURCE: 'resources/read',
	LIST_PROMPTS: 'prompts/list',
	GET_PROMPT: 'prompts/get',
	SET_LEVEL: 'logging/set_level',
	TOOLS_CHANGED: 'notifications/tools/list_changed',
	RESOURCES_CHANGED: 'notifications/resources/list_changed',
	PROMPTS_CHANGED: 'notifications/prompts/list_changed',
	CANCEL_REQUEST: 'requests/cancel',
	PING: 'ping'
} as const;
