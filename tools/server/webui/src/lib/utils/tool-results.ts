/**
 * Utility functions for handling MCP tool result messages
 */

export interface ToolResultInfo {
	toolName: string;
	result: string;
}

/**
 * Check if a message content is a tool result message
 */
export function isToolResultContent(content: string): boolean {
	return content.trim().startsWith('[Tool Result: ');
}

/**
 * Check if a database message is a tool result message
 */
export function isToolResultMessage(msg: { role: string; content: string }): boolean {
	return msg.role === 'user' && isToolResultContent(msg.content);
}

/**
 * Parse tool result from message content
 * Handles both \n and \r\n line endings, and empty results
 */
export function parseToolResult(content: string): ToolResultInfo | null {
	const trimmed = content.trim();

	// Handle both \n and \r\n line endings
	const match = trimmed.match(/^\[Tool Result: ([^\]]+)\][\r\n]+([\s\S]*)$/);
	if (match) {
		return { toolName: match[1], result: match[2] };
	}

	// Also try without newline (in case result is empty)
	const matchNoNewline = trimmed.match(/^\[Tool Result: ([^\]]+)\]$/);
	if (matchNoNewline) {
		return { toolName: matchNoNewline[1], result: '' };
	}

	return null;
}

/**
 * Format an MCP qualified tool name for display
 * Converts "mcp__serverName__toolName" to "serverName:toolName"
 */
export function formatMcpToolName(qualifiedName: string): string {
	if (!qualifiedName.startsWith('mcp__')) {
		return qualifiedName;
	}
	const parts = qualifiedName.split('__');
	if (parts.length < 3) {
		return qualifiedName;
	}
	// Handle tool names that may contain __ themselves
	return `${parts[1]}:${parts.slice(2).join('__')}`;
}

/**
 * Create a tool result message content string
 */
export function createToolResultContent(toolName: string, result: string): string {
	return `[Tool Result: ${toolName}]\n${result}`;
}
