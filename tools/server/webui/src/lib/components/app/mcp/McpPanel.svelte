<script lang="ts">
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import type { McpConnectionState } from '$lib/types/mcp';
	import { Button } from '$lib/components/ui/button';
	import * as Dialog from '$lib/components/ui/dialog';
	import { Card, CardContent } from '$lib/components/ui/card';
	import { Badge } from '$lib/components/ui/badge';
	import { Input } from '$lib/components/ui/input';

	let showAddDialog = $state(false);
	let newServerName = $state('');
	let isConnecting = $state(false);
	let connectError = $state<string | null>(null);

	async function handleAddServer() {
		if (!newServerName.trim()) return;

		isConnecting = true;
		connectError = null;

		try {
			await mcpStore.connect(newServerName.trim());
			newServerName = '';
			showAddDialog = false;
		} catch (error) {
			connectError = error instanceof Error ? error.message : 'Failed to connect';
		} finally {
			isConnecting = false;
		}
	}

	function handleRemoveServer(serverName: string) {
		mcpStore.disconnect(serverName);
	}

	function getConnectionStatus(state: McpConnectionState): {
		label: string;
		variant: 'default' | 'secondary' | 'destructive' | 'outline';
	} {
		if (state.connected) {
			return { label: 'Connected', variant: 'default' };
		}
		if (state.error) {
			return { label: 'Error', variant: 'destructive' };
		}
		return { label: 'Disconnected', variant: 'secondary' };
	}
</script>

<div class="mcp-panel flex flex-col gap-4">
	<!-- Header -->
	<div class="flex items-center justify-between">
		<div>
			<h2 class="text-lg font-semibold">MCP Servers</h2>
			<p class="text-sm text-muted-foreground">Manage Model Context Protocol server connections</p>
		</div>
		<Button variant="outline" size="sm" onclick={() => (showAddDialog = true)}>
			<svg
				xmlns="http://www.w3.org/2000/svg"
				width="16"
				height="16"
				viewBox="0 0 24 24"
				fill="none"
				stroke="currentColor"
				stroke-width="2"
				stroke-linecap="round"
				stroke-linejoin="round"
			>
				<path d="M5 12h14" />
				<path d="M12 5v14" />
			</svg>
			Add Server
		</Button>
	</div>

	<!-- Servers List -->
	{#if mcpStore.connectedServers.length === 0 && mcpStore.connections.size === 0}
		<CardContent class="py-8 text-center">
			<p class="text-muted-foreground">No MCP servers configured</p>
			<p class="mt-1 text-sm text-muted-foreground">
				Add a server to enable tool calling capabilities
			</p>
		</CardContent>
	{:else}
		<div class="flex flex-col gap-2">
			{#each Array.from(mcpStore.connections.values()) as state (state.name)}
				<Card class="p-4">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-3">
							<div>
								<h3 class="font-medium">{state.name}</h3>
								{#if state.connected && state.tools.length > 0}
									<p class="text-sm text-muted-foreground">
										{state.tools.length} tool{state.tools.length === 1 ? '' : 's'} available
									</p>
								{/if}
							</div>
						</div>
						<div class="flex items-center gap-2">
							<Badge variant={getConnectionStatus(state).variant}>
								{getConnectionStatus(state).label}
							</Badge>
							<Button variant="ghost" size="sm" onclick={() => handleRemoveServer(state.name)}>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									width="16"
									height="16"
									viewBox="0 0 24 24"
									fill="none"
									stroke="currentColor"
									stroke-width="2"
									stroke-linecap="round"
									stroke-linejoin="round"
								>
									<path d="M18 6 6 18" />
									<path d="m6 6 12 12" />
								</svg>
							</Button>
						</div>
					</div>
					{#if state.error}
						<p class="mt-2 text-sm text-destructive">{state.error}</p>
					{/if}
				</Card>
			{/each}
		</div>
	{/if}
</div>

<!-- Add Server Dialog -->
<Dialog.Root open={showAddDialog} onOpenChange={(open) => (showAddDialog = open)}>
	<Dialog.Content>
		<Dialog.Header>
			<Dialog.Title>Add MCP Server</Dialog.Title>
			<Dialog.Description>
				Enter the name of the MCP server to connect to. The server must be configured in the
				llama.cpp MCP configuration file.
			</Dialog.Description>
		</Dialog.Header>

		<div class="flex flex-col gap-4 py-4">
			<div class="flex flex-col gap-2">
				<label for="server-name" class="text-sm font-medium">Server Name</label>
				<Input
					id="server-name"
					bind:value={newServerName}
					placeholder="e.g., filesystem, brave-search"
					disabled={isConnecting}
					onkeydown={(e) => e.key === 'Enter' && handleAddServer()}
				/>
			</div>

			{#if connectError}
				<p class="text-sm text-destructive">{connectError}</p>
			{/if}

			<div class="text-sm text-muted-foreground">
				<p>
					Configure MCP servers in <code class="rounded bg-muted px-1 py-0.5"
						>~/.llama.cpp/mcp.json</code
					>
					or set the <code class="rounded bg-muted px-1 py-0.5">LLAMA_MCP_CONFIG</code> environment variable.
				</p>
			</div>
		</div>

		<Dialog.Footer>
			<Button variant="outline" onclick={() => (showAddDialog = false)} disabled={isConnecting}>
				Cancel
			</Button>
			<Button onclick={handleAddServer} disabled={isConnecting || !newServerName.trim()}>
				{#if isConnecting}
					Connecting...
				{:else}
					Connect
				{/if}
			</Button>
		</Dialog.Footer>
	</Dialog.Content>
</Dialog.Root>
