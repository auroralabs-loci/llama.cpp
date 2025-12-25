<script lang="ts">
	import { onMount, tick } from 'svelte';
	import {
		ChevronDown,
		ChevronRight,
		Loader2,
		Plug,
		Power,
		RefreshCw,
		FileJson,
		Wrench
	} from '@lucide/svelte';
	import * as Popover from '$lib/components/ui/popover';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { SearchInput } from '$lib/components/app';
	import { SvelteSet, SvelteMap } from 'svelte/reactivity';

	// LocalStorage key for remembering disconnection preferences
	const MCP_PREFS_KEY = 'mcp-connection-prefs';

	interface McpConnectionPrefs {
		// Servers that were explicitly disconnected by user
		explicitlyDisconnected: string[];
		// If true, user disconnected all - new convos start disconnected
		allDisconnectedByUser: boolean;
	}

	interface Props {
		class?: string;
		disabled?: boolean;
	}

	let { class: className = '', disabled = false }: Props = $props();

	// UI State
	let isOpen = $state(false);
	let searchTerm = $state('');
	let searchInputRef = $state<HTMLInputElement | null>(null);
	let selectedServerName = $state<string | null>(null);
	let highlightedIndex = $state<number>(-1);

	// Server data
	let availableServers = $state<string[]>([]);
	let loadingServers = $state(true);
	let fetchingTools = new SvelteSet<string>();

	// Connection preferences (persisted)
	let connectionPrefs = $state<McpConnectionPrefs>({
		explicitlyDisconnected: [],
		allDisconnectedByUser: false
	});

	// Load preferences from localStorage
	function loadConnectionPrefs(): McpConnectionPrefs {
		try {
			const stored = localStorage.getItem(MCP_PREFS_KEY);
			if (stored) {
				return JSON.parse(stored);
			}
		} catch (e) {
			console.warn('[MCP] Failed to load connection prefs:', e);
		}
		return { explicitlyDisconnected: [], allDisconnectedByUser: false };
	}

	// Save preferences to localStorage
	function saveConnectionPrefs(prefs: McpConnectionPrefs) {
		try {
			localStorage.setItem(MCP_PREFS_KEY, JSON.stringify(prefs));
		} catch (e) {
			console.warn('[MCP] Failed to save connection prefs:', e);
		}
	}

	// Mark server as explicitly disconnected
	function markExplicitlyDisconnected(serverName: string) {
		if (!connectionPrefs.explicitlyDisconnected.includes(serverName)) {
			connectionPrefs = {
				...connectionPrefs,
				explicitlyDisconnected: [...connectionPrefs.explicitlyDisconnected, serverName]
			};
			saveConnectionPrefs(connectionPrefs);
		}
	}

	// Clear explicit disconnection for a server (user connected it)
	function clearExplicitDisconnection(serverName: string) {
		// If user was in "all disconnected" mode, convert to explicit list
		// so other servers stay disconnected
		if (connectionPrefs.allDisconnectedByUser) {
			// Add all OTHER servers to explicitly disconnected
			const otherServers = availableServers.filter((s) => s !== serverName);
			connectionPrefs = {
				explicitlyDisconnected: otherServers,
				allDisconnectedByUser: false
			};
			saveConnectionPrefs(connectionPrefs);
		} else if (connectionPrefs.explicitlyDisconnected.includes(serverName)) {
			// Just remove this server from the explicit list
			connectionPrefs = {
				...connectionPrefs,
				explicitlyDisconnected: connectionPrefs.explicitlyDisconnected.filter(
					(s) => s !== serverName
				)
			};
			saveConnectionPrefs(connectionPrefs);
		}
	}

	// Check if server was explicitly disconnected
	function wasExplicitlyDisconnected(serverName: string): boolean {
		return (
			connectionPrefs.allDisconnectedByUser ||
			connectionPrefs.explicitlyDisconnected.includes(serverName)
		);
	}

	// Filtered servers for search
	let filteredServers = $derived(
		(() => {
			const term = searchTerm.trim().toLowerCase();
			if (!term) return availableServers;
			return availableServers.filter((s) => s.toLowerCase().includes(term));
		})()
	);

	// Reactive server status map - tracks connection states reactively
	// This MUST be a $derived.by to properly track changes to mcpStore
	let serverStatus = $derived.by(() => {
		const statusMap = new SvelteMap<
			string,
			{
				connected: boolean;
				connecting: boolean;
				error?: string;
			}
		>();
		for (const serverName of availableServers) {
			const connection = mcpStore.connections.get(serverName);
			statusMap.set(serverName, {
				connected: connection?.connected ?? false,
				connecting: mcpStore.isConnecting(serverName),
				error: connection?.error
			});
		}
		return statusMap;
	});

	// Count connected servers - MUST be computed from serverStatus to be reactive
	let connectedCount = $derived.by(() => {
		return Array.from(serverStatus.values()).filter((s) => s.connected).length;
	});

	// Get server status (now uses the reactive map)
	function getServerStatus(serverName: string): {
		connected: boolean;
		connecting: boolean;
		error?: string;
	} {
		return serverStatus.get(serverName) ?? { connected: false, connecting: false };
	}

	// Get status color for server (used for dot indicator)
	function getStatusDisplay(status: { connected: boolean; connecting: boolean; error?: string }) {
		if (status.connecting) {
			return {
				dotColor: 'bg-orange-500',
				label: 'Connecting...'
			};
		}
		if (status.connected) {
			return {
				dotColor: 'bg-green-500',
				label: 'Connected'
			};
		}
		if (status.error) {
			return {
				dotColor: 'bg-red-500',
				label: 'Error'
			};
		}
		return {
			dotColor: 'bg-muted-foreground/50',
			label: 'Disconnected'
		};
	}

	// Fetch available servers
	async function fetchAvailableServers(autoConnect = false) {
		loadingServers = true;
		try {
			const response = await fetch('/mcp/servers');
			if (!response.ok) throw new Error('Failed to fetch MCP servers');
			const data = await response.json();
			availableServers = data.servers.map((s: { name: string }) => s.name);

			// Auto-connect servers if requested and not explicitly disconnected
			if (autoConnect && availableServers.length > 0) {
				// If user previously disconnected all, don't auto-connect
				if (connectionPrefs.allDisconnectedByUser) {
					console.log('[MCP] Skipping auto-connect: user previously disconnected all');
					return;
				}

				// Connect servers that weren't explicitly disconnected
				for (const serverName of availableServers) {
					if (!wasExplicitlyDisconnected(serverName)) {
						console.log(`[MCP] Auto-connecting to: ${serverName}`);
						mcpStore.connect(serverName).catch((err) => {
							console.error(`[MCP] Auto-connect failed for ${serverName}:`, err);
						});
					} else {
						console.log(`[MCP] Skipping auto-connect for ${serverName}: explicitly disconnected`);
					}
				}
			}
		} catch (error) {
			console.error('Failed to fetch MCP servers:', error);
			availableServers = [];
		} finally {
			loadingServers = false;
		}
	}

	onMount(() => {
		// Load saved connection preferences
		connectionPrefs = loadConnectionPrefs();
		// Fetch servers and auto-connect on initial load
		fetchAvailableServers(true);
	});

	// Handle popover open/close
	async function handleOpenChange(open: boolean) {
		if (loadingServers) return;

		isOpen = open;
		if (open) {
			searchTerm = '';
			selectedServerName = null;
			await fetchAvailableServers();
			tick().then(() => {
				requestAnimationFrame(() => searchInputRef?.focus());
			});
		} else {
			selectedServerName = null;
		}
	}

	// Toggle connection
	async function toggleConnection(serverName: string) {
		const status = getServerStatus(serverName);
		if (status.connecting) return;

		if (status.connected) {
			// User explicitly disconnecting - remember this
			markExplicitlyDisconnected(serverName);
			mcpStore.disconnect(serverName);
		} else {
			// User explicitly connecting - clear disconnection preference
			clearExplicitDisconnection(serverName);
			await mcpStore.connect(serverName);
		}
	}

	// Check if any servers are currently connecting (for shimmer effect)
	let anyConnecting = $derived(() => {
		return availableServers.some((s) => getServerStatus(s).connecting);
	});

	// Force reconnect
	async function forceReconnect(serverName: string) {
		try {
			mcpStore.disconnect(serverName);
			// Wait for WebSocket to fully close before reconnecting
			// mcpStore.connect() has an internal 500ms wait, so we add buffer
			await new Promise((r) => setTimeout(r, 600));
			await mcpStore.connect(serverName);
		} catch (error) {
			console.error(`[MCP] Reconnect failed for ${serverName}:`, error);
		}
	}

	// Get tools for a server
	function getTools(serverName: string) {
		return mcpStore.getTools(serverName);
	}

	// Fetch tools for a server
	async function fetchServerTools(serverName: string) {
		if (!mcpStore.isConnected(serverName)) return;
		fetchingTools.add(serverName);
		try {
			// Tools are automatically fetched on connect, but we can refresh
			const service = mcpStore.connections.get(serverName);
			if (service?.connected) {
				// Tools should already be loaded from the initial connection
			}
		} finally {
			fetchingTools.delete(serverName);
		}
	}

	// Show server details
	function showServerDetails(serverName: string) {
		selectedServerName = serverName;
		if (mcpStore.isConnected(serverName)) {
			fetchServerTools(serverName);
		}
	}

	// Go back to server list
	function backToServerList() {
		selectedServerName = null;
	}

	// Handle keyboard navigation in search
	function handleSearchKeyDown(event: KeyboardEvent) {
		if (event.isComposing) return;

		if (event.key === 'ArrowDown') {
			event.preventDefault();
			if (filteredServers.length === 0) return;
			highlightedIndex = highlightedIndex < filteredServers.length - 1 ? highlightedIndex + 1 : 0;
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			if (filteredServers.length === 0) return;
			highlightedIndex = highlightedIndex > 0 ? highlightedIndex - 1 : filteredServers.length - 1;
		} else if (event.key === 'Enter' && highlightedIndex >= 0) {
			event.preventDefault();
			showServerDetails(filteredServers[highlightedIndex]);
		}
	}
</script>

<div class={cn('relative inline-flex flex-col items-end gap-1', className)}>
	<Popover.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
		<Popover.Trigger
			class={cn(
				`inline-flex cursor-pointer items-center gap-1.5 rounded-sm px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
				isOpen ? 'text-foreground' : 'text-muted-foreground',
				loadingServers || anyConnecting()
					? 'animate-shimmer bg-gradient-to-r from-muted-foreground/10 via-muted-foreground/20 to-muted-foreground/10 bg-[length:200%_100%]'
					: 'bg-muted-foreground/10'
			)}
			{disabled}
		>
			{#if loadingServers || anyConnecting()}
				<Loader2 class="h-3.5 w-3.5 animate-spin" />
			{:else}
				<Plug class="h-3.5 w-3.5" />
			{/if}
			<span class="truncate font-medium">
				{availableServers.length > 0 ? `${connectedCount}/${availableServers.length} MCP` : 'MCP'}
			</span>
			<ChevronDown class="h-3 w-3.5" />
		</Popover.Trigger>

		<Popover.Content
			class="group/popover-content w-96 max-w-[calc(100vw-2rem)] p-0"
			side="top"
			align="end"
			sideOffset={8}
			collisionPadding={16}
		>
			<div class="flex max-h-[50dvh] flex-col overflow-hidden">
				{#if selectedServerName}
					<!-- Server Details View -->
					{@const status = getServerStatus(selectedServerName)}
					{@const statusDisplay = getStatusDisplay(status)}
					{@const tools = getTools(selectedServerName)}
					{@const isFetching = fetchingTools.has(selectedServerName)}

					<div class="flex items-center gap-2 border-b p-4">
						<button
							type="button"
							onclick={backToServerList}
							class="shrink-0 rounded-sm p-1 transition hover:bg-muted"
						>
							<ChevronDown class="h-4 w-4 rotate-90" />
						</button>
						<div class="flex min-w-0 flex-1 items-center gap-2">
							{#if status.connecting}
								<Loader2 class="h-4 w-4 shrink-0 animate-spin text-orange-500" />
							{:else}
								<span class="h-2 w-2 shrink-0 rounded-full {statusDisplay.dotColor}"></span>
							{/if}
							<span class="truncate font-medium">{selectedServerName}</span>
						</div>
						<button
							onclick={() => selectedServerName && forceReconnect(selectedServerName)}
							class="shrink-0 rounded p-1 transition hover:bg-muted disabled:opacity-50"
							disabled={status.connecting || !selectedServerName}
							title="Reconnect"
						>
							<RefreshCw class="h-4 w-4 {status.connecting ? 'animate-spin' : ''}" />
						</button>
					</div>

					<div class="flex-1 overflow-y-auto p-4">
						{#if isFetching}
							<div class="flex items-center justify-center py-8 text-muted-foreground">
								<Loader2 class="mr-2 h-5 w-5 animate-spin" />
								Loading tools...
							</div>
						{:else if !status.connected}
							<div class="py-8 text-center text-muted-foreground">
								<p class="mb-4">Server is not connected</p>
								<button
									onclick={() => selectedServerName && toggleConnection(selectedServerName)}
									class="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
									disabled={!selectedServerName}
								>
									<Plug class="h-4 w-4" />
									Connect to Server
								</button>
							</div>
						{:else if tools.length === 0}
							<div class="py-8 text-center text-muted-foreground">
								<p>No tools available from this server</p>
							</div>
						{:else}
							<div class="space-y-3">
								<h4 class="text-sm font-medium text-muted-foreground">
									{tools.length} Tool{tools.length !== 1 ? 's' : ''} Available
								</h4>
								{#each tools as tool (tool.name)}
									<div class="group rounded-md border p-3 transition hover:bg-muted/50">
										<div class="flex items-start gap-2">
											<Wrench class="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
											<div class="min-w-0 flex-1">
												<p class="truncate text-sm font-medium">{tool.name}</p>
												{#if tool.description}
													<p class="mt-1 line-clamp-2 text-xs text-muted-foreground">
														{tool.description}
													</p>
												{/if}
												{#if tool.inputSchema}
													<div class="mt-2">
														<button
															type="button"
															class="flex items-center gap-1 text-xs text-muted-foreground transition hover:text-foreground"
															title={JSON.stringify(tool.inputSchema, null, 2)}
														>
															<FileJson class="h-3 w-3" />
															View input schema
														</button>
													</div>
												{/if}
											</div>
										</div>
									</div>
								{/each}
							</div>
						{/if}
					</div>
				{:else}
					<!-- Server List View -->
					<div
						class="order-1 shrink-0 border-b p-4 group-data-[side=top]/popover-content:order-2 group-data-[side=top]/popover-content:border-t group-data-[side=top]/popover-content:border-b-0"
					>
						<SearchInput
							id="mcp-search"
							placeholder="Search servers..."
							bind:value={searchTerm}
							bind:ref={searchInputRef}
							onClose={() => handleOpenChange(false)}
							onKeyDown={handleSearchKeyDown}
						/>
					</div>

					<div
						class="order-2 min-h-0 flex-1 overflow-y-auto group-data-[side=top]/popover-content:order-1"
					>
						{#if loadingServers}
							<div class="flex items-center justify-center py-8 text-muted-foreground">
								<Loader2 class="mr-2 h-5 w-5 animate-spin" />
								Loading servers...
							</div>
						{:else if availableServers.length === 0}
							<div class="p-4 text-center text-sm text-muted-foreground">
								<p class="mb-2">No MCP servers configured</p>
								<p class="text-xs">Add servers to ~/.llama.cpp/mcp.json</p>
							</div>
						{:else if filteredServers.length === 0}
							<div class="p-4 text-center text-sm text-muted-foreground">
								No servers found matching "{searchTerm}"
							</div>
						{:else}
							{#each filteredServers as serverName, index (serverName)}
								{@const status = getServerStatus(serverName)}
								{@const isHighlighted = index === highlightedIndex}

								<div
									class={cn(
										'group flex w-full cursor-pointer items-center gap-2 px-4 py-2 text-left text-sm transition focus:outline-none',
										isHighlighted
											? 'bg-accent text-accent-foreground'
											: 'hover:bg-accent hover:text-accent-foreground',
										status.connected ? 'text-popover-foreground' : 'text-muted-foreground'
									)}
									onmouseenter={() => (highlightedIndex = index)}
									role="option"
									aria-selected={isHighlighted}
									tabindex="0"
									onclick={() => showServerDetails(serverName)}
									onkeydown={(e) => {
										if (e.key === 'Enter' || e.key === ' ') {
											e.preventDefault();
											showServerDetails(serverName);
										}
									}}
								>
									<span class="min-w-0 flex-1 truncate">{serverName}</span>

									<!-- Status dot with hover action button (like model picker) -->
									{#if status.connecting}
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Loader2 class="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
											</Tooltip.Trigger>
											<Tooltip.Content class="z-[9999]">
												<p>Connecting...</p>
											</Tooltip.Content>
										</Tooltip.Root>
									{:else if status.connected}
										<!-- Connected: show green dot, Power icon on hover to disconnect -->
										<Tooltip.Root>
											<Tooltip.Trigger>
												<button
													type="button"
													class="relative ml-2 flex h-4 w-4 shrink-0 items-center justify-center"
													onclick={(e) => {
														e.stopPropagation();
														toggleConnection(serverName);
													}}
													aria-label="Disconnect"
												>
													<span
														class="mr-2 h-2 w-2 rounded-full bg-green-500 transition-opacity group-hover:opacity-0"
													></span>
													<Power
														class="absolute mr-2 h-4 w-4 text-red-500 opacity-0 transition-opacity group-hover:opacity-100 hover:text-red-600"
													/>
												</button>
											</Tooltip.Trigger>
											<Tooltip.Content class="z-[9999]">
												<p>Disconnect</p>
											</Tooltip.Content>
										</Tooltip.Root>
									{:else}
										<!-- Disconnected: show gray dot, Plug icon on hover to connect -->
										<Tooltip.Root>
											<Tooltip.Trigger>
												<button
													type="button"
													class="relative ml-2 flex h-4 w-4 shrink-0 items-center justify-center"
													onclick={(e) => {
														e.stopPropagation();
														toggleConnection(serverName);
													}}
													aria-label="Connect"
												>
													<span
														class="mr-2 h-2 w-2 rounded-full bg-muted-foreground/50 transition-opacity group-hover:opacity-0"
													></span>
													<Plug
														class="absolute mr-2 h-4 w-4 text-green-500 opacity-0 transition-opacity group-hover:opacity-100 hover:text-green-600"
													/>
												</button>
											</Tooltip.Trigger>
											<Tooltip.Content class="z-[9999]">
												<p>Connect</p>
											</Tooltip.Content>
										</Tooltip.Root>
									{/if}

									<!-- Arrow to view server details/tools -->
									<ChevronRight
										class="h-3.5 w-3.5 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
									/>
								</div>
							{/each}
						{/if}
					</div>
				{/if}
			</div>
		</Popover.Content>
	</Popover.Root>
</div>

<style>
	@keyframes shimmer {
		0% {
			background-position: 200% 0;
		}
		100% {
			background-position: -200% 0;
		}
	}

	:global(.animate-shimmer) {
		animation: shimmer 2s ease-in-out infinite;
	}
</style>
