<script lang="ts">
	import {
		ChevronDown,
		ChevronRight,
		Wrench,
		Loader2,
		CheckCircle2,
		XCircle
	} from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		toolCall: ApiChatCompletionToolCall;
		result?: string | null;
		status: 'streaming' | 'calling' | 'complete' | 'error';
		durationMs?: number | null;
		class?: string;
	}

	let {
		toolCall,
		result = null,
		status,
		durationMs = null,
		class: className = ''
	}: Props = $props();

	// Format duration for display
	function formatDuration(ms: number): string {
		if (ms < 1000) {
			return `${ms}ms`;
		} else if (ms < 60000) {
			return `${(ms / 1000).toFixed(1)}s`;
		} else {
			const mins = Math.floor(ms / 60000);
			const secs = ((ms % 60000) / 1000).toFixed(0);
			return `${mins}m ${secs}s`;
		}
	}

	let isArgsExpanded = $state(false);
	let isResultExpanded = $state(false);

	// Parse MCP tool names: mcp__serverName__toolName -> serverName:toolName
	let displayToolName = $derived(() => {
		const name = toolCall.function?.name || 'Unknown tool';
		if (name.startsWith('mcp__')) {
			const parts = name.split('__');
			if (parts.length >= 3) {
				const serverName = parts[1];
				const actualToolName = parts.slice(2).join('__');
				return `${serverName}:${actualToolName}`;
			}
		}
		return name;
	});

	// Parse and format arguments
	let formattedArgs = $derived(() => {
		const rawArgs = toolCall.function?.arguments?.trim();
		if (!rawArgs) return null;
		try {
			return JSON.stringify(JSON.parse(rawArgs), null, 2);
		} catch {
			return rawArgs;
		}
	});

	// Parse and format result
	let formattedResult = $derived(() => {
		if (!result) return null;
		try {
			return JSON.stringify(JSON.parse(result), null, 2);
		} catch {
			return result;
		}
	});

	function handleCopyAll() {
		const payload = {
			toolCall: {
				id: toolCall.id,
				name: toolCall.function?.name,
				arguments: formattedArgs() ? JSON.parse(formattedArgs()!) : null
			},
			result: formattedResult()
				? (() => {
						try {
							return JSON.parse(formattedResult()!);
						} catch {
							return formattedResult();
						}
					})()
				: null
		};
		void copyToClipboard(JSON.stringify(payload, null, 2), 'Tool call copied to clipboard');
	}
</script>

<div class={cn('w-full', className)}>
	<div class="overflow-hidden rounded-lg border bg-muted/30">
		<!-- Header -->
		<div class="flex items-center justify-between bg-muted/50 px-3 py-2">
			<div class="flex items-center gap-2">
				<Wrench class="h-4 w-4 text-muted-foreground" />
				<span class="text-sm font-medium">{displayToolName()}</span>
				{#if status === 'streaming' || status === 'calling'}
					<Loader2 class="h-3.5 w-3.5 animate-spin text-muted-foreground" />
				{:else if status === 'complete'}
					<CheckCircle2 class="h-3.5 w-3.5 text-muted-foreground" />
				{:else if status === 'error'}
					<XCircle class="h-3.5 w-3.5 text-destructive" />
				{/if}
			</div>
			<button
				type="button"
				onclick={handleCopyAll}
				class="rounded p-1.5 transition hover:bg-muted-foreground/10"
				title="Copy tool call"
				aria-label="Copy tool call"
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					width="14"
					height="14"
					viewBox="0 0 24 24"
					fill="none"
					stroke="currentColor"
					stroke-width="2"
					stroke-linecap="round"
					stroke-linejoin="round"
					class="text-muted-foreground"
					><rect width="14" height="14" x="8" y="8" rx="2" ry="2" /><path
						d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"
					/></svg
				>
			</button>
		</div>

		<!-- Arguments Section (collapsible) -->
		{#if formattedArgs()}
			<div class="border-t">
				<button
					type="button"
					onclick={() => (isArgsExpanded = !isArgsExpanded)}
					class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs text-muted-foreground transition hover:bg-muted-foreground/5"
				>
					{#if isArgsExpanded}
						<ChevronDown class="h-3 w-3" />
					{:else}
						<ChevronRight class="h-3 w-3" />
					{/if}
					<span>Arguments</span>
				</button>
				{#if isArgsExpanded}
					<div class="px-3 pb-2">
						<pre
							class="max-h-48 overflow-x-auto overflow-y-auto rounded bg-background/50 p-2 text-xs break-words whitespace-pre-wrap">{formattedArgs()}</pre>
					</div>
				{/if}
			</div>
		{/if}

		<!-- Status / Result Section (only show when calling or have results) -->
		{#if status !== 'streaming'}
			<div class="border-t">
				{#if status === 'calling'}
					<div class="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
						<Loader2 class="h-3 w-3 animate-spin" />
						<span>Calling tool...</span>
					</div>
				{:else if status === 'complete' && formattedResult()}
					<button
						type="button"
						onclick={() => (isResultExpanded = !isResultExpanded)}
						class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs transition hover:bg-muted-foreground/5"
					>
						{#if isResultExpanded}
							<ChevronDown class="h-3 w-3 text-muted-foreground" />
						{:else}
							<ChevronRight class="h-3 w-3 text-muted-foreground" />
						{/if}
						<span>Result</span>
						{#if durationMs !== null && durationMs > 0}
							<span class="text-muted-foreground">({formatDuration(durationMs)})</span>
						{/if}
					</button>
					{#if isResultExpanded}
						<div class="px-3 pb-2">
							<pre
								class="max-h-64 overflow-x-auto overflow-y-auto rounded bg-background/50 p-2 text-xs break-words whitespace-pre-wrap">{formattedResult()}</pre>
						</div>
					{/if}
				{:else if status === 'error'}
					<div class="px-3 py-2">
						<div class="text-xs text-destructive">
							{result || 'Tool call failed'}
						</div>
					</div>
				{:else if status === 'complete' && !formattedResult()}
					<div class="flex items-center gap-2 px-3 py-2 text-xs text-muted-foreground">
						<CheckCircle2 class="h-3 w-3" />
						<span>Completed (no result)</span>
						{#if durationMs !== null && durationMs > 0}
							<span>({formatDuration(durationMs)})</span>
						{/if}
					</div>
				{/if}
			</div>
		{/if}
	</div>
</div>
