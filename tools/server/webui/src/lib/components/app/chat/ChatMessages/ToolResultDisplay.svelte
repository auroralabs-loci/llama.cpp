<script lang="ts">
	import { ChevronDown, ChevronRight, FileJson } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { copyToClipboard } from '$lib/utils';

	interface Props {
		toolName: string;
		result: string;
		class?: string;
	}

	let { toolName, result, class: className = '' }: Props = $props();

	let isExpanded = $state(false);

	// Parse MCP tool names: mcp__serverName__toolName -> serverName:toolName
	let displayToolName = $derived(() => {
		if (toolName.startsWith('mcp__')) {
			const parts = toolName.split('__');
			if (parts.length >= 3) {
				const serverName = parts[1];
				const actualToolName = parts.slice(2).join('__');
				return `${serverName}:${actualToolName}`;
			}
		}
		return toolName;
	});

	// Try to parse the result as JSON for pretty printing
	let parsedResult = $derived(() => {
		try {
			return JSON.parse(result);
		} catch {
			return null;
		}
	});

	let formattedResult = $derived(() => {
		const parsed = parsedResult();
		if (parsed) {
			return JSON.stringify(parsed, null, 2);
		}
		return result;
	});

	function handleCopy() {
		void copyToClipboard(formattedResult(), 'Tool result copied to clipboard');
	}
</script>

<div class={cn('w-full max-w-[80%]', className)}>
	<div class="overflow-hidden rounded-lg border bg-muted/50">
		<!-- Header - always visible -->
		<div class="flex items-center justify-between px-3 py-2">
			<button
				type="button"
				onclick={() => (isExpanded = !isExpanded)}
				class="flex items-center gap-2 rounded px-2 py-1 text-left transition hover:bg-muted-foreground/5"
			>
				<FileJson class="h-4 w-4 text-muted-foreground" />
				<span class="text-sm font-medium">Tool Result: {displayToolName()}</span>
				{#if isExpanded}
					<ChevronDown class="ml-1 h-4 w-4 text-muted-foreground" />
				{:else}
					<ChevronRight class="ml-1 h-4 w-4 text-muted-foreground" />
				{/if}
			</button>
			<button
				type="button"
				onclick={handleCopy}
				class="rounded p-1.5 transition hover:bg-muted-foreground/10"
				title="Copy result"
				aria-label="Copy result"
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

		<!-- Content - collapsed by default -->
		{#if isExpanded}
			<div class="border-t bg-background/50">
				<pre
					class="max-h-96 overflow-x-auto overflow-y-auto p-3 text-xs break-words whitespace-pre-wrap">{formattedResult()}</pre>
			</div>
		{/if}
	</div>
</div>
