<script lang="ts">
	import {
		ModelBadge,
		ChatMessageActions,
		ChatMessageStatistics,
		ChatMessageThinkingBlock,
		MarkdownContent,
		ModelsSelector,
		ToolCallBlock
	} from '$lib/components/app';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { useModelChangeValidation } from '$lib/hooks/use-model-change-validation.svelte';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { autoResizeTextarea, copyToClipboard } from '$lib/utils';
	import { fade } from 'svelte/transition';
	import { Check, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { INPUT_CLASSES } from '$lib/constants/input-classes';
	import Label from '$lib/components/ui/label/label.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';

	interface Props {
		class?: string;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		editedContent?: string;
		isEditing?: boolean;
		message: DatabaseMessage;
		messageContent: string | undefined;
		onCancelEdit?: () => void;
		onCopy: () => void;
		onConfirmDelete: () => void;
		onContinue?: () => void;
		onDelete: () => void;
		onEdit?: () => void;
		onEditKeydown?: (event: KeyboardEvent) => void;
		onEditedContentChange?: (content: string) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onRegenerate: (modelOverride?: string) => void;
		onSaveEdit?: () => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		onShouldBranchAfterEditChange?: (value: boolean) => void;
		showDeleteDialog: boolean;
		shouldBranchAfterEdit?: boolean;
		siblingInfo?: ChatMessageSiblingInfo | null;
		textareaElement?: HTMLTextAreaElement;
		thinkingContent: string | null;
		toolCallContent: ApiChatCompletionToolCall[] | string | null;
		/** Map of tool name -> {result, timestamp} */
		toolResults?: Record<string, { result: string; timestamp: number }>;
		/** Map of tool name -> execution status */
		toolExecutionStatus?: Record<string, 'streaming' | 'calling' | 'complete' | 'error'>;
	}

	let {
		class: className = '',
		deletionInfo,
		editedContent = '',
		isEditing = false,
		message,
		messageContent,
		onCancelEdit,
		onConfirmDelete,
		onContinue,
		onCopy,
		onDelete,
		onEdit,
		onEditKeydown,
		onEditedContentChange,
		onNavigateToSibling,
		onRegenerate,
		onSaveEdit,
		onShowDeleteDialogChange,
		onShouldBranchAfterEditChange,
		showDeleteDialog,
		shouldBranchAfterEdit = false,
		siblingInfo = null,
		textareaElement = $bindable(),
		thinkingContent,
		toolCallContent = null,
		toolResults = {},
		toolExecutionStatus = {}
	}: Props = $props();

	const toolCalls = $derived(
		Array.isArray(toolCallContent) ? (toolCallContent as ApiChatCompletionToolCall[]) : null
	);
	const fallbackToolCalls = $derived(typeof toolCallContent === 'string' ? toolCallContent : null);

	// Get status for a tool call
	function getToolCallStatus(
		toolCall: ApiChatCompletionToolCall
	): 'streaming' | 'calling' | 'complete' | 'error' {
		const toolName = toolCall.function?.name;
		if (!toolName) return 'streaming';

		// Check explicit status first
		if (toolExecutionStatus[toolName]) {
			return toolExecutionStatus[toolName];
		}

		// If we have a result, it's complete
		if (toolResults[toolName] !== undefined) {
			return 'complete';
		}

		// If message has no timestamp, still streaming
		if (!message.timestamp) {
			return 'streaming';
		}

		// Message is complete but no result yet - calling
		return 'calling';
	}

	// Get result for a tool call
	function getToolCallResult(toolCall: ApiChatCompletionToolCall): string | null {
		const toolName = toolCall.function?.name;
		if (!toolName) return null;
		return toolResults[toolName]?.result ?? null;
	}

	// Get duration for a tool call (in ms)
	function getToolCallDuration(toolCall: ApiChatCompletionToolCall): number | null {
		const toolName = toolCall.function?.name;
		if (!toolName) return null;
		const resultInfo = toolResults[toolName];
		if (!resultInfo || !message.timestamp) return null;
		// Ensure non-negative duration in case of timestamp misalignment
		return Math.max(0, resultInfo.timestamp - message.timestamp);
	}

	const processingState = useProcessingState();
	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());
	let displayedModel = $derived((): string | null => {
		if (message.model) {
			return message.model;
		}

		return null;
	});

	const { handleModelChange } = useModelChangeValidation({
		getRequiredModalities: () => conversationsStore.getModalitiesUpToMessage(message.id),
		onSuccess: (modelName) => onRegenerate(modelName)
	});

	function handleCopyModel() {
		const model = displayedModel();

		void copyToClipboard(model ?? '');
	}

	$effect(() => {
		if (isEditing && textareaElement) {
			autoResizeTextarea(textareaElement);
		}
	});
</script>

<div
	class="text-md group w-full leading-7.5 {className}"
	role="group"
	aria-label="Assistant message with actions"
>
	{#if thinkingContent}
		<ChatMessageThinkingBlock
			reasoningContent={thinkingContent}
			isStreaming={!message.timestamp}
			hasRegularContent={!!messageContent?.trim()}
		/>
	{/if}

	{#if message?.role === 'assistant' && isLoading() && !message?.content?.trim()}
		<div class="mt-6 w-full max-w-[48rem]" in:fade>
			<div class="processing-container">
				<span class="processing-text">
					{processingState.getProcessingMessage()}
				</span>
			</div>
		</div>
	{/if}

	{#if isEditing}
		<div class="w-full">
			<textarea
				bind:this={textareaElement}
				bind:value={editedContent}
				class="min-h-[50vh] w-full resize-y rounded-2xl px-3 py-2 text-sm {INPUT_CLASSES}"
				onkeydown={onEditKeydown}
				oninput={(e) => {
					autoResizeTextarea(e.currentTarget);
					onEditedContentChange?.(e.currentTarget.value);
				}}
				placeholder="Edit assistant message..."
			></textarea>

			<div class="mt-2 flex items-center justify-between">
				<div class="flex items-center space-x-2">
					<Checkbox
						id="branch-after-edit"
						bind:checked={shouldBranchAfterEdit}
						onCheckedChange={(checked) => onShouldBranchAfterEditChange?.(checked === true)}
					/>
					<Label for="branch-after-edit" class="cursor-pointer text-sm text-muted-foreground">
						Branch conversation after edit
					</Label>
				</div>
				<div class="flex gap-2">
					<Button class="h-8 px-3" onclick={onCancelEdit} size="sm" variant="outline">
						<X class="mr-1 h-3 w-3" />
						Cancel
					</Button>

					<Button class="h-8 px-3" onclick={onSaveEdit} disabled={!editedContent?.trim()} size="sm">
						<Check class="mr-1 h-3 w-3" />
						Save
					</Button>
				</div>
			</div>
		</div>
	{:else if message.role === 'assistant'}
		{#if config().disableReasoningFormat}
			<pre class="raw-output">{messageContent || ''}</pre>
		{:else}
			<MarkdownContent content={messageContent || ''} />
		{/if}
	{:else}
		<div class="text-sm whitespace-pre-wrap">
			{messageContent}
		</div>
	{/if}

	<!-- Tool calls with unified display (always shown when present) -->
	{#if toolCalls && toolCalls.length > 0}
		<div class="mt-4 space-y-2">
			{#each toolCalls as toolCall, index (toolCall.id ?? `${index}`)}
				<ToolCallBlock
					{toolCall}
					result={getToolCallResult(toolCall)}
					status={getToolCallStatus(toolCall)}
					durationMs={getToolCallDuration(toolCall)}
				/>
			{/each}
		</div>
	{:else if fallbackToolCalls}
		<div class="mt-4 text-xs text-muted-foreground">
			<pre class="overflow-x-auto rounded bg-muted/30 p-2">{fallbackToolCalls}</pre>
		</div>
	{/if}

	<div class="info my-6 grid gap-4">
		{#if displayedModel()}
			<div class="inline-flex flex-wrap items-start gap-2 text-xs text-muted-foreground">
				{#if isRouter}
					<ModelsSelector
						currentModel={displayedModel()}
						onModelChange={handleModelChange}
						disabled={isLoading()}
						upToMessageId={message.id}
					/>
				{:else}
					<ModelBadge model={displayedModel() || undefined} onclick={handleCopyModel} />
				{/if}

				{#if currentConfig.showMessageStats && message.timings && message.timings.predicted_n && message.timings.predicted_ms}
					<ChatMessageStatistics
						promptTokens={message.timings.prompt_n}
						promptMs={message.timings.prompt_ms}
						predictedTokens={message.timings.predicted_n}
						predictedMs={message.timings.predicted_ms}
					/>
				{/if}
			</div>
		{/if}
	</div>

	{#if message.timestamp && !isEditing}
		<ChatMessageActions
			role="assistant"
			justify="start"
			actionsPosition="left"
			{siblingInfo}
			{showDeleteDialog}
			{deletionInfo}
			{onCopy}
			{onEdit}
			{onRegenerate}
			onContinue={currentConfig.enableContinueGeneration && !thinkingContent
				? onContinue
				: undefined}
			{onDelete}
			{onConfirmDelete}
			{onNavigateToSibling}
			{onShowDeleteDialogChange}
		/>
	{/if}
</div>

<style>
	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(
			90deg,
			var(--muted-foreground),
			var(--foreground),
			var(--muted-foreground)
		);
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}

	.raw-output {
		width: 100%;
		max-width: 48rem;
		margin-top: 1.5rem;
		padding: 1rem 1.25rem;
		border-radius: 1rem;
		background: hsl(var(--muted) / 0.3);
		color: var(--foreground);
		font-family:
			ui-monospace, SFMono-Regular, 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas,
			'Liberation Mono', Menlo, monospace;
		font-size: 0.875rem;
		line-height: 1.6;
		white-space: pre-wrap;
		word-break: break-word;
	}
</style>
