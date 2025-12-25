<script lang="ts">
	import { ChatMessage } from '$lib/components/app';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { config } from '$lib/stores/settings.svelte';
	import { getMessageSiblings, isToolResultMessage, parseToolResult } from '$lib/utils';
	import { SvelteSet } from 'svelte/reactivity';

	interface Props {
		class?: string;
		messages?: DatabaseMessage[];
		onUserAction?: () => void;
	}

	let { class: className, messages = [], onUserAction }: Props = $props();

	let allConversationMessages = $state<DatabaseMessage[]>([]);
	const currentConfig = config();

	function refreshAllMessages() {
		const conversation = activeConversation();

		if (conversation) {
			conversationsStore.getConversationMessages(conversation.id).then((messages) => {
				allConversationMessages = messages;
			});
		} else {
			allConversationMessages = [];
		}
	}

	// Single effect that tracks both conversation and message changes
	$effect(() => {
		const conversation = activeConversation();

		if (conversation) {
			refreshAllMessages();
		}
	});

	// Tool result with timing info
	interface ToolResultInfo {
		result: string;
		timestamp: number;
	}

	// Build a map of message ID -> tool results (for assistant messages with tool calls)
	// Check both messages prop and allConversationMessages for completeness
	let toolResultsMap = $derived.by(() => {
		const map: Record<string, Record<string, ToolResultInfo>> = {};

		// Check messages from props first (always available)
		for (const msg of messages) {
			if (isToolResultMessage(msg)) {
				const parsed = parseToolResult(msg.content);
				if (parsed && msg.parent) {
					if (!map[msg.parent]) {
						map[msg.parent] = {};
					}
					map[msg.parent][parsed.toolName] = {
						result: parsed.result,
						timestamp: msg.timestamp
					};
				}
			}
		}

		// Also check allConversationMessages for any we might have missed
		for (const msg of allConversationMessages) {
			if (isToolResultMessage(msg)) {
				const parsed = parseToolResult(msg.content);
				if (parsed && msg.parent) {
					if (!map[msg.parent]) {
						map[msg.parent] = {};
					}
					map[msg.parent][parsed.toolName] = {
						result: parsed.result,
						timestamp: msg.timestamp
					};
				}
			}
		}

		return map;
	});

	// Set of message IDs that are tool results (to hide them)
	// Check both sources
	let toolResultMessageIds = $derived.by(() => {
		const ids = new SvelteSet<string>();
		for (const msg of messages) {
			if (isToolResultMessage(msg)) {
				ids.add(msg.id);
			}
		}
		for (const msg of allConversationMessages) {
			if (isToolResultMessage(msg)) {
				ids.add(msg.id);
			}
		}
		return ids;
	});

	let displayMessages = $derived.by(() => {
		if (!messages.length) {
			return [];
		}

		// Filter out system messages if showSystemMessage is false
		// Also filter out tool result messages (they're shown inline with tool calls)
		let filteredMessages = currentConfig.showSystemMessage
			? messages
			: messages.filter((msg) => msg.type !== 'system');

		// Hide tool result messages
		filteredMessages = filteredMessages.filter((msg) => !toolResultMessageIds.has(msg.id));

		return filteredMessages.map((message) => {
			const siblingInfo = getMessageSiblings(allConversationMessages, message.id);

			return {
				message,
				siblingInfo: siblingInfo || {
					message,
					siblingIds: [message.id],
					currentIndex: 0,
					totalSiblings: 1
				},
				toolResults: toolResultsMap[message.id] || {}
			};
		});
	});

	async function handleNavigateToSibling(siblingId: string) {
		await conversationsStore.navigateToSibling(siblingId);
	}

	async function handleEditWithBranching(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editMessageWithBranching(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleEditWithReplacement(
		message: DatabaseMessage,
		newContent: string,
		shouldBranch: boolean
	) {
		onUserAction?.();

		await chatStore.editAssistantMessage(message.id, newContent, shouldBranch);

		refreshAllMessages();
	}

	async function handleRegenerateWithBranching(message: DatabaseMessage, modelOverride?: string) {
		onUserAction?.();

		await chatStore.regenerateMessageWithBranching(message.id, modelOverride);

		refreshAllMessages();
	}

	async function handleContinueAssistantMessage(message: DatabaseMessage) {
		onUserAction?.();

		await chatStore.continueAssistantMessage(message.id);

		refreshAllMessages();
	}

	async function handleEditUserMessagePreserveResponses(
		message: DatabaseMessage,
		newContent: string,
		newExtras?: DatabaseMessageExtra[]
	) {
		onUserAction?.();

		await chatStore.editUserMessagePreserveResponses(message.id, newContent, newExtras);

		refreshAllMessages();
	}

	async function handleDeleteMessage(message: DatabaseMessage) {
		await chatStore.deleteMessage(message.id);

		refreshAllMessages();
	}
</script>

<div class="flex h-full flex-col space-y-10 pt-16 md:pt-24 {className}" style="height: auto; ">
	{#each displayMessages as { message, siblingInfo, toolResults } (message.id)}
		<ChatMessage
			class="mx-auto w-full max-w-[48rem]"
			{message}
			{siblingInfo}
			{toolResults}
			onDelete={handleDeleteMessage}
			onNavigateToSibling={handleNavigateToSibling}
			onEditWithBranching={handleEditWithBranching}
			onEditWithReplacement={handleEditWithReplacement}
			onEditUserMessagePreserveResponses={handleEditUserMessagePreserveResponses}
			onRegenerateWithBranching={handleRegenerateWithBranching}
			onContinueAssistantMessage={handleContinueAssistantMessage}
		/>
	{/each}
</div>
