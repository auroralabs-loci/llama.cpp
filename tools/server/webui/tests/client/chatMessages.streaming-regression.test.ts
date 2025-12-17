import { describe, it, expect } from 'vitest';
import { render } from 'vitest-browser-svelte';
import { conversationsStore } from '$lib/stores/conversations.svelte';
import TestReactiveMessagesWrapper from './components/TestReactiveMessagesWrapper.svelte';
import type { ChatRole, DatabaseMessage } from '$lib/types';

const msg = (
	id: string,
	role: ChatRole,
	content: string,
	parent: string | null,
	extra: Partial<DatabaseMessage> = {}
): DatabaseMessage => ({
	id,
	convId: 'c1',
	type: 'text',
	role,
	content,
	thinking: '',
	toolCalls: '',
	parent: parent ?? '-1',
	children: [],
	timestamp: Date.now(),
	...extra
});

/**
 * This test is designed to fail when reactivity breaks:
 * - We mount ChatMessages.
 * - We stream messages in phases.
 * - We assert that the second reasoning chunk appears without remount/refresh.
 */
describe('ChatMessages streaming regression', () => {
	it('renders later reasoning without refresh after a tool call', async () => {
		// reset store
		conversationsStore.activeMessages = [];

		const user = msg('u1', 'user', 'Question', null);
		const a1 = msg('a1', 'assistant', '', user.id, {
			thinking: 'reasoning-step-1',
			toolCalls: JSON.stringify([
				{
					id: 'call-1',
					type: 'function',
					function: { name: 'calculator', arguments: JSON.stringify({ expression: '1+1' }) }
				}
			])
		});
		const t1 = msg('t1', 'tool', JSON.stringify({ result: '2' }), a1.id, { toolCallId: 'call-1' });
		const a2 = msg('a2', 'assistant', '', t1.id, { thinking: 'reasoning-step-2' });
		const a3 = msg('a3', 'assistant', 'final-answer', a2.id);

		// Render wrapper consuming live store data
		const { container } = render(TestReactiveMessagesWrapper);

		// Phase 1: user + first assistant (with tool call), then tool result
		conversationsStore.addMessageToActive(user);
		conversationsStore.addMessageToActive(a1);
		conversationsStore.addMessageToActive(t1);

		// Let DOM update
		await Promise.resolve();
		await Promise.resolve();
		expect(container.textContent || '').toContain('reasoning-step-1');
		expect(container.textContent || '').not.toContain('reasoning-step-2');
		expect(container.textContent || '').not.toContain('final-answer');

		// Phase 2: stream in later reasoning (a2)
		conversationsStore.addMessageToActive(a2);
		await Promise.resolve();
		await Promise.resolve();

		const afterA2 = container.textContent || '';
		expect(afterA2).toContain('reasoning-step-1'); // old reasoning still present
		expect(afterA2).toContain('reasoning-step-2'); // new reasoning must appear without refresh

		// Phase 3: final assistant content
		conversationsStore.addMessageToActive(a3);
		await Promise.resolve();
		await Promise.resolve();

		const finalText = container.textContent || '';
		expect(finalText).toContain('final-answer');
	});
});
