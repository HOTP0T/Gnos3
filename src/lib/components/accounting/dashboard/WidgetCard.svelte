<script lang="ts">
	// Shared frame for every dashboard widget. IMPORTANT: none of this
	// component's props change when the dashboard enters/leaves edit mode — the
	// edit affordances (remove button, outline, drag cursor) are pure CSS keyed
	// off an ancestor `.grid-stack.is-editing` class. That's deliberate: mutating
	// a gridstack-owned item from Svelte makes gridstack lose track of it.
	import { createEventDispatcher, getContext } from 'svelte';
	const i18n: any = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let title = '';
	export let href: string | null = null;
	export let linkLabel = '';
	export let padded = true;
</script>

<div class="wcard h-full w-full flex flex-col bg-white dark:bg-gray-900 rounded-xl border border-gray-100/40 dark:border-gray-850/40 overflow-hidden relative">
	<!-- Remove button — CSS-hidden unless the grid is in edit mode. -->
	<button
		class="wcard-remove absolute top-1.5 right-1.5 z-20 text-gray-400 hover:text-red-500 bg-white/80 dark:bg-gray-900/80 rounded-md p-0.5 transition"
		title={i18n?.t ? i18n.t('Remove widget') : 'Remove widget'}
		on:pointerdown|stopPropagation
		on:click|stopPropagation={() => dispatch('remove')}
	>
		<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-4 h-4">
			<path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
		</svg>
	</button>

	{#if title}
		<div class="flex items-center gap-2 px-3.5 pt-3 pb-2 flex-shrink-0">
			<div class="text-xs font-medium text-gray-700 dark:text-gray-300 truncate flex-1">{title}</div>
			{#if href}
				<a {href} class="wcard-link text-[11px] text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition whitespace-nowrap">
					{linkLabel || (i18n?.t ? i18n.t('View all') : 'View all')} &rarr;
				</a>
			{/if}
		</div>
	{/if}
	<div class="flex-1 min-h-0 {padded ? 'px-3.5 pb-3.5' : ''} {title ? '' : padded ? 'pt-3.5' : ''} overflow-hidden">
		<slot />
	</div>
</div>
