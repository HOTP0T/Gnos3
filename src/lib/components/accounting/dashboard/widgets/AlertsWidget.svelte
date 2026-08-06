<script lang="ts">
	// Actionable alerts (drafts / unmatched bank lines / overdue invoices).
	// Reads the shared common-data store so it doesn't refetch stats.
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { goto } from '$app/navigation';
	import { COMMON_DATA_CTX, type CommonData } from '../store';

	export let companyId: number;
	export let options: Record<string, any> = {}; // unused; part of widget contract
	const i18n: any = getContext('i18n');
	const common = getContext<Writable<CommonData>>(COMMON_DATA_CTX);

	$: stats = $common.stats ?? {};
	$: drafts = stats.draft_count ?? 0;
	$: unmatched = stats.unmatched_bank_lines ?? 0;
	$: overdue = stats.overdue_invoices ?? 0;
	$: anyAlerts = drafts > 0 || unmatched > 0 || overdue > 0;
	const T = (k: string) => (i18n?.t ? i18n.t(k) : k);
</script>

<div class="h-full flex flex-col justify-center gap-1.5">
	{#if $common.loading}
		<div class="text-xs text-gray-400">{T('Loading…')}</div>
	{:else if !anyAlerts}
		<div class="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
			<span>&#10003;</span>{T('All clear — nothing needs attention')}
		</div>
	{:else}
		{#if drafts > 0}
			<button class="w-full flex items-center gap-2 px-3 py-1.5 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/40 rounded-lg text-xs text-amber-800 dark:text-amber-300 hover:bg-amber-100 dark:hover:bg-amber-900/30 transition text-left"
				on:click={() => goto(`/accounting/company/${companyId}/entries`)}>
				<span class="text-amber-500">&#9888;</span><span class="font-medium">{drafts}</span> {T('draft entries awaiting review')}
				<span class="ml-auto text-amber-500">&rarr;</span>
			</button>
		{/if}
		{#if unmatched > 0}
			<button class="w-full flex items-center gap-2 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/40 rounded-lg text-xs text-blue-800 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/30 transition text-left"
				on:click={() => goto(`/accounting/company/${companyId}/bank`)}>
				<span class="text-blue-500">&#9679;</span><span class="font-medium">{unmatched}</span> {T('unmatched bank lines')}
				<span class="ml-auto text-blue-500">&rarr;</span>
			</button>
		{/if}
		{#if overdue > 0}
			<button class="w-full flex items-center gap-2 px-3 py-1.5 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800/40 rounded-lg text-xs text-red-800 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/30 transition text-left"
				on:click={() => goto(`/accounting/company/${companyId}/invoices`)}>
				<span class="text-red-500">&#9888;</span><span class="font-medium">{overdue}</span> {T('overdue invoices')}
				<span class="ml-auto text-red-500">&rarr;</span>
			</button>
		{/if}
	{/if}
</div>
