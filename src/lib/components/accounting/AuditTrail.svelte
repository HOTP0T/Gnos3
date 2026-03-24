<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import { getAuditTrail } from '$lib/apis/accounting';

	const i18n = getContext('i18n');
	export let companyId: number;
	export let entityType: string | undefined = undefined;
	export let entityId: number | undefined = undefined;
	export let limit: number = 20;

	let entries: any[] = [];
	let total = 0;
	let loading = true;

	const load = async () => {
		loading = true;
		try {
			const res = await getAuditTrail({
				company_id: companyId,
				entity_type: entityType,
				entity_id: entityId,
				limit,
			});
			entries = res.entries ?? [];
			total = res.total ?? 0;
		} catch (err) { console.error(err); }
		loading = false;
	};

	onMount(load);

	const actionColor = (action: string) => {
		switch (action) {
			case 'created': return 'text-green-600 dark:text-green-400';
			case 'posted': return 'text-blue-600 dark:text-blue-400';
			case 'voided': return 'text-red-600 dark:text-red-400';
			case 'deleted': return 'text-red-600 dark:text-red-400';
			case 'updated': return 'text-yellow-600 dark:text-yellow-400';
			default: return 'text-gray-600 dark:text-gray-400';
		}
	};

	const actionIcon = (action: string) => {
		switch (action) {
			case 'created': return '+';
			case 'posted': return '\u2713';
			case 'voided': return '\u2715';
			case 'deleted': return '\u2212';
			case 'updated': return '~';
			default: return '\u2022';
		}
	};

	const formatTime = (ts: string) => {
		if (!ts) return '';
		const d = new Date(ts);
		return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
	};
</script>

{#if loading}
	<div class="text-xs text-gray-400">{$i18n.t('Loading...')}</div>
{:else if entries.length === 0}
	<div class="text-xs text-gray-400 italic">{$i18n.t('No activity yet')}</div>
{:else}
	<div class="space-y-1.5">
		{#each entries as entry}
			<div class="flex items-start gap-2 text-xs">
				<span class="font-mono font-bold text-sm w-4 text-center {actionColor(entry.action)}">{actionIcon(entry.action)}</span>
				<div class="flex-1 min-w-0">
					<span class="font-medium {actionColor(entry.action)}">{entry.action}</span>
					<span class="text-gray-500"> {entry.entity_type} #{entry.entity_id}</span>
					{#if entry.changes}
						<span class="text-gray-400 ml-1">
							{#if typeof entry.changes === 'object'}
								{Object.keys(entry.changes).join(', ')}
							{/if}
						</span>
					{/if}
				</div>
				<span class="text-gray-400 whitespace-nowrap text-[10px]">{formatTime(entry.timestamp)}</span>
			</div>
		{/each}
	</div>
	{#if total > entries.length}
		<div class="text-[10px] text-gray-400 mt-2">{$i18n.t('Showing')} {entries.length} / {total}</div>
	{/if}
{/if}
