<script lang="ts">
	// Current accounting period summary. Loads its own periods list.
	import { onMount, getContext } from 'svelte';
	import dayjs from 'dayjs';
	import { getPeriods } from '$lib/apis/accounting';
	import Badge from '$lib/components/common/Badge.svelte';

	export let companyId: number;
	export let options: Record<string, any> = {}; // unused; part of widget contract
	const i18n: any = getContext('i18n');

	let period: any = null;
	let loading = true;

	onMount(async () => {
		try {
			const res = await getPeriods({ company_id: companyId });
			const list = Array.isArray(res) ? res : res?.items ?? [];
			period = list.find((p: any) => !p.is_closed) ?? list[0] ?? null;
		} catch {
			period = null;
		}
		loading = false;
	});
</script>

<div class="h-full flex flex-col justify-center">
	{#if loading}
		<div class="text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
	{:else if !period}
		<div class="text-xs text-gray-400">{i18n?.t ? i18n.t('No periods defined') : 'No periods defined'}</div>
	{:else}
		<div class="flex items-center gap-2 flex-wrap">
			<span class="text-sm font-medium text-gray-800 dark:text-gray-200">
				{period.name ?? `${period.start_date} – ${period.end_date}`}
			</span>
			<Badge type={!period.is_closed ? 'success' : 'muted'} content={i18n?.t ? i18n.t(!period.is_closed ? 'Open' : 'Closed') : (!period.is_closed ? 'Open' : 'Closed')} />
		</div>
		{#if period.start_date && period.end_date}
			<div class="text-[11px] text-gray-400 mt-1">
				{dayjs(period.start_date).format('YYYY-MM-DD')} — {dayjs(period.end_date).format('YYYY-MM-DD')}
			</div>
		{/if}
	{/if}
</div>
