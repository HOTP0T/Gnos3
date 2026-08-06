<script lang="ts">
	// Upcoming recurring entries, ordered by next run date. From /recurring.
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import dayjs from 'dayjs';
	import { getRecurringTemplates } from '$lib/apis/accounting';

	export let companyId: number;
	export let options: { limit?: number } = {};
	const i18n: any = getContext('i18n');
	$: limit = options?.limit ?? 12;

	let rows: any[] = [];
	let loading = true;

	onMount(load);
	async function load() {
		loading = true;
		try {
			const res = await getRecurringTemplates(companyId);
			const list = Array.isArray(res) ? res : res?.items ?? [];
			rows = list
				.filter((r: any) => r.is_active !== false)
				.sort((a: any, b: any) => String(a.next_run_date ?? '9999').localeCompare(String(b.next_run_date ?? '9999')))
				.slice(0, limit);
		} catch {
			rows = [];
		}
		loading = false;
	}

	const dueTone = (d?: string) => {
		if (!d) return 'text-gray-400';
		const days = dayjs(d).diff(dayjs(), 'day');
		if (days < 0) return 'text-red-600 dark:text-red-400';
		if (days <= 7) return 'text-amber-600 dark:text-amber-400';
		return 'text-gray-500 dark:text-gray-400';
	};
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if rows.length === 0}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No recurring entries') : 'No recurring entries'}</div>
{:else}
	<div class="h-full overflow-y-auto -mx-1">
		<table class="w-full text-xs text-left text-gray-500 dark:text-gray-400">
			<tbody>
				{#each rows as r}
					<tr
						class="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850/50 transition border-b border-gray-50 dark:border-gray-850/20"
						on:click={() => goto(`/accounting/company/${companyId}/recurring`)}
					>
						<td class="px-1.5 py-1 max-w-[150px] truncate text-gray-700 dark:text-gray-300">{r.name ?? '-'}</td>
						<td class="px-1.5 py-1 text-gray-400">{r.frequency ?? ''}</td>
						<td class="px-1.5 py-1 text-right whitespace-nowrap tabular-nums {dueTone(r.next_run_date)}">
							{r.next_run_date ? dayjs(r.next_run_date).format('YYYY-MM-DD') : '—'}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}
