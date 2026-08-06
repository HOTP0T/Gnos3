<script lang="ts">
	// Unmatched bank statement lines — from /companies/{id}/unmatched-bank-lines.
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import dayjs from 'dayjs';
	import { getUnmatchedBankLines } from '$lib/apis/accounting';
	import { fmtNumber } from '../format';

	export let companyId: number;
	export let options: { limit?: number } = {};
	const i18n: any = getContext('i18n');
	$: limit = options?.limit ?? 20;

	let rows: any[] = [];
	let loading = true;

	onMount(load);
	async function load() {
		loading = true;
		try {
			const res = await getUnmatchedBankLines(companyId, { limit });
			rows = Array.isArray(res) ? res : res?.items ?? [];
		} catch {
			rows = [];
		}
		loading = false;
	}
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if rows.length === 0}
	<div class="h-full flex flex-col items-center justify-center text-xs text-gray-400 gap-1">
		<span>{i18n?.t ? i18n.t('All bank lines reconciled') : 'All bank lines reconciled'}</span>
	</div>
{:else}
	<div class="h-full overflow-y-auto -mx-1">
		<table class="w-full text-xs text-left text-gray-500 dark:text-gray-400">
			<tbody>
				{#each rows as line}
					<tr
						class="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850/50 transition border-b border-gray-50 dark:border-gray-850/20"
						on:click={() => goto(`/accounting/company/${companyId}/bank`)}
					>
						<td class="px-1.5 py-1 whitespace-nowrap">{line.transaction_date ? dayjs(line.transaction_date).format('MM-DD') : '-'}</td>
						<td class="px-1.5 py-1 max-w-[170px] truncate">{line.description ?? line.reference ?? '-'}</td>
						<td class="px-1.5 py-1 text-right tabular-nums {(+line.amount) < 0 ? 'text-red-600 dark:text-red-400' : 'text-gray-700 dark:text-gray-300'}">
							{fmtNumber(line.amount ?? 0)}
						</td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}
