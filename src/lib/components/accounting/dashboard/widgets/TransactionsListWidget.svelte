<script lang="ts">
	// Recent transactions (or drafts, via options.status). Scrollable table.
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { goto } from '$app/navigation';
	import dayjs from 'dayjs';
	import { getTransactions } from '$lib/apis/accounting';
	import Badge from '$lib/components/common/Badge.svelte';
	import { money } from '../format';

	export let companyId: number;
	export let options: { status?: 'all' | 'draft'; limit?: number } = {};

	const i18n: any = getContext('i18n');
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');
	$: nativeCurrency = ($companyCurrencyCtx as any) || 'EUR';

	let rows: any[] = [];
	let loading = true;
	$: status = options?.status ?? 'all';
	$: limit = options?.limit ?? 20;

	onMount(load);
	async function load() {
		loading = true;
		try {
			const res = await getTransactions({
				company_id: companyId,
				limit,
				...(status === 'draft' ? { status: 'draft' } : {})
			});
			rows = Array.isArray(res) ? res : res?.transactions ?? res?.items ?? [];
		} catch {
			rows = [];
		}
		loading = false;
	}

	const statusType = (s: string) => (s === 'posted' ? 'success' : s === 'voided' ? 'error' : 'muted');
	const typeColor = (tt: string) => {
		switch (tt) {
			case 'invoice': return 'bg-blue-500/20 text-blue-700 dark:text-blue-200';
			case 'payment':
			case 'payment_in':
			case 'payment_out': return 'bg-purple-500/20 text-purple-700 dark:text-purple-200';
			case 'bill': return 'bg-orange-500/20 text-orange-700 dark:text-orange-200';
			default: return 'bg-gray-500/20 text-gray-700 dark:text-gray-200';
		}
	};
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if rows.length === 0}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No transactions') : 'No transactions'}</div>
{:else}
	<div class="h-full overflow-y-auto -mx-1">
		<table class="w-full text-xs text-left text-gray-500 dark:text-gray-400">
			<thead class="text-[10px] uppercase text-gray-500 dark:text-gray-400 sticky top-0 bg-white dark:bg-gray-900">
				<tr class="border-b border-gray-100 dark:border-gray-850/40">
					<th class="px-1.5 py-1.5">{i18n?.t ? i18n.t('Date') : 'Date'}</th>
					<th class="px-1.5 py-1.5">{i18n?.t ? i18n.t('Type') : 'Type'}</th>
					<th class="px-1.5 py-1.5">{i18n?.t ? i18n.t('Description') : 'Description'}</th>
					<th class="px-1.5 py-1.5 text-right">{i18n?.t ? i18n.t('Total') : 'Total'}</th>
					<th class="px-1.5 py-1.5">{i18n?.t ? i18n.t('Status') : 'Status'}</th>
				</tr>
			</thead>
			<tbody>
				{#each rows as tx}
					{@const mv = money(tx.total ?? tx.amount ?? 0, nativeCurrency, $displayCurrency, $exchangeRates ?? [], tx.transaction_date)}
					<tr
						class="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850/50 transition border-b border-gray-50 dark:border-gray-850/20"
						on:click={() => goto(`/accounting/company/${companyId}/entries?id=${tx.id}`)}
					>
						<td class="px-1.5 py-1">{tx.transaction_date ? dayjs(tx.transaction_date).format('YYYY-MM-DD') : '-'}</td>
						<td class="px-1.5 py-1">
							<span class="text-[10px] font-medium {typeColor(tx.transaction_type)} px-1 rounded uppercase">{tx.transaction_type ?? '-'}</span>
						</td>
						<td class="px-1.5 py-1 max-w-[160px] truncate">{tx.description ?? tx.memo ?? '-'}</td>
						<td class="px-1.5 py-1 text-right tabular-nums">
							{#if mv.converting && mv.hasRate}
								{mv.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span>
							{:else}
								{mv.original} <span class="text-[9px] text-gray-400">{nativeCurrency}</span>
							{/if}
						</td>
						<td class="px-1.5 py-1"><Badge type={statusType(tx.status)} content={tx.status ?? 'draft'} /></td>
					</tr>
				{/each}
			</tbody>
		</table>
	</div>
{/if}
