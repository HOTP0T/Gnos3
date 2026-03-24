<script lang="ts">
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getAPAging, getARAging } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;
	export let reportType: 'ap' | 'ar' = 'ap';

	let loading = false;
	let data: any = null;
	let asOf = new Date().toISOString().slice(0, 10);

	const load = async () => {
		loading = true;
		try {
			const fn = reportType === 'ap' ? getAPAging : getARAging;
			data = await fn({ company_id: companyId, as_of: asOf });
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '—';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const entityLabel = reportType === 'ap' ? 'Vendor' : 'Customer';
	const title = reportType === 'ap' ? 'Accounts Payable Aging' : 'Accounts Receivable Aging';

	const bucketColor = (bucket: string) => {
		switch (bucket) {
			case 'current': return 'text-green-600 dark:text-green-400';
			case '31-60': return 'text-yellow-600 dark:text-yellow-400';
			case '61-90': return 'text-orange-600 dark:text-orange-400';
			case '90+': return 'text-red-600 dark:text-red-400';
			default: return '';
		}
	};
</script>

<div class="space-y-3">
	<div class="flex flex-wrap gap-3 items-end">
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('As of')}</label>
			<input type="date" bind:value={asOf} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
		</div>
		<button
			class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
			on:click={load}
		>{$i18n.t('Generate')}</button>
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if data}
		<!-- Summary cards -->
		<div class="grid grid-cols-5 gap-2">
			<div class="rounded-lg p-3 bg-green-50 dark:bg-green-900/20 text-center">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{$i18n.t('Current')}</div>
				<div class="text-sm font-bold font-mono text-green-700 dark:text-green-400">{fmt(data.summary.current)}</div>
			</div>
			<div class="rounded-lg p-3 bg-yellow-50 dark:bg-yellow-900/20 text-center">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">31–60</div>
				<div class="text-sm font-bold font-mono text-yellow-700 dark:text-yellow-400">{fmt(data.summary.days_31_60)}</div>
			</div>
			<div class="rounded-lg p-3 bg-orange-50 dark:bg-orange-900/20 text-center">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">61–90</div>
				<div class="text-sm font-bold font-mono text-orange-700 dark:text-orange-400">{fmt(data.summary.days_61_90)}</div>
			</div>
			<div class="rounded-lg p-3 bg-red-50 dark:bg-red-900/20 text-center">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">90+</div>
				<div class="text-sm font-bold font-mono text-red-700 dark:text-red-400">{fmt(data.summary.over_90)}</div>
			</div>
			<div class="rounded-lg p-3 bg-gray-50 dark:bg-gray-850 text-center">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{$i18n.t('Total')}</div>
				<div class="text-sm font-bold font-mono dark:text-gray-200">{fmt(data.summary.total)}</div>
			</div>
		</div>

		{#if data.rows.length === 0}
			<div class="text-sm text-gray-400 italic text-center py-6">{$i18n.t('No outstanding balances')}</div>
		{:else}
			<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2">{$i18n.t(entityLabel)}</th>
							<th class="px-2 py-2">{$i18n.t('Invoice')}</th>
							<th class="px-2 py-2">{$i18n.t('Date')}</th>
							<th class="px-2 py-2">{$i18n.t('Due')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Original')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Paid')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Balance')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Days')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Bucket')}</th>
						</tr>
					</thead>
					<tbody>
						{#each data.rows as row}
							<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
								<td class="px-3 py-1.5 font-medium max-w-[150px] truncate">{row.vendor_or_customer}</td>
								<td class="px-2 py-1.5 font-mono">{row.invoice_number ?? '—'}</td>
								<td class="px-2 py-1.5">{row.invoice_date ?? '—'}</td>
								<td class="px-2 py-1.5">{row.due_date ?? '—'}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(row.original_amount)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(row.paid_amount)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-medium">{fmt(row.balance)}</td>
								<td class="px-2 py-1.5 text-center">{row.days_outstanding}</td>
								<td class="px-2 py-1.5 text-center">
									<span class="font-medium {bucketColor(row.bucket)}">{row.bucket}</span>
								</td>
							</tr>
						{/each}
					</tbody>
					<tfoot class="font-bold bg-gray-50/50 dark:bg-gray-850/30 border-t-2 border-gray-200 dark:border-gray-700">
						<tr>
							<td class="px-3 py-2" colspan="6">{$i18n.t('Total')}</td>
							<td class="px-2 py-2 text-right font-mono">{fmt(data.summary.total)}</td>
							<td colspan="2"></td>
						</tr>
					</tfoot>
				</table>
			</div>
		{/if}
	{/if}
</div>
