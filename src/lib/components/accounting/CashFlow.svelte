<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getCashFlow, getPeriods, exportCashFlow } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let loading = false;
	let data: any = null;

	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; from: string; to: string; fiscalStart: string }> = [];

	function buildMonthOptions(periods: any[]) {
		const options: typeof monthOptions = [];
		for (const p of periods) {
			const start = new Date(p.start_date);
			const end = new Date(p.end_date);
			const fiscalStart = p.start_date;
			let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
			while (cursor <= end) {
				const y = cursor.getFullYear();
				const m = cursor.getMonth();
				const from = `${y}-${String(m + 1).padStart(2, '0')}-01`;
				const lastDay = new Date(y, m + 1, 0).getDate();
				const to = `${y}-${String(m + 1).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
				const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				options.push({ value: `${y}-${String(m + 1).padStart(2, '0')}`, label, from, to, fiscalStart });
				cursor = new Date(y, m + 1, 1);
			}
		}
		const seen = new Map<string, (typeof options)[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
	}

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const load = async () => {
		const opt = monthOptions.find((o) => o.value === selectedMonth);
		if (!opt) { toast.error($i18n.t('Please select a period')); return; }
		loading = true;
		try {
			data = await getCashFlow({ company_id: companyId, date_from: opt.from, date_to: opt.to });
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	onMount(async () => {
		try {
			const res = await getPeriods({ company_id: companyId });
			const periods = res.periods ?? res ?? [];
			monthOptions = buildMonthOptions(periods);
			if (monthOptions.length > 0) selectedMonth = monthOptions[0].value;
		} catch {}
	});
</script>

<div class="space-y-3">
	<div class="flex flex-wrap gap-3 items-end">
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Period')}</label>
			{#if monthOptions.length > 0}
				<select bind:value={selectedMonth} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden">
					{#each monthOptions as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			{:else}
				<span class="text-xs text-gray-400 italic">{$i18n.t('No accounting periods defined')}</span>
			{/if}
		</div>
		<button
			class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
			disabled={!selectedMonth}
			on:click={load}
		>{$i18n.t('Generate')}</button>
		{#if data}
			<button
				class="px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition flex items-center gap-1.5"
				on:click={() => {
					const opt = monthOptions.find(o => o.value === selectedMonth);
					if (opt) exportCashFlow({ company_id: companyId, date_from: opt.from, date_to: opt.to });
				}}
			>
				<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
				{$i18n.t('Export Excel')}
			</button>
		{/if}
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if data}
		<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 overflow-hidden">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead class="text-xs font-bold uppercase bg-gray-100 dark:bg-gray-800">
					<tr class="border-b border-gray-200 dark:border-gray-700">
						<th class="px-4 py-2">{$i18n.t('Item')}</th>
						<th class="px-4 py-2 text-right">{$i18n.t('Amount')}</th>
					</tr>
				</thead>
				<tbody>
					<!-- Operating Activities -->
					<tr class="bg-blue-50/50 dark:bg-blue-900/10 border-b border-gray-100 dark:border-gray-850">
						<td colspan="2" class="px-4 py-2 font-semibold text-blue-800 dark:text-blue-300">{$i18n.t('Operating Activities')}</td>
					</tr>
					<tr class="border-b border-gray-50 dark:border-gray-850/30 text-xs">
						<td class="px-4 py-1.5 pl-8">{$i18n.t('Net Income')}</td>
						<td class="px-4 py-1.5 text-right font-mono">{fmt(data.net_income)}</td>
					</tr>
					{#if data.depreciation}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 text-xs">
							<td class="px-4 py-1.5 pl-8">{$i18n.t('Add: Depreciation')}</td>
							<td class="px-4 py-1.5 text-right font-mono">{fmt(data.depreciation)}</td>
						</tr>
					{/if}
					{#each data.working_capital_changes ?? [] as item}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 text-xs">
							<td class="px-4 py-1.5 pl-8 text-gray-600 dark:text-gray-400">{item.account_code} {item.account_name}</td>
							<td class="px-4 py-1.5 text-right font-mono {item.change < 0 ? 'text-red-600 dark:text-red-400' : ''}">{fmt(item.change)}</td>
						</tr>
					{/each}
					<tr class="bg-blue-50 dark:bg-blue-900/20 border-b border-gray-200 dark:border-gray-700 font-semibold text-xs">
						<td class="px-4 py-2">{$i18n.t('Cash from Operations')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(data.cash_from_operations)}</td>
					</tr>

					<!-- Investing Activities -->
					<tr class="bg-amber-50/50 dark:bg-amber-900/10 border-b border-gray-100 dark:border-gray-850">
						<td colspan="2" class="px-4 py-2 font-semibold text-amber-800 dark:text-amber-300">{$i18n.t('Investing Activities')}</td>
					</tr>
					{#each data.investing_activities ?? [] as item}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 text-xs">
							<td class="px-4 py-1.5 pl-8 text-gray-600 dark:text-gray-400">{item.account_code} {item.account_name}</td>
							<td class="px-4 py-1.5 text-right font-mono {item.change < 0 ? 'text-red-600 dark:text-red-400' : ''}">{fmt(item.change)}</td>
						</tr>
					{/each}
					<tr class="bg-amber-50 dark:bg-amber-900/20 border-b border-gray-200 dark:border-gray-700 font-semibold text-xs">
						<td class="px-4 py-2">{$i18n.t('Cash from Investing')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(data.cash_from_investing)}</td>
					</tr>

					<!-- Financing Activities -->
					<tr class="bg-purple-50/50 dark:bg-purple-900/10 border-b border-gray-100 dark:border-gray-850">
						<td colspan="2" class="px-4 py-2 font-semibold text-purple-800 dark:text-purple-300">{$i18n.t('Financing Activities')}</td>
					</tr>
					{#each data.financing_activities ?? [] as item}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 text-xs">
							<td class="px-4 py-1.5 pl-8 text-gray-600 dark:text-gray-400">{item.account_code} {item.account_name}</td>
							<td class="px-4 py-1.5 text-right font-mono {item.change < 0 ? 'text-red-600 dark:text-red-400' : ''}">{fmt(item.change)}</td>
						</tr>
					{/each}
					<tr class="bg-purple-50 dark:bg-purple-900/20 border-b border-gray-200 dark:border-gray-700 font-semibold text-xs">
						<td class="px-4 py-2">{$i18n.t('Cash from Financing')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(data.cash_from_financing)}</td>
					</tr>

					<!-- Summary -->
					<tr class="bg-gray-100 dark:bg-gray-800 font-bold text-sm">
						<td class="px-4 py-3">{$i18n.t('Net Change in Cash')}</td>
						<td class="px-4 py-3 text-right font-mono {data.net_change_in_cash < 0 ? 'text-red-600 dark:text-red-400' : 'text-green-700 dark:text-green-400'}">{fmt(data.net_change_in_cash)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850 text-xs">
						<td class="px-4 py-1.5">{$i18n.t('Opening Cash')}</td>
						<td class="px-4 py-1.5 text-right font-mono">{fmt(data.opening_cash)}</td>
					</tr>
					<tr class="bg-green-50 dark:bg-green-900/20 font-bold text-sm">
						<td class="px-4 py-3">{$i18n.t('Closing Cash')}</td>
						<td class="px-4 py-3 text-right font-mono">{fmt(data.closing_cash)}</td>
					</tr>
				</tbody>
			</table>
		</div>
	{/if}
</div>
