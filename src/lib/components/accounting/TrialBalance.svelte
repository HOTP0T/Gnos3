<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getTrialBalance, getPeriods, exportTrialBalance } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let loading = false;
	let data: any = null;

	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; from: string; to: string; fiscalStart: string }> = [];

	// Manual override mode
	let manualMode = false;
	let manualPeriodStart = '';
	let manualAsOf = '';
	let manualYtdStart = '';

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
		const seen = new Map<string, typeof options[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
	}

	onMount(async () => {
		try {
			const res = await getPeriods({ company_id: companyId });
			const periods = res.periods ?? res ?? [];
			monthOptions = buildMonthOptions(periods);

			const now = new Date();
			const curKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
			const match = monthOptions.find(o => o.value === curKey);
			if (match) {
				selectedMonth = match.value;
			} else if (monthOptions.length > 0) {
				selectedMonth = monthOptions[0].value;
			}
		} catch (err) {
			console.error('Failed to load periods:', err);
		}
	});

	const load = async () => {
		let periodStart: string | undefined;
		let asOf: string | undefined;
		let ytdStart: string | undefined;

		if (manualMode) {
			periodStart = manualPeriodStart || undefined;
			asOf = manualAsOf || undefined;
			ytdStart = manualYtdStart || undefined;
		} else {
			const opt = monthOptions.find(o => o.value === selectedMonth);
			if (!opt) { toast.error($i18n.t('Please select a period')); return; }
			periodStart = opt.from;
			asOf = opt.to;
			ytdStart = opt.fiscalStart;
		}

		loading = true;
		try {
			data = await getTrialBalance({
				company_id: companyId,
				as_of: asOf,
				period_start: periodStart,
				ytd_start: ytdStart
			});
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const sumField = (rows: any[], field: string): number =>
		rows.reduce((s: number, r: any) => s + (parseFloat(r[field]) || 0), 0);

	$: selectedOpt = monthOptions.find(o => o.value === selectedMonth);
</script>

<div class="space-y-3">
	<div class="flex flex-wrap gap-3 items-end">
		{#if !manualMode}
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Month')}</label>
				{#if monthOptions.length > 0}
					<select
						bind:value={selectedMonth}
						class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
					>
						{#each monthOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				{:else}
					<span class="text-xs text-gray-400">{$i18n.t('No periods defined')}</span>
				{/if}
			</div>
			{#if selectedOpt}
				<div class="text-xs text-gray-400 dark:text-gray-500 self-center">
					{$i18n.t('Period')}: {selectedOpt.from} — {selectedOpt.to} | {$i18n.t('Fiscal Year Start')}: {selectedOpt.fiscalStart}
				</div>
			{/if}
		{:else}
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Period Start')}</label>
				<input type="date" bind:value={manualPeriodStart} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
			</div>
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('As Of (Period End)')}</label>
				<input type="date" bind:value={manualAsOf} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
			</div>
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Fiscal Year Start')}</label>
				<input type="date" bind:value={manualYtdStart} placeholder="YTD" class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
			</div>
		{/if}
		<button
			class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
			disabled={!manualMode && !selectedMonth}
			on:click={load}
		>{$i18n.t('Generate')}</button>
		{#if data}
			<button
				class="px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition flex items-center gap-1.5"
				on:click={() => {
					const opt = monthOptions.find(o => o.value === selectedMonth);
					exportTrialBalance({
						company_id: companyId,
						as_of: manualMode ? manualAsOf : opt?.to,
						period_start: manualMode ? manualPeriodStart : opt?.from,
						ytd_start: manualMode ? manualYtdStart : opt?.fiscalStart
					});
				}}
			>
				<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
				{$i18n.t('Export Excel')}
			</button>
		{/if}
		<button
			class="px-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-850 transition"
			on:click={() => { manualMode = !manualMode; }}
		>{manualMode ? $i18n.t('Use Period Selector') : $i18n.t('Manual Dates')}</button>
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if data}
		<div class="flex items-center gap-2 text-sm">
			<span class="font-medium dark:text-gray-200">{$i18n.t('Trial Balance')}</span>
			{#if data.period_label}<span class="text-xs text-gray-500">{$i18n.t('Period')}: {data.period_label}</span>{/if}
			{#if data.period_start}<span class="text-xs text-gray-500">({data.period_start} — {data.as_of})</span>
			{:else}<span class="text-xs text-gray-500">{$i18n.t('As of')} {data.as_of}</span>{/if}
			<span class="text-xs font-medium px-2 py-0.5 rounded-lg {data.is_balanced ? 'bg-green-500/20 text-green-700 dark:text-green-300' : 'bg-red-500/20 text-red-700 dark:text-red-300'}">{data.is_balanced ? $i18n.t('Balanced') : $i18n.t('Unbalanced')}</span>
		</div>

		<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
			<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300 whitespace-nowrap">
				<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
					<tr>
						{#if data.period_label}<th class="px-2 py-2">{$i18n.t('Period')}</th>{/if}
						<th class="px-2 py-2">{$i18n.t('Account Code')}</th>
						<th class="px-2 py-2">{$i18n.t('Account Name')}</th>
												<th class="px-2 py-2 text-right">{$i18n.t('Opening Bal. (DR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Opening Bal. (CR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Movement (DR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Movement (CR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Accum. YTD (DR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Accum. YTD (CR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Ending Bal. (DR)')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Ending Bal. (CR)')}</th>
					</tr>
				</thead>
				<tbody>
					{#each data.rows as row}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30 {row.is_parent ? 'bg-gray-50/80 dark:bg-gray-850/40 font-semibold' : ''}">
							{#if data.period_label}<td class="px-2 py-1.5">{data.period_label}</td>{/if}
							<td class="px-2 py-1.5 font-mono" style="padding-left: {8 + (row.level || 0) * 16}px">{row.account_code}</td>
							<td class="px-2 py-1.5">{row.account_name}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(row.opening_debit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(row.opening_credit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(row.movement_debit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(row.movement_credit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(row.accumulated_debit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(row.accumulated_credit)}</td>
							<td class="px-2 py-1.5 text-right font-mono {row.ending_debit ? 'text-blue-700 dark:text-blue-400' : ''}">{fmt(row.ending_debit)}</td>
							<td class="px-2 py-1.5 text-right font-mono {row.ending_credit ? 'text-red-700 dark:text-red-400' : ''}">{fmt(row.ending_credit)}</td>
						</tr>
					{/each}
				</tbody>
				<tfoot class="font-medium bg-gray-50 dark:bg-gray-850/50 text-gray-800 dark:text-gray-200">
					<tr class="border-t-2 border-gray-200 dark:border-gray-700">
						{#if data.period_label}<td class="px-2 py-2"></td>{/if}
						<td class="px-2 py-2" colspan="2">{$i18n.t('Total')}</td>
												<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'opening_debit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'opening_credit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'movement_debit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'movement_credit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'accumulated_debit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'accumulated_credit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'ending_debit'))}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(sumField(data.rows, 'ending_credit'))}</td>
					</tr>
				</tfoot>
			</table>
		</div>
	{/if}
</div>
