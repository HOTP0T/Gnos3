<script lang="ts">
	// Statutory financial statements — the fixed-line filings a jurisdiction
	// prescribes, as opposed to the account-by-account reports next door. Which
	// layout applies is decided by the company's country on the backend: China
	// (小企业会计准则, assets side by side with liabilities + equity) and Hong Kong
	// (SME-FRS, one vertical column down to net assets, then equity). The
	// component renders whatever line list the API returns.
	import { onMount, getContext } from 'svelte';
	import { buildMonthOptions, type MonthOption } from '$lib/utils/fiscalYear';
	import { toast } from 'svelte-sonner';
	import type { Writable } from 'svelte/store';

	import {
		getPeriods,
		getCompany,
		getStatutoryBalanceSheet,
		getStatutoryProfitLoss,
		exportStatutoryBalanceSheet,
		exportStatutoryProfitLoss,
		type StatementLine
	} from '$lib/apis/accounting';
	import { convertAmount } from '$lib/utils/currency';
	import ReportAmount from '$lib/components/accounting/ReportAmount.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;
	export let layoutLabel: string | null = null;
	export let layoutKey: string | null = null;
	export let presentation: 'side_by_side' | 'vertical' | null = null;

	$: hk = layoutKey === 'hk_sme';
	$: vertical = (presentation ?? (hk ? 'vertical' : 'side_by_side')) === 'vertical';
	$: tabs = hk
		? [
				{ id: 'balance-sheet', label: 'Statement of Financial Position 財務狀況表' },
				{ id: 'profit-loss', label: 'Statement of Profit or Loss 損益表' }
			]
		: [
				{ id: 'balance-sheet', label: '资产负债表 Balance Sheet' },
				{ id: 'profit-loss', label: '利润表 P&L' }
			];

	type Which = 'balance-sheet' | 'profit-loss';
	let which: Which = 'balance-sheet';
	let loading = false;
	let bs: any = null;
	let pl: any = null;

	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrency = getContext<Writable<string>>('companyCurrency');

	$: active = which === 'balance-sheet' ? bs : pl;
	$: baseCcy = active?.currency || $companyCurrency || 'CNY';
	$: converting = !!($displayCurrency && baseCcy && $displayCurrency !== baseCcy);
	$: fx =
		converting && active
			? convertAmount(1, baseCcy, $displayCurrency, $exchangeRates ?? [], active?.as_of)
			: null;
	$: fxProps = {
		factor: converting ? (fx?.hasRate ? fx.rate : null) : 1,
		converting,
		displayCcy: converting ? $displayCurrency : baseCcy,
		baseCcy
	};

	let selectedMonth = '';
	let monthOptions: MonthOption[] = [];


	onMount(async () => {
		try {
			const [res, co] = await Promise.all([getPeriods({ company_id: companyId }), getCompany(companyId).catch(() => null)]);
			monthOptions = buildMonthOptions(res.periods ?? res ?? [], Number(co?.fiscal_year_start_month ?? 1) || 1);
			const now = new Date();
			const curKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
			selectedMonth = monthOptions.find((o) => o.value === curKey)?.value ?? monthOptions[0]?.value ?? '';
			if (selectedMonth) await load();
		} catch (err) {
			console.error('Failed to load periods:', err);
		}
	});

	const load = async () => {
		const opt = monthOptions.find((o) => o.value === selectedMonth);
		if (!opt) {
			toast.error($i18n.t('Please select a period'));
			return;
		}
		loading = true;
		try {
			if (which === 'balance-sheet') {
				bs = await getStatutoryBalanceSheet({ company_id: companyId, as_of: opt.to });
			} else {
				pl = await getStatutoryProfitLoss({ company_id: companyId, as_of: opt.to });
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const doExport = async () => {
		const opt = monthOptions.find((o) => o.value === selectedMonth);
		if (!opt) return;
		if (which === 'balance-sheet') await exportStatutoryBalanceSheet({ company_id: companyId, as_of: opt.to });
		else await exportStatutoryProfitLoss({ company_id: companyId, as_of: opt.to });
	};

	// Reload when the statement or the month changes.
	$: if (selectedMonth && which) {
		load();
	}

	const label = (l: StatementLine) => (hk ? `${l.label_en}  ${l.label_zh}` : `${l.label_zh} ${l.label_en}`).trim();
	const visible = (rows: any[]) => (rows ?? []).filter((l: any) => !l.hidden);
	const monthHeader = (m: string) => (m.includes('-') ? m : `${pl?.fiscal_year}/${m}`);
	const rowClass = (l: StatementLine) =>
		l.header
			? 'font-semibold text-gray-700 dark:text-gray-300 bg-gray-50 dark:bg-gray-850/40'
			: l.is_total
				? 'font-semibold border-t border-gray-200 dark:border-gray-700'
				: l.memo
					? 'text-gray-500 dark:text-gray-400 italic'
					: '';

	$: monthsShown = pl ? (pl.months ?? []).slice(0, pl.months_through ?? 12) : [];
</script>

<div class="space-y-3">
	<!-- Controls -->
	<div class="flex flex-wrap items-center gap-3">
		<div class="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 w-fit">
			{#each tabs as t}
				<button
					class="px-3 py-1.5 text-sm font-medium rounded-md transition
						{which === t.id
						? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
						: 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'}"
					on:click={() => (which = t.id as Which)}
				>
					{t.label}
				</button>
			{/each}
		</div>

		<select
			bind:value={selectedMonth}
			class="text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
		>
			{#each monthOptions as o}
				<option value={o.value}>{o.label}</option>
			{/each}
		</select>

		<button
			class="px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition"
			on:click={doExport}
			disabled={!active}
		>
			{$i18n.t('Export Excel')}
		</button>

		{#if layoutLabel}
			<span class="text-xs text-gray-400">{layoutLabel}</span>
		{/if}
	</div>

	{#if loading}
		<div class="flex justify-center py-10"><Spinner /></div>
	{:else if which === 'balance-sheet' && bs}
		{#if !bs.is_balanced}
			<div class="px-3 py-2 text-sm rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300">
				{$i18n.t('Assets do not equal liabilities plus equity. The ledger detail behind one of the lines needs review.')}
			</div>
		{/if}
		{#if bs.unmapped?.length}
			<div class="px-3 py-2 text-xs rounded-lg bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-300">
				{$i18n.t('Accounts with no dedicated line, included under an "other" line:')}
				{bs.unmapped.map((u: any) => `${u.account_code}`).join(', ')}
			</div>
		{/if}

		<div class="grid grid-cols-1 {vertical ? '' : 'xl:grid-cols-2'} gap-4">
			{#each vertical
				? [{ title: `${bs.company_name} — ${$i18n.t('as at')} ${bs.as_of}`, rows: [...visible(bs.assets), ...visible(bs.liabilities), ...visible(bs.equity)] }]
				: [{ title: '资产 ASSETS', rows: bs.assets }, { title: '负债和所有者权益 LIABILITIES AND EQUITY', rows: [...bs.liabilities, ...bs.equity] }] as block}
				<div class="overflow-x-auto {vertical ? 'max-w-4xl' : ''}">
					<table class="w-full text-xs">
						<thead class="text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
							<tr>
								<th class="px-2 py-2 text-left">{block.title}</th>
								<th class="px-2 py-2 text-center w-12">{$i18n.t(vertical ? 'Note' : 'Line')}</th>
								{#if vertical}
									<th class="px-2 py-2 text-right w-32">{bs.as_of}</th>
									<th class="px-2 py-2 text-right w-32">{bs.beginning_date}</th>
								{:else}
									<th class="px-2 py-2 text-right w-32">{bs.beginning_date}</th>
									<th class="px-2 py-2 text-right w-32">{bs.as_of}</th>
								{/if}
							</tr>
						</thead>
						<tbody>
							{#each block.rows as l}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 {rowClass(l)}">
									<td class="px-2 py-1.5" style="padding-left: {0.5 + l.indent * 1}rem">{label(l)}</td>
									<td class="px-2 py-1.5 text-center text-gray-400">{l.line_no ?? ''}</td>
									<td class="px-2 py-1.5 text-right font-mono">
										{#if !l.header}<ReportAmount value={vertical ? l.ending : l.beginning} {...fxProps} />{/if}
									</td>
									<td class="px-2 py-1.5 text-right font-mono">
										{#if !l.header}<ReportAmount value={vertical ? l.beginning : l.ending} {...fxProps} />{/if}
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			{/each}
		</div>
	{:else if which === 'profit-loss' && pl}
		{#if pl.unmapped?.length}
			<div class="px-3 py-2 text-xs rounded-lg bg-amber-50 dark:bg-amber-900/20 text-amber-800 dark:text-amber-300">
				{$i18n.t('Accounts with no dedicated line, included under an "other" line:')}
				{pl.unmapped.map((u: any) => `${u.account_code}`).join(', ')}
			</div>
		{/if}
		<div class="overflow-x-auto">
			<table class="w-full text-xs">
				<thead class="text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
					<tr>
						<th class="px-2 py-2 text-left">{hk ? 'Item' : '项目 Item'}</th>
						<th class="px-2 py-2 text-center w-12">{$i18n.t(hk ? 'Note' : 'Line')}</th>
						{#each monthsShown as m}
							<th class="px-2 py-2 text-right w-28">{monthHeader(m)}</th>
						{/each}
						<th class="px-2 py-2 text-right w-32 font-semibold">YTD {pl.fiscal_year}</th>
					</tr>
				</thead>
				<tbody>
					{#each visible(pl.lines) as l}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 {rowClass(l)}">
							<td class="px-2 py-1.5" style="padding-left: {0.5 + l.indent * 1}rem">{label(l)}</td>
							<td class="px-2 py-1.5 text-center text-gray-400">{l.line_no ?? ''}</td>
							{#each monthsShown as m}
								<td class="px-2 py-1.5 text-right font-mono">
									<ReportAmount value={l.months?.[m]} {...fxProps} />
								</td>
							{/each}
							<td class="px-2 py-1.5 text-right font-mono font-semibold">
								<ReportAmount value={l.ytd} {...fxProps} />
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{:else}
		<div class="py-10 text-center text-sm text-gray-400 italic">
			{$i18n.t('Select a period to build the statement.')}
		</div>
	{/if}
</div>
