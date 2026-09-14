<script lang="ts">
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		getTaxConfig,
		getTaxDeclaration,
		createTaxEntry,
		saveTaxFiling,
		exportTaxWorksheet,
		getPeriods
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import TaxFilingsList from './TaxFilingsList.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let companyId: number;

	// State
	let loading = false;
	let calculating = false;
	let creating = false;
	let saving = false;
	let taxConfig: any = null;
	let declaration: any = null;
	let filingsList: any;

	// Credit brought forward from the previous return (上期留抵 / crédit reporté).
	// Blank = taken from the previous filing (or recomputed); a value overrides it.
	let openingCreditInput = '';

	// Month selector
	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; from: string; to: string }> = [];

	// ─── Helpers ────────────────────────────────────────────────────────────────

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '0.00';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	// Period options follow the country's filing frequency: one entry per
	// month, quarter or fiscal period, spanning the company's accounting periods.
	function buildPeriodOptions(periods: any[], frequency: string) {
		const options: typeof monthOptions = [];
		const pad = (n: number) => String(n).padStart(2, '0');
		const lastDay = (y: number, m: number) => new Date(y, m + 1, 0).getDate();
		for (const p of periods) {
			const start = new Date(p.start_date);
			const end = new Date(p.end_date);
			if (frequency === 'yearly') {
				options.push({
					value: `${p.start_date}_${p.end_date}`,
					label: p.name ?? `${p.start_date} → ${p.end_date}`,
					from: p.start_date,
					to: p.end_date
				});
				continue;
			}
			const step = frequency === 'quarterly' ? 3 : 1;
			let cursor = new Date(start.getFullYear(), Math.floor(start.getMonth() / step) * step, 1);
			while (cursor <= end) {
				const y = cursor.getFullYear();
				const m = cursor.getMonth();
				const mEnd = m + step - 1;
				const from = `${y}-${pad(m + 1)}-01`;
				const to = `${y}-${pad(mEnd + 1)}-${pad(lastDay(y, mEnd))}`;
				const label =
					step === 3
						? `Q${m / 3 + 1} ${y}`
						: cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				const value = step === 3 ? `${y}-Q${m / 3 + 1}` : `${y}-${pad(m + 1)}`;
				options.push({ value, label, from, to });
				cursor = new Date(y, m + step, 1);
			}
		}
		const seen = new Map<string, (typeof options)[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.from.localeCompare(a.from));
	}

	// ─── Data loading ───────────────────────────────────────────────────────────

	onMount(async () => {
		loading = true;
		try {
			const [config, res] = await Promise.all([
				getTaxConfig(companyId),
				getPeriods({ company_id: companyId })
			]);
			taxConfig = config;
			const periods = res.periods ?? res ?? [];
			monthOptions = buildPeriodOptions(periods, config?.filing_frequency ?? 'monthly');

			// Default to the period containing today, else the latest one.
			const today = new Date().toISOString().slice(0, 10);
			const match = monthOptions.find((o) => o.from <= today && today <= o.to);
			selectedMonth = match?.value ?? monthOptions[0]?.value ?? '';
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load tax configuration')}: ${err}`);
		}
		loading = false;
	});

	// ─── Calculate ──────────────────────────────────────────────────────────────

	const handleCalculate = async () => {
		const opt = monthOptions.find((o) => o.value === selectedMonth);
		if (!opt) {
			toast.error($i18n.t('Please select a period'));
			return;
		}
		calculating = true;
		declaration = null;
		try {
			declaration = await getTaxDeclaration({
				company_id: companyId,
				period_start: opt.from,
				period_end: opt.to,
				opening_credit: openingCreditInput.trim() === '' ? undefined : parseFloat(openingCreditInput) || 0
			});
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to calculate tax declaration') + ': ' + msg);
		}
		calculating = false;
	};

	// ─── Create settlement entry ──────────────────────────────────────────────────

	const handleCreateEntry = async () => {
		if (!declaration?.suggested_entry) return;
		creating = true;
		try {
			const result = await createTaxEntry(companyId, {
				tax_type: 'vat',
				period_start: declaration.period_start,
				period_end: declaration.period_end,
				entry: declaration.suggested_entry,
				tax_amount: declaration.grand_total ?? declaration.payable ?? 0,
				currency: declaration.currency,
				details: declaration
			});
			const txId = result?.transaction_id ?? '';
			toast.success(
				$i18n.t('Settlement entry created as Draft and linked to the filing') + (txId ? ` (ID: ${txId})` : '')
			);
			filingsList?.reload();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create settlement entry') + ': ' + msg);
		}
		creating = false;
	};

	// ─── Save filing + export ─────────────────────────────────────────────────────

	const handleSaveFiling = async () => {
		if (!declaration) return;
		saving = true;
		try {
			await saveTaxFiling(companyId, {
				tax_type: 'vat',
				period_start: declaration.period_start,
				period_end: declaration.period_end,
				tax_amount: declaration.grand_total ?? declaration.payable ?? 0,
				currency: declaration.currency,
				details: declaration
			});
			toast.success($i18n.t('Filing saved'));
			filingsList?.reload();
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save filing')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	const handleExport = async () => {
		if (!declaration) return;
		try {
			await exportTaxWorksheet({
				company_id: companyId,
				tax_type: 'vat',
				period_start: declaration.period_start,
				period_end: declaration.period_end,
				opening_credit: openingCreditInput.trim() === '' ? undefined : parseFloat(openingCreditInput) || 0
			});
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to export')}: ${err?.detail ?? err}`);
		}
	};

	$: taxName = taxConfig?.tax_name ?? 'Tax';
	$: collectedLabel = taxConfig?.collected_label ?? $i18n.t('Tax Collected');
	$: deductibleLabel = taxConfig?.deductible_label ?? $i18n.t('Tax Deductible');
	$: payableLabel = taxConfig?.payable_label ?? $i18n.t('Tax Payable');
	$: hasSurcharges = !!declaration && (declaration.surcharges ?? []).length > 0;
	$: periodLabel =
		taxConfig?.filing_frequency === 'quarterly'
			? $i18n.t('Quarter')
			: taxConfig?.filing_frequency === 'yearly'
				? $i18n.t('Fiscal period')
				: $i18n.t('Month');
	$: creditSourceText = (() => {
		const src: string = declaration?.opening_credit_source ?? 'none';
		if (src === 'manual') return $i18n.t('entered manually');
		if (src.startsWith('filing:')) return `${$i18n.t('from the filing ending')} ${src.slice(7)}`;
		if (src.startsWith('computed:')) return `${$i18n.t('recomputed from the period ending')} ${src.slice(9)} (${$i18n.t('no filing saved')})`;
		return $i18n.t('no previous period');
	})();
</script>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">
				{$i18n.t('VAT Declaration')}
			</div>
			{#if taxConfig}
				<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
					{taxConfig.tax_name_full ?? taxName}
				</div>
			{/if}
		</div>
	</div>

	<!-- Description -->
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t('Calculate VAT (and surcharges) for a period, save it as a filing, and export the worksheet.')}
	</div>

	{#if loading}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else}
		<!-- Month selector + Calculate -->
		<div class="flex flex-wrap gap-3 items-end mb-4">
			<div>
				<label
					for="tax-month"
					class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>
					{periodLabel}
				</label>
				{#if monthOptions.length > 0}
					<select
						id="tax-month"
						bind:value={selectedMonth}
						class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
					>
						{#each monthOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				{:else}
					<span class="text-xs text-gray-400 italic">
						{$i18n.t('No accounting periods defined')}
					</span>
				{/if}
			</div>
			<div>
				<label for="tax-opening-credit" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Credit brought forward')}
				</label>
				<input
					id="tax-opening-credit"
					type="number"
					step="0.01"
					bind:value={openingCreditInput}
					placeholder={$i18n.t('from previous filing')}
					class="text-sm rounded-lg px-3 py-1.5 w-44 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
				/>
			</div>
			<button
				class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
				disabled={!selectedMonth || calculating}
				on:click={handleCalculate}
			>
				{$i18n.t('Calculate')}
			</button>
		</div>

		<!-- Loading banner -->
		{#if calculating}
			<div
				class="relative overflow-hidden rounded-xl border border-blue-200/50 dark:border-blue-800/30 bg-blue-50 dark:bg-blue-900/20 p-4 mb-4"
			>
				<div class="absolute top-0 left-0 h-1 bg-blue-500 animate-pulse" style="width: 100%;"></div>
				<div class="flex items-center gap-3">
					<Spinner className="size-5 text-blue-600 dark:text-blue-400" />
					<span class="text-sm font-medium text-blue-700 dark:text-blue-300">
						{taxName}
						{$i18n.t('declaration loading...')}
					</span>
				</div>
			</div>
		{/if}

		<!-- Results -->
		{#if declaration && !calculating}
			{#if (declaration.warnings ?? []).length > 0}
				<div
					class="bg-amber-50 dark:bg-amber-900/20 border border-amber-200/50 dark:border-amber-800/30 rounded-xl p-3 mb-4 text-xs text-amber-800 dark:text-amber-200 space-y-1"
				>
					{#each declaration.warnings as w}
						<div>{w}</div>
					{/each}
					<button class="underline" on:click={() => dispatch('gotoAccounts')}>{$i18n.t('Open Tax accounts')}</button>
				</div>
			{/if}

			<!-- Summary Card -->
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30 mb-4"
			>
				<div class="text-sm font-medium dark:text-gray-200 mb-3">
					{$i18n.t('Summary')}
				</div>
				<div class="grid grid-cols-1 md:grid-cols-4 gap-4">
					<div
						class="bg-green-50 dark:bg-green-900/20 rounded-lg p-3 text-center border border-green-200/30 dark:border-green-800/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">{collectedLabel}</div>
						<div class="text-lg font-bold text-green-700 dark:text-green-400">
							{fmt(declaration.collected ?? 0)}
						</div>
					</div>
					<div
						class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 text-center border border-blue-200/30 dark:border-blue-800/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">{deductibleLabel}</div>
						<div class="text-lg font-bold text-blue-700 dark:text-blue-400">
							{fmt(declaration.deductible ?? 0)}
						</div>
					</div>
					<div
						class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 text-center border border-purple-200/30 dark:border-purple-800/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Credit brought forward')}</div>
						<div class="text-lg font-bold text-purple-700 dark:text-purple-400">
							{fmt(declaration.opening_credit ?? 0)}
						</div>
					</div>
					<div
						class="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-center border border-gray-200/30 dark:border-gray-700/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
							{hasSurcharges ? $i18n.t('Total VAT and surcharges') : payableLabel}
						</div>
						<div class="text-lg font-bold dark:text-gray-200">
							{fmt(hasSurcharges ? declaration.grand_total : declaration.payable ?? 0)}
							<span class="text-xs font-normal text-gray-400">{declaration.currency ?? ''}</span>
						</div>
					</div>
				</div>
			</div>

			<!-- VAT computation (the return's own arithmetic) -->
			<div
				class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-4 overflow-x-auto"
			>
				<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 text-sm font-medium dark:text-gray-200">
					{$i18n.t('VAT computation')}
				</div>
				<table class="w-full text-sm text-left">
					<tbody class="text-gray-900 dark:text-gray-100">
						<tr class="border-b border-gray-100 dark:border-gray-850">
							<td class="px-4 py-2">{collectedLabel}</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.collected)}</td>
						</tr>
						<tr class="border-b border-gray-100 dark:border-gray-850">
							<td class="px-4 py-2">{$i18n.t('Less')}: {deductibleLabel}</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.deductible)}</td>
						</tr>
						<tr class="border-b border-gray-100 dark:border-gray-850 font-medium">
							<td class="px-4 py-2">{$i18n.t('Net for the period')}</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.net)}</td>
						</tr>
						<tr class="border-b border-gray-100 dark:border-gray-850">
							<td class="px-4 py-2">
								{$i18n.t('Credit brought forward')}
								<span class="text-xs text-gray-400">({creditSourceText})</span>
							</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.opening_credit)}</td>
						</tr>
						<tr class="border-b border-gray-100 dark:border-gray-850">
							<td class="px-4 py-2">{$i18n.t('Less')}: {$i18n.t('credit used this period')}</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.credit_used)}</td>
						</tr>
						<tr class="border-b border-gray-100 dark:border-gray-850 font-bold bg-blue-50/50 dark:bg-blue-900/20">
							<td class="px-4 py-2">{payableLabel}</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.payable)} {declaration.currency ?? ''}</td>
						</tr>
						<tr class="font-medium">
							<td class="px-4 py-2">{taxConfig?.receivable_label ?? $i18n.t('Credit carried forward')} → {$i18n.t('next period')}</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(declaration.credit)}</td>
						</tr>
					</tbody>
				</table>
			</div>

			<!-- Per-account breakdown -->
			<div
				class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-4 overflow-x-auto"
			>
				<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 text-sm font-medium dark:text-gray-200">
					{$i18n.t('By account')}
				</div>
				<table class="w-full text-sm text-left">
					<thead class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800">
						<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
							<th class="px-3 py-2">{$i18n.t('Account')}</th>
							<th class="px-3 py-2">{$i18n.t('Role')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Debit')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Credit')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Amount')}</th>
						</tr>
					</thead>
					<tbody class="text-gray-900 dark:text-gray-100 text-xs">
						{#each declaration.detail_collectee ?? [] as d}
							<tr class="border-b border-gray-100 dark:border-gray-850">
								<td class="px-3 py-2 font-mono">{d.account_code} <span class="font-sans text-gray-500">{d.account_name}</span></td>
								<td class="px-3 py-2 text-green-700 dark:text-green-400">{collectedLabel}</td>
								<td class="px-3 py-2 text-right font-mono">{fmt(d.debit)}</td>
								<td class="px-3 py-2 text-right font-mono">{fmt(d.credit)}</td>
								<td class="px-3 py-2 text-right font-mono">{fmt(d.amount)}</td>
							</tr>
						{/each}
						{#each declaration.detail_deductible ?? [] as d}
							<tr class="border-b border-gray-100 dark:border-gray-850">
								<td class="px-3 py-2 font-mono">{d.account_code} <span class="font-sans text-gray-500">{d.account_name}</span></td>
								<td class="px-3 py-2 text-blue-700 dark:text-blue-400">{deductibleLabel}</td>
								<td class="px-3 py-2 text-right font-mono">{fmt(d.debit)}</td>
								<td class="px-3 py-2 text-right font-mono">{fmt(d.credit)}</td>
								<td class="px-3 py-2 text-right font-mono">{fmt(d.amount)}</td>
							</tr>
						{/each}
						{#if (declaration.detail_collectee ?? []).length + (declaration.detail_deductible ?? []).length === 0}
							<tr><td colspan="5" class="px-3 py-4 text-center text-gray-400">{$i18n.t('No VAT accounts mapped for this company.')}</td></tr>
						{/if}
					</tbody>
				</table>
			</div>

			<!-- VAT & Surcharges breakdown (only when the country levies surcharges) -->
			{#if hasSurcharges}
				<div
					class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-4 overflow-x-auto"
				>
					<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 text-sm font-medium dark:text-gray-200">
						{$i18n.t('VAT & Surcharges')}
					</div>
					<table class="w-full text-sm text-left">
						<tbody class="text-gray-900 dark:text-gray-100">
							<tr class="border-b border-gray-100 dark:border-gray-850">
								<td class="px-4 py-2">{payableLabel}</td>
								<td class="px-4 py-2 text-right font-mono">{fmt(declaration.payable)}</td>
							</tr>
							{#each declaration.surcharges as s}
								<tr class="border-b border-gray-100 dark:border-gray-850">
									<td class="px-4 py-2">
										{$i18n.t(s.label)}
										<span class="text-xs text-gray-400">({(s.rate * 100).toFixed(2)}%)</span>
										{#if s.payable_account}
											<span class="text-[10px] font-mono text-gray-400 ml-1">→ {s.payable_account.code}</span>
										{/if}
									</td>
									<td class="px-4 py-2 text-right font-mono">{fmt(s.amount)}</td>
								</tr>
							{/each}
							<tr class="border-b border-gray-100 dark:border-gray-850 font-medium">
								<td class="px-4 py-2">{$i18n.t('Total surcharges')}</td>
								<td class="px-4 py-2 text-right font-mono">{fmt(declaration.surcharge_total)}</td>
							</tr>
							<tr class="font-bold bg-blue-50/50 dark:bg-blue-900/20">
								<td class="px-4 py-2">{$i18n.t('Total VAT and surcharges')}</td>
								<td class="px-4 py-2 text-right font-mono">{fmt(declaration.grand_total)} {declaration.currency ?? ''}</td>
							</tr>
						</tbody>
					</table>
				</div>
			{/if}

			<!-- Actions -->
			<div class="flex flex-wrap gap-2 mb-4">
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
					disabled={saving}
					on:click={handleSaveFiling}
				>
					{saving ? $i18n.t('Saving...') : $i18n.t('Save as Filing')}
				</button>
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
					on:click={handleExport}
				>
					{$i18n.t('Export Excel')}
				</button>
			</div>

			<!-- Suggested Entry Table -->
			{#if (declaration.entry_blockers ?? []).length > 0}
				<div
					class="bg-white dark:bg-gray-900 rounded-xl border border-red-200/50 dark:border-red-800/30 mb-4 p-4"
				>
					<div class="text-sm font-medium dark:text-gray-200 mb-1">{$i18n.t('Settlement entry not available')}</div>
					<ul class="text-xs text-red-700 dark:text-red-300 list-disc pl-4 space-y-0.5">
						{#each declaration.entry_blockers as b}
							<li>{b}</li>
						{/each}
					</ul>
					<button
						class="mt-2 px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
						on:click={() => dispatch('gotoAccounts')}
					>
						{$i18n.t('Open Tax accounts')}
					</button>
				</div>
			{:else if declaration.suggested_entry?.lines?.length > 0}
				<div
					class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-4"
				>
					<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850">
						<div class="text-sm font-medium dark:text-gray-200">
							{$i18n.t('Suggested Settlement Entry')}
						</div>
					</div>
					<div class="overflow-x-auto">
						<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
							<thead
								class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
							>
								<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
									<th class="px-3 py-2">{$i18n.t('Account Code')}</th>
									<th class="px-3 py-2">{$i18n.t('Description')}</th>
									<th class="px-3 py-2 text-right">{$i18n.t('Debit')}</th>
									<th class="px-3 py-2 text-right">{$i18n.t('Credit')}</th>
								</tr>
							</thead>
							<tbody>
								{#each declaration.suggested_entry.lines as line}
									<tr
										class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
									>
										<td class="px-3 py-2 font-mono font-medium dark:text-gray-200">
											{line.account_code ?? ''}
										</td>
										<td class="px-3 py-2">{line.description ?? line.account_name ?? ''}</td>
										<td class="px-3 py-2 text-right font-mono">{line.debit ? fmt(line.debit) : ''}</td>
										<td class="px-3 py-2 text-right font-mono">{line.credit ? fmt(line.credit) : ''}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
					<div class="px-4 py-3 border-t border-gray-100 dark:border-gray-850">
						<button
							class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
							disabled={creating}
							on:click={handleCreateEntry}
						>
							{creating
								? $i18n.t('Creating...')
								: $i18n.t('Create Settlement Entry (Draft)')}
						</button>
					</div>
				</div>
			{/if}
		{/if}

		<TaxFilingsList {companyId} taxType="vat" bind:this={filingsList} />
	{/if}
</div>
