<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		getTaxConfig,
		getTaxDeclaration,
		createTaxEntry,
		getPeriods
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	// State
	let loading = false;
	let calculating = false;
	let creating = false;
	let taxConfig: any = null;
	let declaration: any = null;

	// Month selector
	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; from: string; to: string }> = [];

	// ─── Helpers ────────────────────────────────────────────────────────────────

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '0.00';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	function buildMonthOptions(periods: any[]) {
		const options: typeof monthOptions = [];
		for (const p of periods) {
			const start = new Date(p.start_date);
			const end = new Date(p.end_date);
			let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
			while (cursor <= end) {
				const y = cursor.getFullYear();
				const m = cursor.getMonth();
				const from = `${y}-${String(m + 1).padStart(2, '0')}-01`;
				const lastDay = new Date(y, m + 1, 0).getDate();
				const to = `${y}-${String(m + 1).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
				const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				options.push({ value: `${y}-${String(m + 1).padStart(2, '0')}`, label, from, to });
				cursor = new Date(y, m + 1, 1);
			}
		}
		const seen = new Map<string, (typeof options)[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
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
			monthOptions = buildMonthOptions(periods);

			const now = new Date();
			const curKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
			const match = monthOptions.find((o) => o.value === curKey);
			if (match) {
				selectedMonth = match.value;
			} else if (monthOptions.length > 0) {
				selectedMonth = monthOptions[0].value;
			}
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
				period_end: opt.to
			});
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to calculate tax declaration') + ': ' + msg);
		}
		calculating = false;
	};

	// ─── Create Entry ───────────────────────────────────────────────────────────

	const handleCreateEntry = async () => {
		if (!declaration?.suggested_entry) return;
		creating = true;
		try {
			const result = await createTaxEntry(companyId, declaration.suggested_entry);
			const txId = result?.id ?? result?.transaction_id ?? '';
			toast.success($i18n.t('Settlement entry created as Draft') + (txId ? ` (ID: ${txId})` : ''));
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create settlement entry') + ': ' + msg);
		}
		creating = false;
	};

	$: taxName = taxConfig?.tax_name ?? 'Tax';
	$: collectedLabel = taxConfig?.collected_label ?? $i18n.t('Tax Collected');
	$: deductibleLabel = taxConfig?.deductible_label ?? $i18n.t('Tax Deductible');
	$: payableLabel = taxConfig?.payable_label ?? $i18n.t('Tax Payable');
</script>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">
				{$i18n.t('Tax Declaration')}
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
		{$i18n.t('Calculate tax amounts for a given period and generate settlement entries.')}
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
					{$i18n.t('Month')}
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
			<button
				class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
				disabled={!selectedMonth || calculating}
				on:click={handleCalculate}
			>
				{$i18n.t('Calculate')}
			</button>
		</div>

		<!-- AI Loading Banner -->
		{#if calculating}
			<div
				class="relative overflow-hidden rounded-xl border border-blue-200/50 dark:border-blue-800/30 bg-blue-50 dark:bg-blue-900/20 p-4 mb-4"
			>
				<div
					class="absolute top-0 left-0 h-1 bg-blue-500 animate-pulse"
					style="width: 100%;"
				/>
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
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
							{collectedLabel}
						</div>
						<div class="text-lg font-bold text-green-700 dark:text-green-400">
							{fmt(declaration.collected ?? declaration.tax_collected ?? 0)}
						</div>
					</div>
					<div
						class="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3 text-center border border-blue-200/30 dark:border-blue-800/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
							{deductibleLabel}
						</div>
						<div class="text-lg font-bold text-blue-700 dark:text-blue-400">
							{fmt(declaration.deductible ?? declaration.tax_deductible ?? 0)}
						</div>
					</div>
					<div
						class="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-3 text-center border border-purple-200/30 dark:border-purple-800/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
							{$i18n.t('Net')}
						</div>
						<div
							class="text-lg font-bold {(declaration.net ?? declaration.net_tax ?? 0) >= 0 ? 'text-red-700 dark:text-red-400' : 'text-green-700 dark:text-green-400'}"
						>
							{fmt(declaration.net ?? declaration.net_tax ?? 0)}
						</div>
					</div>
					<div
						class="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 text-center border border-gray-200/30 dark:border-gray-700/30"
					>
						<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
							{payableLabel}
						</div>
						<div class="text-lg font-bold dark:text-gray-200">
							{fmt(declaration.payable ?? declaration.net ?? declaration.net_tax ?? 0)}
						</div>
					</div>
				</div>
			</div>

			<!-- Suggested Entry Table -->
			{#if declaration.suggested_entry?.lines?.length > 0}
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
										<td class="px-3 py-2">
											{line.description ?? line.account_name ?? ''}
										</td>
										<td class="px-3 py-2 text-right font-mono">
											{line.debit ? fmt(line.debit) : ''}
										</td>
										<td class="px-3 py-2 text-right font-mono">
											{line.credit ? fmt(line.credit) : ''}
										</td>
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
	{/if}
</div>
