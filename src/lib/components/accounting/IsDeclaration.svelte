<script lang="ts">
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getCitDeclaration, saveTaxFiling, exportTaxWorksheet, createTaxEntry } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import TaxFilingsList from './TaxFilingsList.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let companyId: number;

	let calculating = false;
	let saving = false;
	let creating = false;
	let periodStart = '';
	let periodEnd = '';
	let priorYearLosses = '0';
	// Blank = let the backend net off the CIT filings already paid inside the period.
	let citAlreadyPaid = '';
	let result: any = null;
	let filingsList: any;

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};
	const pct = (r: any) =>
		r === null || r === undefined ? '—' : `${(Number(r) * 100).toFixed(2)}%`;

	onMount(() => {
		const now = new Date();
		periodStart = `${now.getFullYear()}-01-01`;
		periodEnd = now.toISOString().slice(0, 10);
	});

	const calc = async () => {
		if (!periodStart || !periodEnd) {
			toast.error($i18n.t('Select a period'));
			return;
		}
		calculating = true;
		result = null;
		try {
			result = await getCitDeclaration({
				company_id: companyId,
				period_start: periodStart,
				period_end: periodEnd,
				prior_year_losses: parseFloat(priorYearLosses) || 0,
				cit_already_paid: citAlreadyPaid.trim() === '' ? undefined : parseFloat(citAlreadyPaid) || 0
			});
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to calculate CIT')}: ${err?.detail ?? err}`);
		}
		calculating = false;
	};

	const createEntry = async () => {
		if (!result?.suggested_entry) return;
		creating = true;
		try {
			const r = await createTaxEntry(companyId, {
				tax_type: 'cit',
				period_start: result.period_start,
				period_end: result.period_end,
				entry: result.suggested_entry,
				tax_amount: result.cit_due ?? 0,
				currency: result.currency,
				details: result
			});
			toast.success(
				$i18n.t('Settlement entry created as Draft and linked to the filing') +
					(r?.transaction_id ? ` (ID: ${r.transaction_id})` : '')
			);
			filingsList?.reload();
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to create settlement entry')}: ${err?.detail ?? err}`);
		}
		creating = false;
	};

	const save = async () => {
		if (!result) return;
		saving = true;
		try {
			await saveTaxFiling(companyId, {
				tax_type: 'cit',
				period_start: result.period_start,
				period_end: result.period_end,
				tax_amount: result.cit_due ?? 0,
				currency: result.currency,
				details: result
			});
			toast.success($i18n.t('Filing saved'));
			filingsList?.reload();
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save filing')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	const doExport = async () => {
		if (!result) return;
		try {
			await exportTaxWorksheet({
				company_id: companyId,
				tax_type: 'cit',
				period_start: result.period_start,
				period_end: result.period_end,
				prior_year_losses: parseFloat(priorYearLosses) || 0,
				cit_already_paid: citAlreadyPaid.trim() === '' ? undefined : parseFloat(citAlreadyPaid) || 0
			});
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to export')}: ${err?.detail ?? err}`);
		}
	};

	const inputCls =
		'text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden';
	const lblCls = 'block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1';
</script>

<div class="py-2">
	<div class="flex md:self-center text-lg font-medium px-0.5 gap-2 mb-1">
		<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Corporate Income Tax (IS)')}</div>
	</div>
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t('Computes taxable profit from ledger revenue − cost; rates come from Settings → Countries.')}
	</div>

	<!-- Inputs -->
	<div class="flex flex-wrap gap-3 items-end mb-4">
		<div>
			<label class={lblCls} for="cit-start">{$i18n.t('Period start')}</label>
			<input id="cit-start" type="date" bind:value={periodStart} class={inputCls} />
		</div>
		<div>
			<label class={lblCls} for="cit-end">{$i18n.t('Period end')}</label>
			<input id="cit-end" type="date" bind:value={periodEnd} class={inputCls} />
		</div>
		<div>
			<label class={lblCls} for="cit-losses">{$i18n.t('Prior-year losses')}</label>
			<input id="cit-losses" type="number" step="0.01" bind:value={priorYearLosses} class={inputCls} />
		</div>
		<div>
			<label class={lblCls} for="cit-paid">{$i18n.t('CIT already paid')}</label>
			<input
				id="cit-paid"
				type="number"
				step="0.01"
				bind:value={citAlreadyPaid}
				placeholder={$i18n.t('from paid filings')}
				class={inputCls}
			/>
		</div>
		<button
			class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
			disabled={calculating}
			on:click={calc}
		>
			{$i18n.t('Calculate')}
		</button>
	</div>

	{#if calculating}
		<div class="flex items-center gap-3 my-6">
			<Spinner className="size-5 text-blue-600 dark:text-blue-400" />
			<span class="text-sm text-blue-700 dark:text-blue-300">{$i18n.t('Calculating...')}</span>
		</div>
	{/if}

	{#if result && !calculating}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-3 overflow-x-auto"
		>
			<table class="w-full text-sm text-left">
				<tbody class="text-gray-900 dark:text-gray-100">
					<tr class="border-b border-gray-100 dark:border-gray-850">
						<td class="px-4 py-2">{$i18n.t('Total Revenue')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.revenue)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850">
						<td class="px-4 py-2">{$i18n.t('Total Cost')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.cost)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850 font-medium">
						<td class="px-4 py-2">{$i18n.t('Income before tax')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.income_before_tax)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850">
						<td class="px-4 py-2">{$i18n.t('Less: prior-year losses')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.prior_year_losses)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850 font-semibold bg-gray-50 dark:bg-gray-800/40">
						<td class="px-4 py-2">{$i18n.t('Profit to be declared')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.profit_to_declare)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850">
						<td class="px-4 py-2">{$i18n.t('Corporate income tax')} ({pct(result.cit_rate)})</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.cit)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850">
						<td class="px-4 py-2">{$i18n.t('Less: preferential exemption')} ({pct(result.cit_preferential_rate)})</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.preferential)}</td>
					</tr>
					<tr class="border-b border-gray-100 dark:border-gray-850">
						<td class="px-4 py-2">
							{$i18n.t('Less: CIT already paid')}
							{#if citAlreadyPaid.trim() === ''}
								<span class="text-xs text-gray-400">({$i18n.t('paid filings in the period')}: {fmt(result.cit_paid_from_filings)})</span>
							{/if}
						</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.cit_already_paid)}</td>
					</tr>
					<tr class="font-bold bg-blue-50/50 dark:bg-blue-900/20">
						<td class="px-4 py-2">{$i18n.t('CIT to be paid')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.cit_due)} {result.currency ?? ''}</td>
					</tr>
				</tbody>
			</table>
		</div>

		{#if (result.warnings ?? []).length > 0}
			<div
				class="bg-amber-50 dark:bg-amber-900/20 border border-amber-200/50 dark:border-amber-800/30 rounded-xl p-3 mb-3 text-xs text-amber-800 dark:text-amber-200 space-y-1"
			>
				{#each result.warnings as w}<div>{w}</div>{/each}
			</div>
		{/if}

		{#if (result.entry_blockers ?? []).length > 0}
			<div class="bg-white dark:bg-gray-900 rounded-xl border border-red-200/50 dark:border-red-800/30 mb-3 p-4">
				<div class="text-sm font-medium dark:text-gray-200 mb-1">{$i18n.t('Accrual entry not available')}</div>
				<ul class="text-xs text-red-700 dark:text-red-300 list-disc pl-4 space-y-0.5">
					{#each result.entry_blockers as b}<li>{b}</li>{/each}
				</ul>
				<button
					class="mt-2 px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
					on:click={() => dispatch('gotoAccounts')}
				>
					{$i18n.t('Open Tax accounts')}
				</button>
			</div>
		{:else if result.suggested_entry?.lines?.length > 0}
			<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-3">
				<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 text-sm font-medium dark:text-gray-200">
					{$i18n.t('Suggested Accrual Entry')}
				</div>
				<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
					<thead class="text-xs font-bold uppercase bg-gray-100 dark:bg-gray-800">
						<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
							<th class="px-3 py-2">{$i18n.t('Account Code')}</th>
							<th class="px-3 py-2">{$i18n.t('Description')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Debit')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Credit')}</th>
						</tr>
					</thead>
					<tbody>
						{#each result.suggested_entry.lines as line}
							<tr class="border-b border-gray-100 dark:border-gray-850 text-xs">
								<td class="px-3 py-2 font-mono font-medium">{line.account_code} <span class="font-sans text-gray-500">{line.account_name ?? ''}</span></td>
								<td class="px-3 py-2">{line.description ?? ''}</td>
								<td class="px-3 py-2 text-right font-mono">{line.debit ? fmt(line.debit) : ''}</td>
								<td class="px-3 py-2 text-right font-mono">{line.credit ? fmt(line.credit) : ''}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<div class="flex gap-2 mb-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
				disabled={saving}
				on:click={save}
			>
				{saving ? $i18n.t('Saving...') : $i18n.t('Save as Filing')}
			</button>
			{#if result.suggested_entry?.lines?.length > 0}
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
					disabled={creating}
					on:click={createEntry}
				>
					{creating ? $i18n.t('Creating...') : $i18n.t('Create Accrual Entry (Draft)')}
				</button>
			{/if}
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
				on:click={doExport}
			>
				{$i18n.t('Export Excel')}
			</button>
		</div>
	{/if}

	<TaxFilingsList {companyId} taxType="cit" bind:this={filingsList} />
</div>
