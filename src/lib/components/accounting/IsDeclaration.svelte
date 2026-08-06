<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getCitDeclaration, saveTaxFiling, exportTaxWorksheet } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import TaxFilingsList from './TaxFilingsList.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	let calculating = false;
	let saving = false;
	let periodStart = '';
	let periodEnd = '';
	let priorYearLosses = '0';
	let citAlreadyPaid = '0';
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
				cit_already_paid: parseFloat(citAlreadyPaid) || 0
			});
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to calculate CIT')}: ${err?.detail ?? err}`);
		}
		calculating = false;
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
				cit_already_paid: parseFloat(citAlreadyPaid) || 0
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
			<input id="cit-paid" type="number" step="0.01" bind:value={citAlreadyPaid} class={inputCls} />
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
						<td class="px-4 py-2">{$i18n.t('Less: CIT already paid')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.cit_already_paid)}</td>
					</tr>
					<tr class="font-bold bg-blue-50/50 dark:bg-blue-900/20">
						<td class="px-4 py-2">{$i18n.t('CIT to be paid')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(result.cit_due)} {result.currency ?? ''}</td>
					</tr>
				</tbody>
			</table>
		</div>

		<div class="flex gap-2 mb-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
				disabled={saving}
				on:click={save}
			>
				{saving ? $i18n.t('Saving...') : $i18n.t('Save as Filing')}
			</button>
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
