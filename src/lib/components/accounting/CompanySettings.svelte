<script lang="ts">
	import { getContext } from 'svelte';
	import { slide } from 'svelte/transition';
	import { toast } from 'svelte-sonner';

	import ChartOfAccounts from '$lib/components/accounting/ChartOfAccounts.svelte';
	import AccountingPeriods from '$lib/components/accounting/AccountingPeriods.svelte';
	import CategorizationRules from '$lib/components/accounting/CategorizationRules.svelte';
	import ExchangeRates from '$lib/components/accounting/ExchangeRates.svelte';
	import FixedAssets from '$lib/components/accounting/FixedAssets.svelte';
	import RecurringTemplates from '$lib/components/accounting/RecurringTemplates.svelte';
	import ExcelImportModal from '$lib/components/accounting/ExcelImportModal.svelte';
	import {
		downloadChartImportTemplate,
		downloadPeriodImportTemplate
	} from '$lib/apis/accounting';

	const i18n = getContext('i18n');

	export let companyId: number;

	// Excel import modal state
	let showImportModal = false;
	let importType: 'chart' | 'period' = 'chart';

	// Keys to force re-render of child components after import
	let chartKey = 0;
	let periodKey = 0;

	// Collapse state — chart of accounts collapsed by default
	let collapsed: Record<string, boolean> = {
		chart: true,
		periods: false,
		categorization: false,
		exchange: false,
		assets: false,
		recurring: false
	};

	const toggle = (key: string) => {
		collapsed[key] = !collapsed[key];
	};

	const openImportAccounts = () => {
		importType = 'chart';
		showImportModal = true;
	};

	const openImportPeriods = () => {
		importType = 'period';
		showImportModal = true;
	};

	const handleImported = () => {
		if (importType === 'chart') {
			chartKey += 1;
			toast.success($i18n.t('Accounts imported successfully'));
		} else {
			periodKey += 1;
			toast.success($i18n.t('Periods imported successfully'));
		}
	};
</script>

<ExcelImportModal
	bind:show={showImportModal}
	type={importType}
	{companyId}
	on:imported={handleImported}
/>

<div class="py-2 space-y-3">
	<!-- Chart of Accounts Section -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
			on:click={() => toggle('chart')}
		>
			<div class="flex items-center gap-2">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4 transition-transform {collapsed.chart ? '-rotate-90' : ''}"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
				</svg>
				<span class="text-base font-medium dark:text-gray-200">
					{$i18n.t('Chart of Accounts')}
				</span>
			</div>
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<!-- svelte-ignore a11y-no-static-element-interactions -->
			<div class="flex items-center gap-2" on:click|stopPropagation>
				<button
					class="px-3.5 py-1.5 text-sm rounded-xl bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-500 dark:hover:bg-blue-600 font-medium transition flex items-center gap-1.5"
					on:click={openImportAccounts}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
					</svg>
					{$i18n.t('Import from Excel')}
				</button>
				<button
					class="px-3.5 py-1.5 text-sm rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white dark:bg-emerald-500 dark:hover:bg-emerald-600 font-medium transition flex items-center gap-1.5"
					on:click={downloadChartImportTemplate}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
					</svg>
					{$i18n.t('Download Template')}
				</button>
			</div>
		</div>
		{#if !collapsed.chart}
			<div class="px-4 pb-3" transition:slide={{ duration: 200 }}>
				{#key chartKey}
					<ChartOfAccounts {companyId} />
				{/key}
			</div>
		{/if}
	</div>

	<!-- Accounting Periods Section -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
			on:click={() => toggle('periods')}
		>
			<div class="flex items-center gap-2">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4 transition-transform {collapsed.periods ? '-rotate-90' : ''}"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
				</svg>
				<span class="text-base font-medium dark:text-gray-200">
					{$i18n.t('Accounting Periods')}
				</span>
			</div>
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<!-- svelte-ignore a11y-no-static-element-interactions -->
			<div class="flex items-center gap-2" on:click|stopPropagation>
				<button
					class="px-3.5 py-1.5 text-sm rounded-xl bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-500 dark:hover:bg-blue-600 font-medium transition flex items-center gap-1.5"
					on:click={openImportPeriods}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
					</svg>
					{$i18n.t('Import from Excel')}
				</button>
				<button
					class="px-3.5 py-1.5 text-sm rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white dark:bg-emerald-500 dark:hover:bg-emerald-600 font-medium transition flex items-center gap-1.5"
					on:click={downloadPeriodImportTemplate}
				>
					<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-4">
						<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
					</svg>
					{$i18n.t('Download Template')}
				</button>
			</div>
		</div>
		{#if !collapsed.periods}
			<div class="px-4 pb-3" transition:slide={{ duration: 200 }}>
				{#key periodKey}
					<AccountingPeriods {companyId} />
				{/key}
			</div>
		{/if}
	</div>

	<!-- Categorization Rules Section -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
			on:click={() => toggle('categorization')}
		>
			<div class="flex items-center gap-2">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4 transition-transform {collapsed.categorization ? '-rotate-90' : ''}"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
				</svg>
				<span class="text-base font-medium dark:text-gray-200">
					{$i18n.t('Categorization Rules')}
				</span>
			</div>
		</div>
		{#if !collapsed.categorization}
			<div class="px-4 pb-3" transition:slide={{ duration: 200 }}>
				<CategorizationRules {companyId} />
			</div>
		{/if}
	</div>

	<!-- Exchange Rates Section -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
			on:click={() => toggle('exchange')}
		>
			<div class="flex items-center gap-2">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4 transition-transform {collapsed.exchange ? '-rotate-90' : ''}"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
				</svg>
				<span class="text-base font-medium dark:text-gray-200">
					{$i18n.t('Exchange Rates')}
				</span>
			</div>
		</div>
		{#if !collapsed.exchange}
			<div class="px-4 pb-3" transition:slide={{ duration: 200 }}>
				<ExchangeRates {companyId} />
			</div>
		{/if}
	</div>

	<!-- Fixed Assets Section -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
			on:click={() => toggle('assets')}
		>
			<div class="flex items-center gap-2">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4 transition-transform {collapsed.assets ? '-rotate-90' : ''}"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
				</svg>
				<span class="text-base font-medium dark:text-gray-200">
					{$i18n.t('Fixed Assets')}
				</span>
			</div>
		</div>
		{#if !collapsed.assets}
			<div class="px-4 pb-3" transition:slide={{ duration: 200 }}>
				<FixedAssets {companyId} />
			</div>
		{/if}
	</div>

	<!-- Recurring Templates Section -->
	<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
			on:click={() => toggle('recurring')}
		>
			<div class="flex items-center gap-2">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4 transition-transform {collapsed.recurring ? '-rotate-90' : ''}"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
				</svg>
				<span class="text-base font-medium dark:text-gray-200">
					{$i18n.t('Recurring Transactions')}
				</span>
			</div>
		</div>
		{#if !collapsed.recurring}
			<div class="px-4 pb-3" transition:slide={{ duration: 200 }}>
				<RecurringTemplates {companyId} />
			</div>
		{/if}
	</div>
</div>
