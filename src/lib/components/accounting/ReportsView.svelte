<script lang="ts">
	import { onMount, getContext } from 'svelte';

	import GeneralLedger from './GeneralLedger.svelte';
	import TrialBalance from './TrialBalance.svelte';
	import ProfitLoss from './ProfitLoss.svelte';
	import BalanceSheet from './BalanceSheet.svelte';
	import CashFlow from './CashFlow.svelte';
	import AgingReport from './AgingReport.svelte';
	import StatutoryStatements from './StatutoryStatements.svelte';
	import { getStatementLayout, type StatementLayout } from '$lib/apis/accounting';

	const i18n = getContext('i18n');

	export let companyId: number;

	type ReportTab =
		| 'statements'
		| 'general-ledger'
		| 'trial-balance'
		| 'profit-loss'
		| 'balance-sheet'
		| 'cash-flow'
		| 'ap-aging'
		| 'ar-aging';

	let activeReport: ReportTab = 'general-ledger';

	// A company whose country has a prescribed filing layout (today: China,
	// 小企业会计准则) gets an extra tab holding those statements, and opens on it —
	// that is the form its accountant actually files.
	let statementLayout: StatementLayout | null = null;
	$: hasStatements = !!statementLayout?.layout;

	onMount(async () => {
		try {
			statementLayout = await getStatementLayout(companyId);
			if (statementLayout?.layout) activeReport = 'statements';
		} catch (err) {
			console.error('Failed to resolve statement layout:', err);
		}
	});

	const baseTabs: Array<{ id: ReportTab; label: string }> = [
		{ id: 'general-ledger', label: 'General Ledger' },
		{ id: 'trial-balance', label: 'Trial Balance' },
		{ id: 'profit-loss', label: 'P&L' },
		{ id: 'balance-sheet', label: 'Balance Sheet' },
		{ id: 'cash-flow', label: 'Cash Flow' },
		{ id: 'ap-aging', label: 'AP Aging' },
		{ id: 'ar-aging', label: 'AR Aging' }
	];

	$: tabs = hasStatements
		? [{ id: 'statements' as ReportTab, label: 'Statutory Statements' }, ...baseTabs]
		: baseTabs;
</script>

<div class="py-3 space-y-4">
	<!-- Sub-tab bar -->
	<div class="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 w-fit">
		{#each tabs as tab}
			<button
				class="px-4 py-2 text-sm font-medium rounded-md transition
					{activeReport === tab.id
					? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
					: 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'}"
				on:click={() => (activeReport = tab.id)}
			>
				{$i18n.t(tab.label)}
			</button>
		{/each}
	</div>

	<!-- Report content -->
	{#if activeReport === 'statements'}
		<StatutoryStatements {companyId} layoutLabel={statementLayout?.label ?? null} layoutKey={statementLayout?.layout ?? null} presentation={statementLayout?.presentation ?? null} />
	{:else if activeReport === 'general-ledger'}
		<GeneralLedger {companyId} />
	{:else if activeReport === 'trial-balance'}
		<TrialBalance {companyId} />
	{:else if activeReport === 'profit-loss'}
		<ProfitLoss {companyId} />
	{:else if activeReport === 'balance-sheet'}
		<BalanceSheet {companyId} />
	{:else if activeReport === 'cash-flow'}
		<CashFlow {companyId} />
	{:else if activeReport === 'ap-aging'}
		<AgingReport {companyId} reportType="ap" />
	{:else if activeReport === 'ar-aging'}
		<AgingReport {companyId} reportType="ar" />
	{/if}
</div>
