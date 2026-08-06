<script lang="ts">
	import { page } from '$app/stores';
	import ExpenseSheetsList from '$lib/components/accounting/expenses/ExpenseSheetsList.svelte';
	import ExpenseSubmissions from '$lib/components/accounting/expenses/ExpenseSubmissions.svelte';
	import ExpenseDashboard from '$lib/components/accounting/expenses/ExpenseDashboard.svelte';
	import { getPendingCount } from '$lib/apis/expenses';

	$: companyId = parseInt($page.params.id ?? '0', 10);

	// Overview | employee-submitted items (portal) | accountant-assembled sheets.
	let tab: 'overview' | 'submissions' | 'sheets' = 'overview';
	let pending = 0;

	const refreshPending = async (id: number) => {
		try {
			pending = (await getPendingCount(id)).count;
		} catch {
			pending = 0;
		}
	};
	$: if (companyId) refreshPending(companyId);
</script>

<div class="flex flex-col gap-4">
	<div class="flex gap-1 border-b border-gray-100 dark:border-gray-850">
		<button
			class="border-b-2 px-3 py-2 text-sm font-medium transition {tab === 'overview'
				? 'border-gray-900 text-gray-900 dark:border-white dark:text-white'
				: 'border-transparent text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'}"
			on:click={() => (tab = 'overview')}
		>
			Overview
		</button>
		<button
			class="border-b-2 px-3 py-2 text-sm font-medium transition {tab === 'submissions'
				? 'border-gray-900 text-gray-900 dark:border-white dark:text-white'
				: 'border-transparent text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'}"
			on:click={() => (tab = 'submissions')}
		>
			Employee submissions
			{#if pending > 0}
				<span
					class="ml-1.5 inline-flex min-w-[1.1rem] items-center justify-center rounded-full bg-blue-600 px-1.5 py-0.5 text-[10px] font-semibold text-white"
					>{pending > 99 ? '99+' : pending}</span
				>
			{/if}
		</button>
		<button
			class="border-b-2 px-3 py-2 text-sm font-medium transition {tab === 'sheets'
				? 'border-gray-900 text-gray-900 dark:border-white dark:text-white'
				: 'border-transparent text-gray-500 hover:text-gray-800 dark:hover:text-gray-200'}"
			on:click={() => (tab = 'sheets')}
		>
			Expense sheets
		</button>
	</div>

	{#if tab === 'overview'}
		<ExpenseDashboard {companyId} />
	{:else if tab === 'submissions'}
		<ExpenseSubmissions {companyId} />
	{:else}
		<ExpenseSheetsList {companyId} />
	{/if}
</div>
