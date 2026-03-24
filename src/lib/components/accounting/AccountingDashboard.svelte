<script lang="ts">
	import { onMount, onDestroy, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getBalanceSheet,
		getProfitLoss,
		getTransactions,
		getPeriods
	} from '$lib/apis/accounting';
	import { theme } from '$lib/stores';
	import { convertAmount } from '$lib/utils/currency';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Badge from '$lib/components/common/Badge.svelte';
	import AuditTrail from '$lib/components/accounting/AuditTrail.svelte';

	const i18n = getContext('i18n');
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');

	export let companyId: number;

	$: isDark = $theme.includes('dark');

	let loading = true;

	// Stats
	let totalAssets = 0;
	let totalLiabilities = 0;
	let netIncome = 0;
	let draftCount = 0;

	// Chart data
	let monthlyRevenueExpenses: Array<{
		period: string;
		revenue: number;
		expenses: number;
	}> = [];

	// Recent transactions
	let recentTransactions: any[] = [];

	// Periods
	let currentPeriod: any = null;

	// Chart instances
	let revenueExpenseCanvas: HTMLCanvasElement;
	let revenueExpenseChart: any = null;
	let Chart: any = null;

	const loadData = async () => {
		loading = true;
		try {
			const now = new Date();
			const monthStart = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-01`;
			const monthEnd = dayjs(now).format('YYYY-MM-DD');

			const [balanceSheet, profitLoss, transactions, periods] = await Promise.all([
				getBalanceSheet({ company_id: companyId }),
				getProfitLoss({ date_from: monthStart, date_to: monthEnd, company_id: companyId }),
				getTransactions({ limit: 10, company_id: companyId }),
				getPeriods({ company_id: companyId })
			]);

			totalAssets = balanceSheet?.total_assets ?? 0;
			totalLiabilities = balanceSheet?.total_liabilities ?? 0;
			netIncome = profitLoss?.net_income ?? 0;

			// Recent transactions
			const txList = Array.isArray(transactions)
				? transactions
				: transactions?.transactions ?? transactions?.items ?? [];
			recentTransactions = txList.slice(0, 10);
			draftCount = txList.filter((t: any) => t.status === 'draft').length;

			// Build monthly revenue vs expenses data from the last 6 months
			const monthlyMap: Record<string, { revenue: number; expenses: number }> = {};
			for (let i = 5; i >= 0; i--) {
				const d = dayjs().subtract(i, 'month');
				const key = d.format('YYYY-MM');
				monthlyMap[key] = { revenue: 0, expenses: 0 };
			}

			// Fetch P&L for last 6 months
			const sixMonthsAgo = dayjs().subtract(5, 'month').startOf('month').format('YYYY-MM-DD');
			try {
				const fullPL = await getProfitLoss({ date_from: sixMonthsAgo, date_to: monthEnd, company_id: companyId });
				if (fullPL?.revenue) {
					for (const item of fullPL.revenue) {
						const key = item.period ?? item.month;
						if (key && monthlyMap[key]) {
							monthlyMap[key].revenue += parseFloat(String(item.amount ?? item.balance ?? 0));
						}
					}
				}
				if (fullPL?.expenses) {
					for (const item of fullPL.expenses) {
						const key = item.period ?? item.month;
						if (key && monthlyMap[key]) {
							monthlyMap[key].expenses += Math.abs(
								parseFloat(String(item.amount ?? item.balance ?? 0))
							);
						}
					}
				}
			} catch {
				// If detailed P&L fails, use summary values for current month only
				const currentKey = dayjs().format('YYYY-MM');
				if (monthlyMap[currentKey]) {
					monthlyMap[currentKey].revenue = parseFloat(String(profitLoss?.total_revenue ?? 0));
					monthlyMap[currentKey].expenses = Math.abs(
						parseFloat(String(profitLoss?.total_expenses ?? 0))
					);
				}
			}

			monthlyRevenueExpenses = Object.entries(monthlyMap).map(([period, vals]) => ({
				period,
				revenue: vals.revenue,
				expenses: vals.expenses
			}));

			// Periods
			const periodList = Array.isArray(periods) ? periods : periods?.items ?? [];
			currentPeriod =
				periodList.find((p: any) => !p.is_closed) ?? periodList[0] ?? null;
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const createCharts = async () => {
		if (!Chart) {
			const module = await import('chart.js/auto');
			Chart = module.default;
		}

		// Monthly revenue vs expenses bar chart
		if (revenueExpenseCanvas && monthlyRevenueExpenses.length) {
			if (revenueExpenseChart) revenueExpenseChart.destroy();

			revenueExpenseChart = new Chart(revenueExpenseCanvas, {
				type: 'bar',
				data: {
					labels: monthlyRevenueExpenses.map((d) => d.period),
					datasets: [
						{
							label: 'Revenue',
							data: monthlyRevenueExpenses.map((d) => d.revenue),
							backgroundColor: '#6bc87a',
							borderRadius: 4,
							barPercentage: 0.7,
							categoryPercentage: 0.8
						},
						{
							label: 'Expenses',
							data: monthlyRevenueExpenses.map((d) => d.expenses),
							backgroundColor: '#d97c5a',
							borderRadius: 4,
							barPercentage: 0.7,
							categoryPercentage: 0.8
						}
					]
				},
				options: {
					responsive: true,
					maintainAspectRatio: false,
					plugins: {
						legend: {
							display: true,
							position: 'top',
							labels: {
								color: isDark ? '#d1d5db' : '#374151',
								font: { size: 11 },
								boxWidth: 12,
								padding: 8
							}
						},
						tooltip: {
							backgroundColor: 'rgba(17, 24, 39, 0.9)',
							titleColor: '#f3f4f6',
							bodyColor: '#d1d5db',
							borderColor: 'rgba(75, 85, 99, 0.3)',
							borderWidth: 1,
							padding: 8,
							callbacks: {
								label: (ctx: any) => `${ctx.dataset.label}: $${ctx.raw.toLocaleString()}`
							}
						}
					},
					scales: {
						x: {
							grid: { display: false },
							ticks: { color: '#6b7280', font: { size: 10 } },
							border: { display: false }
						},
						y: {
							grid: { color: 'rgba(107, 114, 128, 0.1)', drawTicks: false },
							ticks: {
								color: '#6b7280',
								font: { size: 10 },
								padding: 8,
								callback: (value: number) => `$${value.toLocaleString()}`
							},
							border: { display: false }
						}
					},
					animation: { duration: 400, easing: 'easeOutQuart' }
				}
			});
		}
	};

	$: if (!loading && revenueExpenseCanvas && isDark !== undefined) {
		createCharts();
	}

	onMount(() => {
		loadData();
	});

	onDestroy(() => {
		if (revenueExpenseChart) {
			revenueExpenseChart.destroy();
			revenueExpenseChart = null;
		}
	});

	// ─── Currency conversion ────────────────────────────────────────────────────
	$: nativeCurrency = $companyCurrencyCtx || 'EUR';

	// Try to get native currency from company data when loaded
	const _trySetNativeCurrency = (currency: string) => {
		if (currency) nativeCurrency = currency;
	};

	function cvt(amount: any, date?: string): { display: string; original: string; hasRate: boolean } {
		const num = typeof amount === 'string' ? parseFloat(amount) : (amount ?? 0);
		if (!num || !$displayCurrency || $displayCurrency === nativeCurrency) {
			return { display: '', original: '', hasRate: true };
		}
		const result = convertAmount(num, nativeCurrency, $displayCurrency, ($exchangeRates ?? []), date);
		return {
			display: result.hasRate ? result.converted.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '',
			original: num.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}),
			hasRate: result.hasRate,
		};
	}

	$: isConverting = $displayCurrency && $displayCurrency !== nativeCurrency;

	const formatCurrency = (value: number | string): string => {
		const num = typeof value === 'string' ? parseFloat(value) : value;
		const cur = nativeCurrency || 'EUR';
		return `${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${cur}`;
	};

	const getStatusBadgeType = (status: string): string => {
		switch (status) {
			case 'posted':
				return 'success';
			case 'voided':
				return 'error';
			case 'draft':
			default:
				return 'muted';
		}
	};

	const getTypeBadgeColor = (type: string): string => {
		switch (type) {
			case 'invoice':
				return 'bg-blue-500/20 text-blue-700 dark:text-blue-200';
			case 'payment':
				return 'bg-purple-500/20 text-purple-700 dark:text-purple-200';
			case 'bill':
				return 'bg-orange-500/20 text-orange-700 dark:text-orange-200';
			case 'journal':
			default:
				return 'bg-gray-500/20 text-gray-700 dark:text-gray-200';
		}
	};
</script>

{#if loading}
	<div class="flex justify-center my-10">
		<Spinner className="size-5" />
	</div>
{:else}
	<div class="py-3 space-y-4">
		<!-- Stat Cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Total Assets')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200">
					{#key $displayCurrency}
					{#if isConverting}
						{@const c = cvt(totalAssets)}
						{#if c.hasRate}
							<span>{c.display} <span class="text-sm text-gray-400">{$displayCurrency}</span></span>
							<div class="text-[10px] text-gray-400">{c.original} {nativeCurrency}</div>
						{:else}
							{formatCurrency(totalAssets)}
						{/if}
					{:else}
						{formatCurrency(totalAssets)}
					{/if}
					{/key}
				</div>
			</div>

			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Total Liabilities')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200">
					{#key $displayCurrency}
					{#if isConverting}
						{@const c = cvt(totalLiabilities)}
						{#if c.hasRate}
							<span>{c.display} <span class="text-sm text-gray-400">{$displayCurrency}</span></span>
							<div class="text-[10px] text-gray-400">{c.original} {nativeCurrency}</div>
						{:else}
							{formatCurrency(totalLiabilities)}
						{/if}
					{:else}
						{formatCurrency(totalLiabilities)}
					{/if}
					{/key}
				</div>
			</div>

			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border {netIncome >= 0
					? 'border-green-200/50 dark:border-green-800/30'
					: 'border-red-200/50 dark:border-red-800/30'}"
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Net Income (Current Month)')}
				</div>
				<div
					class="text-2xl font-medium {netIncome >= 0
						? 'text-green-700 dark:text-green-400'
						: 'text-red-700 dark:text-red-400'}"
				>
					{#key $displayCurrency}
					{#if isConverting}
						{@const c = cvt(netIncome)}
						{#if c.hasRate}
							<span>{c.display} <span class="text-sm text-gray-400">{$displayCurrency}</span></span>
							<div class="text-[10px] text-gray-400">{c.original} {nativeCurrency}</div>
						{:else}
							{formatCurrency(netIncome)}
						{/if}
					{:else}
						{formatCurrency(netIncome)}
					{/if}
					{/key}
				</div>
			</div>

			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border {draftCount > 0
					? 'border-yellow-200 dark:border-yellow-800/50'
					: 'border-gray-100/30 dark:border-gray-850/30'}"
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Draft Entries')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200 flex items-center gap-2">
					{draftCount}
					{#if draftCount > 0}
						<Badge type="warning" content={$i18n.t('Pending')} />
					{/if}
				</div>
			</div>
		</div>

		<!-- Quick Actions -->
		<div class="flex gap-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition"
				on:click={() => goto(`/accounting/company/${companyId}/entries`)}
			>
				{$i18n.t('New Journal Entry')}
			</button>
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-500 dark:hover:bg-purple-600 transition"
				on:click={() => goto(`/accounting/company/${companyId}/payments`)}
			>
				{$i18n.t('Record Payment')}
			</button>
		</div>

		<!-- Revenue vs Expenses Chart -->
		<div class="grid grid-cols-1 gap-3">
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-sm font-medium dark:text-gray-200 mb-3">
					{$i18n.t('Monthly Revenue vs Expenses')}
				</div>
				{#if monthlyRevenueExpenses.length === 0}
					<div class="flex items-center justify-center h-48 text-gray-500 text-sm">
						{$i18n.t('No data')}
					</div>
				{:else}
					<div class="h-48">
						<canvas bind:this={revenueExpenseCanvas}></canvas>
					</div>
				{/if}
			</div>
		</div>

		<!-- Accounting Period Summary -->
		{#if currentPeriod}
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-sm font-medium dark:text-gray-200 mb-2">
					{$i18n.t('Current Accounting Period')}
				</div>
				<div class="flex items-center gap-3 text-sm text-gray-600 dark:text-gray-400">
					<span class="font-medium dark:text-gray-200">
						{currentPeriod.name ?? currentPeriod.label ?? `${currentPeriod.start_date} - ${currentPeriod.end_date}`}
					</span>
					<Badge
						type={!currentPeriod.is_closed ? 'success' : 'muted'}
						content={$i18n.t(!currentPeriod.is_closed ? 'Open' : 'Closed')}
					/>
					{#if currentPeriod.start_date && currentPeriod.end_date}
						<span class="text-xs text-gray-400 dark:text-gray-500">
							{dayjs(currentPeriod.start_date).format('YYYY-MM-DD')} &mdash; {dayjs(
								currentPeriod.end_date
							).format('YYYY-MM-DD')}
						</span>
					{/if}
				</div>
			</div>
		{/if}

		<!-- Recent Transactions -->
		{#if recentTransactions.length > 0}
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="flex justify-between items-center mb-3">
					<div class="text-sm font-medium dark:text-gray-200">
						{$i18n.t('Recent Transactions')}
					</div>
					<a
						href="/accounting/company/{companyId}/entries"
						class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition"
					>
						{$i18n.t('View All')} &rarr;
					</a>
				</div>

				<table
					class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto"
				>
					<thead
						class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200"
					>
						<tr
							class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30"
						>
							<th class="px-2.5 py-2">{$i18n.t('Date')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Type')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Description')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Total')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Status')}</th>
						</tr>
					</thead>
					<tbody>
						{#each recentTransactions as transaction}
							<tr
								class="bg-white dark:bg-gray-900 dark:border-gray-850 text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
								on:click={() => goto(`/accounting/company/${companyId}/entries?id=${transaction.id}`)}
							>
								<td class="px-3 py-1.5">
									{transaction.transaction_date
										? dayjs(transaction.transaction_date).format('YYYY-MM-DD')
										: '-'}
								</td>
								<td class="px-3 py-1.5">
									<span
										class="text-xs font-medium {getTypeBadgeColor(
											transaction.transaction_type
										)} w-fit px-[5px] rounded-lg uppercase line-clamp-1"
									>
										{transaction.transaction_type ?? '-'}
									</span>
								</td>
								<td class="px-3 py-1.5 max-w-[200px] truncate">
									{transaction.description ?? transaction.memo ?? '-'}
								</td>
								<td class="px-3 py-1.5 text-right">
									{#key $displayCurrency}
									{#if isConverting}
										{@const c = cvt(transaction.total ?? transaction.amount ?? 0, transaction.transaction_date)}
										{#if c.hasRate}
											<span class="font-medium">{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
											<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
										{:else}
											<span>{c.original} {nativeCurrency}</span>
											<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
										{/if}
									{:else}
										{formatCurrency(transaction.total ?? transaction.amount ?? 0)}
									{/if}
									{/key}
								</td>
								<td class="px-3 py-1.5">
									<Badge
										type={getStatusBadgeType(transaction.status)}
										content={transaction.status ?? 'draft'}
									/>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<!-- Activity / Audit Trail -->
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
		>
			<div class="text-sm font-medium dark:text-gray-200 mb-3">
				{$i18n.t('Recent Activity')}
			</div>
			<AuditTrail {companyId} limit={10} />
		</div>
	</div>
{/if}
