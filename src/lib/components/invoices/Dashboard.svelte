<script lang="ts">
	import { onMount, onDestroy, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import { getInvoices, getVendors, getSpendingSummary, getNeedsReview } from '$lib/apis/invoices';
	import { theme } from '$lib/stores';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Badge from '$lib/components/common/Badge.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	$: isDark = $theme.includes('dark');

	let loading = true;

	// Stats
	let totalInvoices = 0;
	let totalSpending = 0;
	let needsReviewCount = 0;
	let vendorCount = 0;
	let currency = 'USD';

	// Chart data
	let monthlyData: Array<{ period: string; total_amount: number; invoice_count: number }> = [];
	let vendorData: Array<{ period: string; total_amount: number; invoice_count: number }> = [];

	// Needs review list
	let reviewInvoices: any[] = [];

	// Chart instances
	let monthlyCanvas: HTMLCanvasElement;
	let vendorCanvas: HTMLCanvasElement;
	let monthlyChart: any = null;
	let vendorChart: any = null;
	let Chart: any = null;

	const loadData = async () => {
		loading = true;
		try {
			const [invoicesRes, vendors, monthly, byVendor, review] = await Promise.all([
				getInvoices(localStorage.token, { limit: 1, offset: 0 }),
				getVendors(localStorage.token),
				getSpendingSummary(localStorage.token, { period: 'monthly' }),
				getSpendingSummary(localStorage.token, { period: 'by_vendor' }),
				getNeedsReview(localStorage.token)
			]);

			vendorCount = vendors?.length ?? 0;
			// Use vendor invoice_count sum for accurate total (works with both old and new API)
			totalInvoices = (vendors ?? []).reduce(
				(sum: number, v: any) => sum + (v.invoice_count || 0),
				0
			);
			needsReviewCount = review?.length ?? 0;
			reviewInvoices = (review ?? []).slice(0, 10);

			totalSpending = (vendors ?? []).reduce(
				(sum: number, v: any) => sum + parseFloat(v.total_amount || '0'),
				0
			);

			monthlyData = monthly ?? [];
			vendorData = (byVendor ?? []).slice(0, 10);
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

		// Monthly spending bar chart
		if (monthlyCanvas && monthlyData.length) {
			if (monthlyChart) monthlyChart.destroy();

			monthlyChart = new Chart(monthlyCanvas, {
				type: 'bar',
				data: {
					labels: monthlyData.map((d) => d.period),
					datasets: [
						{
							label: 'Spending',
							data: monthlyData.map((d) => parseFloat(String(d.total_amount))),
							backgroundColor: '#5ba3c8',
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
						legend: { display: false },
						tooltip: {
							backgroundColor: 'rgba(17, 24, 39, 0.9)',
							titleColor: '#f3f4f6',
							bodyColor: '#d1d5db',
							borderColor: 'rgba(75, 85, 99, 0.3)',
							borderWidth: 1,
							padding: 8,
							callbacks: {
								label: (ctx: any) => `$${ctx.raw.toLocaleString()}`
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

		// Vendor spending doughnut chart
		if (vendorCanvas && vendorData.length) {
			if (vendorChart) vendorChart.destroy();

			const colors = [
				'#5ba3c8',
				'#d97c5a',
				'#6bc87a',
				'#c85ba3',
				'#c8b85b',
				'#5b6bc8',
				'#c85b5b',
				'#5bc8b8',
				'#a35bc8',
				'#c8a35b'
			];

			const legendTextColor = isDark ? '#d1d5db' : '#374151';

			// Truncate labels for display, keep originals for tooltips
			const truncatedLabels = vendorData.map((d) =>
				d.period.length > 20 ? d.period.substring(0, 20) + '...' : d.period
			);

			vendorChart = new Chart(vendorCanvas, {
				type: 'doughnut',
				data: {
					labels: truncatedLabels,
					datasets: [
						{
							data: vendorData.map((d) => parseFloat(String(d.total_amount))),
							backgroundColor: colors.slice(0, vendorData.length),
							borderWidth: 0
						}
					]
				},
				options: {
					responsive: true,
					maintainAspectRatio: false,
					cutout: '60%',
					plugins: {
						legend: {
							position: 'right',
							labels: {
								color: legendTextColor,
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
								label: (ctx: any) => {
								const fullName = vendorData[ctx.dataIndex]?.period ?? ctx.label;
								return `${fullName}: $${ctx.raw.toLocaleString()}`;
							}
							}
						}
					},
					animation: { duration: 400, easing: 'easeOutQuart' }
				}
			});
		}
	};

	$: if (!loading && monthlyCanvas && vendorCanvas && isDark !== undefined) {
		createCharts();
	}

	onMount(() => {
		loadData();
	});

	onDestroy(() => {
		if (monthlyChart) {
			monthlyChart.destroy();
			monthlyChart = null;
		}
		if (vendorChart) {
			vendorChart.destroy();
			vendorChart = null;
		}
	});
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
					{$i18n.t('Total Invoices')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200">
					{totalInvoices.toLocaleString()}
				</div>
			</div>

			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Total Spending')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200">
					${totalSpending.toLocaleString(undefined, {
						minimumFractionDigits: 2,
						maximumFractionDigits: 2
					})}
				</div>
			</div>

			<button
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30 text-left cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
				on:click={() => goto('/invoices/data?needs_review=true')}
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Needs Review')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200 flex items-center gap-2">
					{needsReviewCount}
					{#if needsReviewCount > 0}
						<Badge type="warning" content={$i18n.t('Attention')} />
					{/if}
				</div>
			</button>

			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-xs text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Vendors')}
				</div>
				<div class="text-2xl font-medium dark:text-gray-200">
					{vendorCount}
				</div>
			</div>
		</div>

		<!-- Charts -->
		<div class="grid grid-cols-1 lg:grid-cols-2 gap-3">
			<!-- Monthly Spending -->
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-sm font-medium dark:text-gray-200 mb-3">
					{$i18n.t('Monthly Spending')}
				</div>
				{#if monthlyData.length === 0}
					<div class="flex items-center justify-center h-48 text-gray-500 text-sm">
						{$i18n.t('No data')}
					</div>
				{:else}
					<div class="h-48">
						<canvas bind:this={monthlyCanvas}></canvas>
					</div>
				{/if}
			</div>

			<!-- Spending by Vendor -->
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="text-sm font-medium dark:text-gray-200 mb-3">
					{$i18n.t('Spending by Vendor')}
				</div>
				{#if vendorData.length === 0}
					<div class="flex items-center justify-center h-48 text-gray-500 text-sm">
						{$i18n.t('No data')}
					</div>
				{:else}
					<div class="h-48">
						<canvas bind:this={vendorCanvas}></canvas>
					</div>
				{/if}
			</div>
		</div>

		<!-- Needs Review Table -->
		{#if reviewInvoices.length > 0}
			<div
				class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30"
			>
				<div class="flex justify-between items-center mb-3">
					<div class="text-sm font-medium dark:text-gray-200">
						{$i18n.t('Invoices Needing Review')}
					</div>
					<a
						href="/invoices/data"
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
							<th class="px-2.5 py-2">{$i18n.t('Vendor')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Invoice #')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Date')}</th>
							<th class="px-2.5 py-2 text-right">{$i18n.t('Total')}</th>
							<th class="px-2.5 py-2">{$i18n.t('Confidence')}</th>
							<th class="px-2.5 py-2">{$i18n.t('K4mi')}</th>
						</tr>
					</thead>
					<tbody>
						{#each reviewInvoices as invoice}
							<tr
								class="bg-white dark:bg-gray-900 dark:border-gray-850 text-xs cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
								on:click={() => goto(`/invoices/data?vendor=${encodeURIComponent(invoice.vendor_name)}`)}
							>
								<td class="px-3 py-1.5">{invoice.vendor_name}</td>
								<td class="px-3 py-1.5">{invoice.invoice_number ?? '-'}</td>
								<td class="px-3 py-1.5">
									{invoice.invoice_date
										? dayjs(invoice.invoice_date).format('YYYY-MM-DD')
										: '-'}
								</td>
								<td class="px-3 py-1.5 text-right">
									{invoice.currency ?? 'USD'}{parseFloat(
										invoice.total_amount
									).toLocaleString(undefined, {
										minimumFractionDigits: 2
									})}
								</td>
								<td class="px-3 py-1.5">
									{#if invoice.confidence_score !== null}
										<Badge
											type={parseFloat(invoice.confidence_score) >= 0.7
												? 'success'
												: parseFloat(invoice.confidence_score) >= 0.4
													? 'warning'
													: 'error'}
											content={`${(parseFloat(invoice.confidence_score) * 100).toFixed(0)}%`}
										/>
									{:else}
										<Badge type="muted" content="N/A" />
									{/if}
								</td>
								<td class="px-3 py-1.5">
									<!-- svelte-ignore a11y-click-events-have-key-events -->
									<!-- svelte-ignore a11y-no-static-element-interactions -->
									<a
										href="http://localhost:8000/documents/{invoice.k4mi_document_id}/details"
										target="_blank"
										rel="noopener noreferrer"
										class="text-blue-500 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300 underline"
										on:click|stopPropagation
									>
										#{invoice.k4mi_document_id}
									</a>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	</div>
{/if}
