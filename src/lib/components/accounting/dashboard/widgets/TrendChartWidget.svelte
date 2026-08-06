<script lang="ts">
	// Revenue vs Expenses (grouped bars) or Net Income (line) over N months.
	// Backed by the /reports/monthly-series endpoint.
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { theme } from '$lib/stores';
	import { getMonthlySeries, type MonthlySeriesPoint } from '$lib/apis/accounting';
	import EChart from '../EChart.svelte';
	import { chartTheme, tooltipStyle } from '../palette';
	import { fmtCompact, fmtNumber } from '../format';

	export let companyId: number;
	export let options: { mode?: 'revenue_expenses' | 'net_income'; months?: number } = {};

	const i18n: any = getContext('i18n');
	$: isDark = $theme.includes('dark');
	$: t = chartTheme(isDark);

	let points: MonthlySeriesPoint[] = [];
	let currency = '';
	let loading = true;
	let error = false;

	$: mode = options?.mode ?? 'revenue_expenses';
	$: months = options?.months ?? 12;

	onMount(load);
	async function load() {
		loading = true;
		error = false;
		try {
			const res = await getMonthlySeries({ company_id: companyId, months });
			points = res.points ?? [];
			currency = res.currency ?? '';
		} catch {
			error = true;
		}
		loading = false;
	}

	$: labels = points.map((p) => p.period);
	$: hasData = points.some((p) => p.revenue || p.expenses || p.net_income);

	$: option = buildOption(t, mode, points, labels, currency);
	function buildOption(t: any, mode: string, pts: MonthlySeriesPoint[], labels: string[], cur: string) {
		const base = {
			grid: { left: 6, right: 12, top: 28, bottom: 4, containLabel: true },
			tooltip: {
				trigger: 'axis',
				...tooltipStyle(t),
				valueFormatter: (v: number) => `${fmtNumber(v)} ${cur}`
			},
			xAxis: {
				type: 'category',
				data: labels,
				axisLine: { lineStyle: { color: t.axis } },
				axisTick: { show: false },
				axisLabel: { color: t.muted, fontSize: 10 }
			},
			yAxis: {
				type: 'value',
				splitLine: { lineStyle: { color: t.grid } },
				axisLabel: { color: t.muted, fontSize: 10, formatter: (v: number) => fmtCompact(v) }
			}
		};
		if (mode === 'net_income') {
			return {
				...base,
				series: [
					{
						name: 'Net Income',
						type: 'line',
						smooth: true,
						symbolSize: 7,
						data: pts.map((p) => +p.net_income.toFixed(2)),
						lineStyle: { color: t.net, width: 2 },
						itemStyle: { color: t.net },
						areaStyle: { color: t.net, opacity: t.isDark ? 0.12 : 0.08 }
					}
				]
			};
		}
		return {
			...base,
			legend: {
				data: ['Revenue', 'Expenses'],
				top: 0,
				right: 0,
				textStyle: { color: t.secondary, fontSize: 11 },
				itemWidth: 10,
				itemHeight: 10
			},
			series: [
				{
					name: 'Revenue',
					type: 'bar',
					data: pts.map((p) => +p.revenue.toFixed(2)),
					itemStyle: { color: t.revenue, borderRadius: [3, 3, 0, 0] },
					barMaxWidth: 22
				},
				{
					name: 'Expenses',
					type: 'bar',
					data: pts.map((p) => +p.expenses.toFixed(2)),
					itemStyle: { color: t.expenses, borderRadius: [3, 3, 0, 0] },
					barMaxWidth: 22
				}
			]
		};
	}
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if error}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No data') : 'No data'}</div>
{:else if !hasData}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No data') : 'No data'}</div>
{:else}
	<EChart {option} />
{/if}
