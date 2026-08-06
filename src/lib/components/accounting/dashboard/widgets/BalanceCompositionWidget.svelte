<script lang="ts">
	// Balance-sheet composition: Assets vs Liabilities + Equity (stacked columns).
	import { onMount, getContext } from 'svelte';
	import { theme } from '$lib/stores';
	import { getBalanceSheet } from '$lib/apis/accounting';
	import EChart from '../EChart.svelte';
	import { chartTheme, tooltipStyle } from '../palette';
	import { fmtNumber, fmtCompact } from '../format';

	export let companyId: number;

	const i18n: any = getContext('i18n');
	$: isDark = $theme.includes('dark');
	$: t = chartTheme(isDark);

	let bs: any = null;
	let currency = '';
	let loading = true;

	onMount(load);
	async function load() {
		loading = true;
		try {
			bs = await getBalanceSheet({ company_id: companyId });
			currency = bs?.currency ?? '';
		} catch {
			bs = null;
		}
		loading = false;
	}

	$: assets = bs?.total_assets ?? 0;
	$: liabilities = bs?.total_liabilities ?? 0;
	$: equity = bs?.total_equity ?? 0;
	$: hasData = assets || liabilities || equity;

	$: option = {
		grid: { left: 6, right: 12, top: 28, bottom: 4, containLabel: true },
		tooltip: {
			trigger: 'axis',
			axisPointer: { type: 'shadow' },
			...tooltipStyle(t),
			valueFormatter: (v: number) => (v ? `${fmtNumber(v)} ${currency}` : '')
		},
		legend: { top: 0, textStyle: { color: t.secondary, fontSize: 11 }, itemWidth: 10, itemHeight: 10 },
		xAxis: {
			type: 'category',
			data: ['Assets', 'Liab. + Equity'],
			axisLine: { lineStyle: { color: t.axis } },
			axisTick: { show: false },
			axisLabel: { color: t.muted, fontSize: 10 }
		},
		yAxis: {
			type: 'value',
			splitLine: { lineStyle: { color: t.grid } },
			axisLabel: { color: t.muted, fontSize: 10, formatter: (v: number) => fmtCompact(v) }
		},
		series: [
			{ name: 'Assets', type: 'bar', stack: 'total', barMaxWidth: 60, data: [+assets, 0], itemStyle: { color: t.series[0] } },
			{ name: 'Liabilities', type: 'bar', stack: 'total', barMaxWidth: 60, data: [0, +liabilities], itemStyle: { color: t.series[7] } },
			{ name: 'Equity', type: 'bar', stack: 'total', barMaxWidth: 60, data: [0, +equity], itemStyle: { color: t.series[4] } }
		]
	};
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if !hasData}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No data') : 'No data'}</div>
{:else}
	<EChart {option} />
{/if}
