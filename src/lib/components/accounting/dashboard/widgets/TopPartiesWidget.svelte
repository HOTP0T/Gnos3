<script lang="ts">
	// Top vendors (by billed) or customers (by invoiced) — horizontal bars.
	// Backed by /reports/top-parties.
	import { onMount, getContext } from 'svelte';
	import { theme } from '$lib/stores';
	import { getTopParties, type TopPartyRow } from '$lib/apis/accounting';
	import EChart from '../EChart.svelte';
	import { chartTheme, tooltipStyle } from '../palette';
	import { fmtNumber, fmtCompact } from '../format';

	export let companyId: number;
	export let options: { direction?: 'vendors' | 'customers'; limit?: number } = {};

	const i18n: any = getContext('i18n');
	$: isDark = $theme.includes('dark');
	$: t = chartTheme(isDark);

	let rows: TopPartyRow[] = [];
	let currency = '';
	let loading = true;

	$: direction = options?.direction ?? 'vendors';
	$: limit = options?.limit ?? 8;

	onMount(load);
	async function load() {
		loading = true;
		try {
			const res = await getTopParties({ company_id: companyId, direction, limit });
			rows = res.rows ?? [];
			currency = res.currency ?? '';
		} catch {
			rows = [];
		}
		loading = false;
	}

	// echarts category axis draws bottom→top, so reverse for descending top-down.
	$: ordered = [...rows].reverse();
	$: barColor = direction === 'vendors' ? t.series[7] : t.series[1];

	$: option = {
		grid: { left: 6, right: 16, top: 8, bottom: 4, containLabel: true },
		tooltip: {
			trigger: 'axis',
			axisPointer: { type: 'shadow' },
			...tooltipStyle(t),
			valueFormatter: (v: number) => `${fmtNumber(v)} ${currency}`
		},
		xAxis: {
			type: 'value',
			splitLine: { lineStyle: { color: t.grid } },
			axisLabel: { color: t.muted, fontSize: 10, formatter: (v: number) => fmtCompact(v) }
		},
		yAxis: {
			type: 'category',
			data: ordered.map((r) => r.name),
			axisLine: { lineStyle: { color: t.axis } },
			axisTick: { show: false },
			axisLabel: {
				color: t.secondary,
				fontSize: 10,
				width: 90,
				overflow: 'truncate'
			}
		},
		series: [
			{
				type: 'bar',
				data: ordered.map((r) => +r.total_amount.toFixed(2)),
				itemStyle: { color: barColor, borderRadius: [0, 3, 3, 0] },
				barMaxWidth: 16
			}
		]
	};
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if rows.length === 0}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No data') : 'No data'}</div>
{:else}
	<EChart {option} />
{/if}
