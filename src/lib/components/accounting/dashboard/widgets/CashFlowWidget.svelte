<script lang="ts">
	// Cash-flow waterfall: Opening → Operating → Investing → Financing → Closing.
	// Backed by /reports/cash-flow (indirect method), YTD by default.
	import { onMount, getContext } from 'svelte';
	import { theme } from '$lib/stores';
	import { getCashFlow } from '$lib/apis/accounting';
	import EChart from '../EChart.svelte';
	import { chartTheme, tooltipStyle } from '../palette';
	import { fmtNumber, fmtCompact } from '../format';

	export let companyId: number;
	export let options: { range?: 'ytd' | 'mtd' } = {};

	const i18n: any = getContext('i18n');
	$: isDark = $theme.includes('dark');
	$: t = chartTheme(isDark);

	let cf: any = null;
	let loading = true;
	let error = false;
	$: range = options?.range ?? 'ytd';

	onMount(load);
	async function load() {
		loading = true;
		error = false;
		try {
			const now = new Date();
			const y = now.getFullYear();
			const m = String(now.getMonth() + 1).padStart(2, '0');
			const d = String(now.getDate()).padStart(2, '0');
			const date_to = `${y}-${m}-${d}`;
			const date_from = range === 'mtd' ? `${y}-${m}-01` : `${y}-01-01`;
			cf = await getCashFlow({ company_id: companyId, date_from, date_to });
		} catch {
			error = true;
		}
		loading = false;
	}

	const cats = ['Opening', 'Operating', 'Investing', 'Financing', 'Closing'];

	$: built = buildWaterfall(cf);
	function buildWaterfall(cf: any) {
		if (!cf) return { placeholder: [], bars: [], deltas: [], hasData: false };
		const opening = +(cf.opening_cash ?? 0);
		const op = +(cf.cash_from_operations ?? 0);
		const inv = +(cf.cash_from_investing ?? 0);
		const fin = +(cf.cash_from_financing ?? 0);
		const closing = +(cf.closing_cash ?? opening + op + inv + fin);

		const placeholder: number[] = [];
		const bars: any[] = [];
		const deltas: number[] = [opening, op, inv, fin, closing];

		const totalColor = t.series[0];
		const pushDelta = (base: number, delta: number) => {
			const b = delta >= 0 ? base : base + delta;
			placeholder.push(b);
			bars.push({ value: Math.abs(delta), itemStyle: { color: delta >= 0 ? t.good : t.critical, borderRadius: 3 } });
		};

		// Opening (total from 0)
		placeholder.push(0);
		bars.push({ value: opening, itemStyle: { color: totalColor, borderRadius: 3 } });
		let running = opening;
		pushDelta(running, op); running += op;
		pushDelta(running, inv); running += inv;
		pushDelta(running, fin); running += fin;
		// Closing (total from 0)
		placeholder.push(0);
		bars.push({ value: closing, itemStyle: { color: totalColor, borderRadius: 3 } });

		const hasData = [opening, op, inv, fin, closing].some((v) => v !== 0);
		return { placeholder, bars, deltas, hasData };
	}

	$: option = {
		grid: { left: 6, right: 12, top: 12, bottom: 4, containLabel: true },
		tooltip: {
			trigger: 'axis',
			axisPointer: { type: 'shadow' },
			...tooltipStyle(t),
			formatter: (params: any[]) => {
				const i = params[0].dataIndex;
				const val = built.deltas[i] ?? 0;
				const sign = i === 0 || i === 4 ? '' : val >= 0 ? '+' : '−';
				return `${cats[i]}<br/><b>${sign}${fmtNumber(Math.abs(val))}</b>`;
			}
		},
		xAxis: {
			type: 'category',
			data: cats,
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
			{ type: 'bar', stack: 'wf', itemStyle: { color: 'transparent' }, emphasis: { itemStyle: { color: 'transparent' } }, data: built.placeholder, silent: true },
			{ type: 'bar', stack: 'wf', barMaxWidth: 34, data: built.bars }
		]
	};
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if error || !built.hasData}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('No data') : 'No data'}</div>
{:else}
	<EChart {option} />
{/if}
