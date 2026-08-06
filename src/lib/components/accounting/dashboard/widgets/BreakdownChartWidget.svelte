<script lang="ts">
	// Expense or revenue breakdown by account (donut), from /reports/profit-loss.
	// Uses leaf accounts (is_parent === false) to avoid double-counting rollups;
	// top 8 + "Other".
	import { onMount, getContext } from 'svelte';
	import { theme } from '$lib/stores';
	import { getProfitLoss } from '$lib/apis/accounting';
	import EChart from '../EChart.svelte';
	import { chartTheme, tooltipStyle } from '../palette';
	import { fmtNumber } from '../format';

	export let companyId: number;
	export let options: { kind?: 'expenses' | 'revenue'; range?: 'ytd' | 'mtd' } = {};

	const i18n: any = getContext('i18n');
	$: isDark = $theme.includes('dark');
	$: t = chartTheme(isDark);

	let rows: { name: string; value: number }[] = [];
	let currency = '';
	let loading = true;

	$: kind = options?.kind ?? 'expenses';
	$: range = options?.range ?? 'ytd';

	onMount(load);
	async function load() {
		loading = true;
		try {
			const now = new Date();
			const y = now.getFullYear();
			const m = String(now.getMonth() + 1).padStart(2, '0');
			const d = String(now.getDate()).padStart(2, '0');
			const date_to = `${y}-${m}-${d}`;
			const date_from = range === 'mtd' ? `${y}-${m}-01` : `${y}-01-01`;
			const pl = await getProfitLoss({ company_id: companyId, date_from, date_to });
			currency = pl?.currency ?? '';
			const items = (kind === 'revenue' ? pl?.revenue : pl?.expenses) ?? [];
			const leaves = items
				.filter((a: any) => !a.is_parent)
				.map((a: any) => ({ name: a.account_name || a.account_code, value: Math.abs(parseFloat(String(a.amount ?? 0))) }))
				.filter((a: any) => a.value > 0)
				.sort((a: any, b: any) => b.value - a.value);
			if (leaves.length > 8) {
				const top = leaves.slice(0, 8);
				const other = leaves.slice(8).reduce((s: number, a: any) => s + a.value, 0);
				top.push({ name: 'Other', value: other });
				rows = top;
			} else {
				rows = leaves;
			}
		} catch {
			rows = [];
		}
		loading = false;
	}

	$: option = {
		tooltip: {
			trigger: 'item',
			...tooltipStyle(t),
			valueFormatter: (v: number) => `${fmtNumber(v)} ${currency}`
		},
		legend: {
			type: 'scroll',
			orient: 'vertical',
			right: 4,
			top: 'center',
			textStyle: { color: t.secondary, fontSize: 11 },
			itemWidth: 10,
			itemHeight: 10
		},
		series: [
			{
				type: 'pie',
				radius: ['45%', '72%'],
				center: ['32%', '50%'],
				avoidLabelOverlap: true,
				itemStyle: { borderColor: t.surface, borderWidth: 2 },
				label: { show: false },
				data: rows.map((r, i) => ({
					...r,
					itemStyle: { color: t.series[i % t.series.length] }
				}))
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
