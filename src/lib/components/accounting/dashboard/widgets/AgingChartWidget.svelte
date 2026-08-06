<script lang="ts">
	// AR or AP aging buckets (donut), from /reports/ar-aging | /reports/ap-aging.
	// Buckets carry an ordinal severity ramp (recent → overdue).
	import { onMount, getContext } from 'svelte';
	import { theme } from '$lib/stores';
	import { getARAging, getAPAging } from '$lib/apis/accounting';
	import EChart from '../EChart.svelte';
	import { chartTheme, tooltipStyle } from '../palette';
	import { fmtNumber } from '../format';

	export let companyId: number;
	export let options: { kind?: 'ar' | 'ap' } = {};

	const i18n: any = getContext('i18n');
	$: isDark = $theme.includes('dark');
	$: t = chartTheme(isDark);

	let summary: any = null;
	let loading = true;

	$: kind = options?.kind ?? 'ar';

	onMount(load);
	async function load() {
		loading = true;
		try {
			const res = kind === 'ap'
				? await getAPAging({ company_id: companyId })
				: await getARAging({ company_id: companyId });
			summary = res?.summary ?? null;
		} catch {
			summary = null;
		}
		loading = false;
	}

	$: buckets = summary
		? [
				{ name: 'Current', value: summary.current ?? 0, color: t.series[0] },
				{ name: '31–60d', value: summary.days_31_60 ?? 0, color: t.warning },
				{ name: '61–90d', value: summary.days_61_90 ?? 0, color: t.serious ?? t.expenses },
				{ name: '90d+', value: summary.over_90 ?? 0, color: t.critical }
		  ].filter((b) => b.value > 0)
		: [];

	$: total = buckets.reduce((s, b) => s + b.value, 0);

	$: option = {
		tooltip: {
			trigger: 'item',
			...tooltipStyle(t),
			valueFormatter: (v: number) => fmtNumber(v)
		},
		legend: {
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
				radius: ['50%', '74%'],
				center: ['34%', '50%'],
				itemStyle: { borderColor: t.surface, borderWidth: 2 },
				label: {
					show: true,
					position: 'center',
					formatter: () => (total ? fmtNumber(total) : ''),
					color: t.primary,
					fontSize: 13,
					fontWeight: 600
				},
				emphasis: { label: { show: true } },
				data: buckets.map((b) => ({ name: b.name, value: b.value, itemStyle: { color: b.color } }))
			}
		]
	};
</script>

{#if loading}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Loading…') : 'Loading…'}</div>
{:else if buckets.length === 0}
	<div class="h-full flex items-center justify-center text-xs text-gray-400">{i18n?.t ? i18n.t('Nothing outstanding') : 'Nothing outstanding'}</div>
{:else}
	<EChart {option} />
{/if}
