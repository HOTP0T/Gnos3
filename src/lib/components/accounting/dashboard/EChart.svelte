<script lang="ts">
	// Thin, theme-aware ECharts wrapper. Lazy-imports echarts on first mount so
	// the ~1MB library is only paid for when a chart widget is actually shown.
	// Re-renders on `option` change and resizes with its container (gridstack).
	import { onMount, onDestroy } from 'svelte';

	export let option: any = null;
	export let height = '100%';

	let el: HTMLDivElement;
	let chart: any = null;
	let echarts: any = null;
	let ro: ResizeObserver | null = null;
	let mounted = false;

	onMount(async () => {
		const mod = await import('echarts');
		echarts = mod;
		chart = echarts.init(el, null, { renderer: 'canvas' });
		if (option) chart.setOption(option, true);
		ro = new ResizeObserver(() => {
			if (chart) chart.resize();
		});
		ro.observe(el);
		mounted = true;
	});

	// notMerge=true so a full option swap (e.g. theme change) replaces cleanly.
	$: if (mounted && chart && option) chart.setOption(option, true);

	onDestroy(() => {
		ro?.disconnect();
		ro = null;
		if (chart) {
			chart.dispose();
			chart = null;
		}
	});
</script>

<div bind:this={el} style="width:100%;height:{height};min-height:0;"></div>
