<script lang="ts">
	// Renders a report money amount. When a display currency (≠ base) is selected it
	// shows the converted amount on top with the original base-currency amount in
	// small grey text underneath as reference — matching the tables/bank views.
	//
	// `value` is the raw amount in `baseCcy`. `factor` is the base→display multiplier
	// (null = no rate available: falls back to the base amount + a warning).
	export let value: any;
	export let factor: number | null = 1;
	export let converting = false;
	export let displayCcy = '';
	export let baseCcy = '';
	export let abs = false; // show absolute value (for split debit/credit columns)
	export let zeroText: string | null = ''; // shown for a zero value; null = format 0 normally
	export let minFrac = 2;
	export let maxFrac = 2;
	export let locale: string | undefined = undefined;

	const toNum = (v: any): number => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return isNaN(n) ? 0 : n;
	};
	const f = (n: number): string => {
		const x = abs ? Math.abs(n) : n;
		if (x === 0 && zeroText !== null) return zeroText;
		return x.toLocaleString(locale, { minimumFractionDigits: minFrac, maximumFractionDigits: maxFrac });
	};

	$: base = toNum(value);
	$: hasRate = factor != null;
	$: topStr = converting && hasRate ? f(base * (factor as number)) : f(base);
	$: origStr = f(base);
</script>

{#if !converting}
	{f(base)}
{:else}
	{#key displayCcy}
		{#if hasRate}
			<span>{topStr}{#if topStr} <span class="text-[9px] text-gray-400 dark:text-gray-500 font-normal">{displayCcy}</span>{/if}</span>
			{#if origStr}<span class="block text-[9px] text-gray-400 dark:text-gray-500 font-normal">{origStr} {baseCcy}</span>{/if}
		{:else}
			<span>{origStr}{#if origStr} <span class="text-[9px] text-gray-400 dark:text-gray-500 font-normal">{baseCcy}</span>{/if}</span>
			<span class="text-[9px] text-amber-500" title="No exchange rate">&#9888;</span>
		{/if}
	{/key}
{/if}
