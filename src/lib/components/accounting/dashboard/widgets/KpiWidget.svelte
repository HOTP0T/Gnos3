<script lang="ts">
	// Generic KPI tile. `metric` (from the widget's registry options) selects
	// which number to show; the value is read from the shared common-data store
	// so all KPI tiles share a single batch of API calls.
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { COMMON_DATA_CTX, type CommonData } from '../store';
	import { money, fmtNumber } from '../format';

	export let companyId: number; // unused here but part of the widget contract
	export let options: { metric?: string } = {};

	const common = getContext<Writable<CommonData>>(COMMON_DATA_CTX);
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');

	$: nativeCurrency = ($companyCurrencyCtx as any) || 'EUR';

	type Kind = 'money' | 'count' | 'ratio';
	interface Metric {
		label: string;
		kind: Kind;
		value: (d: CommonData) => number;
		sub?: (d: CommonData) => string | null;
		tone?: (v: number, d: CommonData) => 'pos' | 'neg' | 'warn' | 'neutral';
	}

	const METRICS: Record<string, Metric> = {
		assets: { label: 'Total Assets', kind: 'money', value: (d) => d.balanceSheet?.total_assets ?? 0 },
		liabilities: { label: 'Total Liabilities', kind: 'money', value: (d) => d.balanceSheet?.total_liabilities ?? 0 },
		net_income: {
			label: 'Net Income (Current Month)',
			kind: 'money',
			value: (d) => d.profitLoss?.net_income ?? 0,
			tone: (v) => (v >= 0 ? 'pos' : 'neg')
		},
		cash: {
			label: 'Cash Position',
			kind: 'money',
			value: (d) => d.stats?.cash_position ?? 0,
			tone: (v) => (v >= 0 ? 'neutral' : 'neg')
		},
		ar: { label: 'Accounts Receivable', kind: 'money', value: (d) => d.stats?.outstanding_ar ?? 0 },
		ap: { label: 'Accounts Payable', kind: 'money', value: (d) => d.stats?.outstanding_ap ?? 0 },
		overdue: {
			label: 'Overdue Invoices',
			kind: 'count',
			value: (d) => d.stats?.overdue_invoices ?? 0,
			sub: (d) => ((d.stats?.overdue_invoices ?? 0) > 0 ? fmtNumber(d.stats?.overdue_amount ?? 0) : null),
			tone: (v) => (v > 0 ? 'neg' : 'neutral')
		},
		drafts: {
			label: 'Draft Entries',
			kind: 'count',
			value: (d) => d.stats?.draft_count ?? 0,
			tone: (v) => (v > 0 ? 'warn' : 'neutral')
		},
		unmatched: {
			label: 'Unmatched Bank Lines',
			kind: 'count',
			value: (d) => d.stats?.unmatched_bank_lines ?? 0,
			tone: (v) => (v > 0 ? 'warn' : 'neutral')
		},
		current_ratio: {
			label: 'Assets / Liabilities',
			kind: 'ratio',
			value: (d) => {
				const l = d.balanceSheet?.total_liabilities ?? 0;
				return l ? (d.balanceSheet?.total_assets ?? 0) / l : 0;
			},
			tone: (v) => (v >= 1 ? 'pos' : v > 0 ? 'warn' : 'neutral')
		}
	};

	$: metric = METRICS[options?.metric ?? 'cash'] ?? METRICS.cash;
	$: loading = $common.loading;
	$: rawValue = metric.value($common);
	$: tone = metric.tone ? metric.tone(rawValue, $common) : 'neutral';
	$: subText = metric.sub ? metric.sub($common) : null;
	$: mv =
		metric.kind === 'money'
			? money(rawValue, nativeCurrency, $displayCurrency, $exchangeRates ?? [])
			: null;

	const toneClass: Record<string, string> = {
		pos: 'text-green-700 dark:text-green-400',
		neg: 'text-red-700 dark:text-red-400',
		warn: 'text-amber-600 dark:text-amber-400',
		neutral: 'text-gray-900 dark:text-gray-100'
	};
</script>

<div class="h-full flex flex-col justify-between">
	<div class="text-[11px] text-gray-500 dark:text-gray-400 leading-tight">{metric.label}</div>
	{#if loading}
		<div class="h-7 w-2/3 rounded bg-gray-100 dark:bg-gray-800 animate-pulse mt-1"></div>
	{:else if metric.kind === 'money' && mv}
		<div class="text-2xl font-medium leading-none {toneClass[tone]}">
			{#if mv.converting && mv.hasRate}
				{mv.display} <span class="text-sm text-gray-400">{$displayCurrency}</span>
				<div class="text-[10px] text-gray-400 mt-0.5">{mv.original} {nativeCurrency}</div>
			{:else}
				{mv.original} <span class="text-sm text-gray-400">{nativeCurrency}</span>
			{/if}
		</div>
	{:else if metric.kind === 'ratio'}
		<div class="text-2xl font-medium leading-none {toneClass[tone]}">
			{rawValue ? rawValue.toFixed(2) : '—'}
		</div>
	{:else}
		<div class="text-2xl font-medium leading-none flex items-baseline gap-2 {toneClass[tone]}">
			{rawValue}
			{#if subText}<span class="text-sm font-normal text-red-500 dark:text-red-400">{subText}</span>{/if}
		</div>
	{/if}
</div>
