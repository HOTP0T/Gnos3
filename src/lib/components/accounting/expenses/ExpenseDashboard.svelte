<script lang="ts">
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import { getExpenseStats, type ExpenseStats, type ExpenseStatGroup } from '$lib/apis/expenses';

	export let companyId: number;

	let loading = true;
	let stats: ExpenseStats | null = null;

	function fmtCur(byCur: Record<string, number>): string {
		const entries = Object.entries(byCur || {}).filter(([, v]) => v);
		if (!entries.length) return '—';
		return entries
			.map(([c, v]) => `${c} ${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`)
			.join(' · ');
	}
	const total = (byCur: Record<string, number>) =>
		Object.values(byCur || {}).reduce((a, b) => a + b, 0);

	$: statusOf = (k: string) => stats?.by_status?.[k] ?? { count: 0, by_currency: {} };
	$: catMax = Math.max(1, ...(stats?.by_category ?? []).map((c) => total(c.by_currency)));
	$: empMax = Math.max(1, ...(stats?.by_employee ?? []).map((e) => total(e.by_currency)));

	async function load() {
		loading = true;
		try {
			stats = await getExpenseStats(companyId);
		} catch (e: any) {
			toast.error(e?.message ?? 'Failed to load stats');
		} finally {
			loading = false;
		}
	}
	onMount(load);

	const TILES = [
		{ key: 'submitted', label: 'Pending review', cls: 'text-blue-600 dark:text-blue-400' },
		{ key: 'approved', label: 'To reimburse', cls: 'text-emerald-600 dark:text-emerald-400' },
		{ key: 'reimbursed', label: 'Reimbursed', cls: 'text-violet-600 dark:text-violet-400' },
		{ key: 'rejected', label: 'Rejected', cls: 'text-red-500 dark:text-red-400' }
	];
</script>

{#if loading}
	<div class="flex justify-center py-12"><Spinner /></div>
{:else if stats}
	<div class="flex flex-col gap-4">
		<!-- Stat tiles -->
		<div class="grid grid-cols-2 gap-3 md:grid-cols-4">
			{#each TILES as t}
				{@const b = statusOf(t.key)}
				<div class="rounded-xl border border-gray-100 bg-white p-3 dark:border-gray-850 dark:bg-gray-900">
					<div class="text-[11px] uppercase tracking-wide text-gray-500 dark:text-gray-400">{t.label}</div>
					<div class="mt-1 text-2xl font-semibold {t.cls}">{b.count}</div>
					<div class="text-xs text-gray-500 dark:text-gray-400">{fmtCur(b.by_currency)}</div>
				</div>
			{/each}
		</div>

		<div class="grid grid-cols-1 gap-4 md:grid-cols-2">
			<!-- Spend by category -->
			<div class="rounded-xl border border-gray-100 p-4 dark:border-gray-850">
				<h3 class="mb-3 text-sm font-semibold text-gray-800 dark:text-gray-100">Spend by category</h3>
				{#if !stats.by_category.length}
					<p class="text-sm text-gray-400">No data.</p>
				{:else}
					<div class="flex flex-col gap-2.5">
						{#each stats.by_category as c (c.category_id ?? 0)}
							<div>
								<div class="mb-1 flex items-center justify-between text-xs">
									<span class="text-gray-700 dark:text-gray-200">{c.label} <span class="text-gray-400">· {c.count}</span></span>
									<span class="font-medium text-gray-600 dark:text-gray-300">{fmtCur(c.by_currency)}</span>
								</div>
								<div class="h-2 overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
									<div class="h-full rounded-full bg-blue-500" style="width:{Math.max(3, (total(c.by_currency) / catMax) * 100)}%"></div>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>

			<!-- Spend by employee -->
			<div class="rounded-xl border border-gray-100 p-4 dark:border-gray-850">
				<h3 class="mb-3 text-sm font-semibold text-gray-800 dark:text-gray-100">Spend by employee</h3>
				{#if !stats.by_employee.length}
					<p class="text-sm text-gray-400">No data.</p>
				{:else}
					<div class="flex flex-col gap-2.5">
						{#each stats.by_employee as e (e.employee_id ?? 0)}
							<div>
								<div class="mb-1 flex items-center justify-between text-xs">
									<span class="text-gray-700 dark:text-gray-200">{e.name} <span class="text-gray-400">· {e.count}</span></span>
									<span class="font-medium text-gray-600 dark:text-gray-300">{fmtCur(e.by_currency)}</span>
								</div>
								<div class="h-2 overflow-hidden rounded-full bg-gray-100 dark:bg-gray-800">
									<div class="h-full rounded-full bg-emerald-500" style="width:{Math.max(3, (total(e.by_currency) / empMax) * 100)}%"></div>
								</div>
							</div>
						{/each}
					</div>
				{/if}
			</div>
		</div>

		{#if stats.by_category.some((c) => Object.keys(c.by_currency).length > 1) || Object.values(stats.by_status).some((s) => Object.keys(s.by_currency).length > 1)}
			<p class="text-xs text-gray-400">
				Amounts are grouped by currency; bar lengths are approximate across currencies until FX
				conversion is enabled.
			</p>
		{/if}
	</div>
{/if}
