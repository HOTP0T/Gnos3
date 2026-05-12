<script lang="ts">
	import { onMount, onDestroy, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getBusinessCards,
		getBusinessCardStats,
		type BusinessCard,
		type BusinessCardStats
	} from '$lib/apis/business-cards';
	import { K4MI_BASE_URL } from '$lib/constants';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	let loading = true;
	let stats: BusinessCardStats | null = null;
	let recentCards: BusinessCard[] = [];
	let reviewCards: BusinessCard[] = [];
	let refreshInterval: ReturnType<typeof setInterval> | null = null;

	const loadData = async () => {
		loading = true;
		try {
			const [statsResp, recent, review] = await Promise.all([
				getBusinessCardStats(localStorage.token),
				getBusinessCards(localStorage.token, {
					sort_by: 'created_at',
					sort_dir: 'desc',
					limit: 10
				}),
				getBusinessCards(localStorage.token, {
					needs_review: true,
					sort_by: 'created_at',
					sort_dir: 'desc',
					limit: 10
				})
			]);

			stats = statsResp;
			recentCards = recent.business_cards ?? [];
			reviewCards = review.business_cards ?? [];

			if ((stats?.processing ?? 0) > 0 && !refreshInterval) {
				refreshInterval = setInterval(() => loadData(), 5000);
			} else if ((stats?.processing ?? 0) === 0 && refreshInterval) {
				clearInterval(refreshInterval);
				refreshInterval = null;
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	onMount(loadData);
	onDestroy(() => {
		if (refreshInterval) clearInterval(refreshInterval);
	});

	const k4miHref = (id: number | null) =>
		id ? `${K4MI_BASE_URL}/documents/${id}/details` : '#';
</script>

<div class="py-3">
	<div class="flex items-center justify-between mb-4">
		<h1 class="text-xl font-semibold">{$i18n.t('Business cards')}</h1>
		<button
			class="text-xs px-3 py-1.5 rounded-full bg-gray-50 dark:bg-gray-850 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
			on:click={loadData}
			disabled={loading}
		>
			{loading ? $i18n.t('Loading...') : $i18n.t('Refresh')}
		</button>
	</div>

	{#if loading && !stats}
		<div class="flex justify-center py-16"><Spinner /></div>
	{:else if stats}
		<!-- Stat tiles -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
			<div class="rounded-xl bg-gray-50 dark:bg-gray-850 px-4 py-3">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Total')}</div>
				<div class="text-2xl font-semibold">{stats.total}</div>
			</div>
			<div class="rounded-xl bg-gray-50 dark:bg-gray-850 px-4 py-3">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Completed')}</div>
				<div class="text-2xl font-semibold">{stats.completed}</div>
			</div>
			<button
				class="text-left rounded-xl bg-gray-50 dark:bg-gray-850 px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
				on:click={() => goto('/business-cards/data?needs_review=true')}
			>
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Needs review')}</div>
				<div class="text-2xl font-semibold">{stats.needs_review}</div>
			</button>
			<div class="rounded-xl bg-gray-50 dark:bg-gray-850 px-4 py-3">
				<div class="text-xs text-gray-500 dark:text-gray-400">
					{$i18n.t('Added in last 30 days')}
				</div>
				<div class="text-2xl font-semibold">{stats.recent_added_30d}</div>
			</div>
		</div>

		{#if stats.processing > 0}
			<div class="mb-4 px-4 py-2 rounded-xl bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 text-sm flex items-center gap-2">
				<Spinner className="size-3" />
				{stats.processing} {$i18n.t('cards still processing — refreshing every 5s')}
			</div>
		{/if}

		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<!-- Top companies -->
			<section class="rounded-xl bg-gray-50 dark:bg-gray-850 p-4">
				<h2 class="text-sm font-semibold mb-3">{$i18n.t('Top companies')}</h2>
				{#if stats.by_company.length === 0}
					<div class="text-xs text-gray-500 dark:text-gray-400 py-3">
						{$i18n.t('No company data yet.')}
					</div>
				{:else}
					{@const max = Math.max(...stats.by_company.map((c) => c.count))}
					<ul class="space-y-1.5">
						{#each stats.by_company as row}
							<li class="flex items-center gap-3 text-sm">
								<span class="flex-1 truncate">{row.company_name}</span>
								<div class="flex-1 h-2 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden">
									<div
										class="h-full bg-gray-700 dark:bg-gray-300"
										style="width: {(row.count / max) * 100}%"
									></div>
								</div>
								<span class="w-10 text-right tabular-nums">{row.count}</span>
							</li>
						{/each}
					</ul>
				{/if}
			</section>

			<!-- Recent additions -->
			<section class="rounded-xl bg-gray-50 dark:bg-gray-850 p-4">
				<div class="flex items-center justify-between mb-3">
					<h2 class="text-sm font-semibold">{$i18n.t('Recent additions')}</h2>
					<a
						class="text-xs text-gray-500 dark:text-gray-400 hover:underline"
						href="/business-cards/data">{$i18n.t('View all')}</a
					>
				</div>
				{#if recentCards.length === 0}
					<div class="text-xs text-gray-500 dark:text-gray-400 py-3">
						{$i18n.t('No business cards yet. Tag a K4mi document with')} <code>business_card</code> {$i18n.t('to get started.')}
					</div>
				{:else}
					<ul class="divide-y divide-gray-200/60 dark:divide-gray-800/60">
						{#each recentCards as card}
							<li class="py-2 flex items-center gap-3">
								<div class="flex-1 min-w-0">
									<div class="text-sm font-medium truncate">{card.full_name}</div>
									<div class="text-xs text-gray-500 dark:text-gray-400 truncate">
										{[card.job_title, card.company_name].filter(Boolean).join(' · ') || '—'}
									</div>
								</div>
								<div class="text-xs text-gray-400">
									{card.created_at ? dayjs(card.created_at).format('MMM D') : ''}
								</div>
								{#if card.k4mi_document_id}
									<Tooltip content={$i18n.t('Open in K4mi')}>
										<a
											class="text-xs text-gray-500 dark:text-gray-400 hover:underline"
											href={k4miHref(card.k4mi_document_id)}
											target="_blank"
											rel="noopener noreferrer"
										>K4mi</a>
									</Tooltip>
								{/if}
							</li>
						{/each}
					</ul>
				{/if}
			</section>
		</div>

		{#if reviewCards.length > 0}
			<section class="mt-4 rounded-xl bg-amber-50 dark:bg-amber-950/30 p-4">
				<div class="flex items-center justify-between mb-2">
					<h2 class="text-sm font-semibold text-amber-900 dark:text-amber-200">
						{$i18n.t('Needs review')}
					</h2>
					<a
						class="text-xs text-amber-700 dark:text-amber-300 hover:underline"
						href="/business-cards/data?needs_review=true">{$i18n.t('Review all')}</a
					>
				</div>
				<ul class="divide-y divide-amber-200/40 dark:divide-amber-800/40">
					{#each reviewCards as card}
						<li class="py-2 flex items-center gap-3 text-sm">
							<span class="flex-1 truncate">{card.full_name}</span>
							<span class="text-xs text-amber-700 dark:text-amber-300">
								{card.confidence_score !== null
									? `${Math.round((card.confidence_score ?? 0) * 100)}%`
									: ''}
							</span>
							{#if card.k4mi_document_id}
								<a
									class="text-xs text-amber-700 dark:text-amber-300 hover:underline"
									href={k4miHref(card.k4mi_document_id)}
									target="_blank"
									rel="noopener noreferrer"
								>K4mi</a>
							{/if}
						</li>
					{/each}
				</ul>
			</section>
		{/if}
	{/if}
</div>
