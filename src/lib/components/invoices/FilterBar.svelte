<script lang="ts">
	import { getContext, createEventDispatcher } from 'svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let vendor = '';
	export let dateFrom = '';
	export let dateTo = '';
	export let minAmount: string = '';
	export let maxAmount: string = '';
	export let status = '';
	export let needsReview: boolean | undefined = undefined;
	export let tag = '';
	export let show = false;
	export let availableTags: string[] = [];

	const applyFilters = () => {
		dispatch('filter', {
			vendor,
			date_from: dateFrom || undefined,
			date_to: dateTo || undefined,
			min_amount: minAmount ? parseFloat(minAmount) : undefined,
			max_amount: maxAmount ? parseFloat(maxAmount) : undefined,
			status: status || undefined,
			needs_review: needsReview,
			tag: tag || undefined
		});
	};

	const clearFilters = () => {
		vendor = '';
		dateFrom = '';
		dateTo = '';
		minAmount = '';
		maxAmount = '';
		status = '';
		needsReview = undefined;
		tag = '';
		applyFilters();
	};
</script>

{#if show}
	<div
		class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 p-3 mb-2"
	>
		<div class="grid grid-cols-2 md:grid-cols-4 gap-2">
			<!-- Vendor -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Vendor')}</label>
				<input
					class="w-full text-sm bg-transparent dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					type="text"
					bind:value={vendor}
					on:input={applyFilters}
					placeholder={$i18n.t('Search vendor...')}
				/>
			</div>

			<!-- Date From -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Date From')}</label>
				<input
					class="w-full text-sm bg-transparent dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					type="date"
					bind:value={dateFrom}
					on:change={applyFilters}
				/>
			</div>

			<!-- Date To -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Date To')}</label>
				<input
					class="w-full text-sm bg-transparent dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					type="date"
					bind:value={dateTo}
					on:change={applyFilters}
				/>
			</div>

			<!-- Status -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Status')}</label>
				<select
					class="w-full text-sm bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					bind:value={status}
					on:change={applyFilters}
				>
					<option value="">{$i18n.t('All')}</option>
					<option value="completed">{$i18n.t('Completed')}</option>
					<option value="processing">{$i18n.t('Processing')}</option>
					<option value="pending">{$i18n.t('Pending')}</option>
					<option value="failed">{$i18n.t('Failed')}</option>
				</select>
			</div>

			<!-- Min Amount -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Min Amount')}</label>
				<input
					class="w-full text-sm bg-transparent dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					type="number"
					step="0.01"
					bind:value={minAmount}
					on:input={applyFilters}
					placeholder="0.00"
				/>
			</div>

			<!-- Max Amount -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Max Amount')}</label>
				<input
					class="w-full text-sm bg-transparent dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					type="number"
					step="0.01"
					bind:value={maxAmount}
					on:input={applyFilters}
					placeholder="0.00"
				/>
			</div>

			<!-- Needs Review -->
			<div class="flex flex-col">
				<label class="text-xs text-gray-500 mb-1">{$i18n.t('Needs Review')}</label>
				<select
					class="w-full text-sm bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2.5 py-1.5"
					bind:value={needsReview}
					on:change={applyFilters}
				>
					<option value={undefined}>{$i18n.t('All')}</option>
					<option value={true}>{$i18n.t('Yes')}</option>
					<option value={false}>{$i18n.t('No')}</option>
				</select>
			</div>

			<!-- Clear -->
			<div class="flex flex-col justify-end">
				<button
					class="text-xs px-3 py-1.5 bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 transition rounded-lg font-medium dark:text-gray-200"
					on:click={clearFilters}
				>
					{$i18n.t('Clear Filters')}
				</button>
			</div>
		</div>

		{#if availableTags.length > 0}
			<div class="mt-2 flex flex-wrap gap-1.5">
				{#each availableTags as t}
					<button
						class="text-xs px-2 py-0.5 rounded-full border transition {tag === t
							? 'bg-blue-500 border-blue-500 text-white'
							: 'bg-transparent border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:border-blue-400 hover:text-blue-500'}"
						on:click={() => {
							tag = tag === t ? '' : t;
							applyFilters();
						}}
					>
						{t}
					</button>
				{/each}
			</div>
		{/if}
	</div>
{/if}
