<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { page as pageStore } from '$app/stores';
	import { toast } from 'svelte-sonner';

	import {
		getBusinessCards,
		getBusinessCardCompanies,
		deleteBusinessCard,
		reprocessBusinessCard,
		syncBusinessCardFromK4mi,
		type BusinessCard
	} from '$lib/apis/business-cards';
	import { K4MI_BASE_URL } from '$lib/constants';

	import Pagination from '$lib/components/common/Pagination.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import BusinessCardDetailDrawer from './BusinessCardDetailDrawer.svelte';

	const i18n = getContext('i18n');

	let cards: BusinessCard[] = [];
	let companies: string[] = [];
	let total = 0;
	let loading = true;

	let page = 1;
	let perPage = 50;

	let searchQuery = '';
	let searchDebounce: ReturnType<typeof setTimeout>;
	let companyFilter = '';
	let needsReviewOnly = false;
	let statusFilter = '';

	let sortBy = '';
	let sortDir: 'asc' | 'desc' = 'asc';

	let showDeleteConfirm = false;
	let deleteTarget: BusinessCard | null = null;
	let reprocessingIds: Set<number> = new Set();
	let syncingIds: Set<number> = new Set();

	let selectedCard: BusinessCard | null = null;

	let density: 'compact' | 'comfortable' = 'comfortable';
	let columnsMenuOpen = false;
	let visibleColumns: Set<string> = new Set();

	const COLUMNS: Array<{
		key: keyof BusinessCard;
		label: string;
		sortable: boolean;
		readonly?: boolean;
		minW: string;
		alwaysVisible?: boolean;
	}> = [
		{ key: 'full_name', label: 'Name', sortable: true, minW: '12rem', alwaysVisible: true },
		{ key: 'job_title', label: 'Job title', sortable: false, minW: '10rem' },
		{ key: 'company_name', label: 'Company', sortable: true, minW: '10rem' },
		{ key: 'email', label: 'Email', sortable: true, minW: '14rem' },
		{ key: 'phone', label: 'Phone', sortable: false, minW: '8rem' },
		{ key: 'mobile', label: 'Mobile', sortable: false, minW: '8rem' },
		{ key: 'website', label: 'Website', sortable: false, minW: '10rem' },
		{ key: 'k4mi_notes', label: 'Notes', sortable: false, readonly: true, minW: '16rem' }
	];

	$: shownColumns = COLUMNS.filter(
		(c) => c.alwaysVisible || visibleColumns.has(String(c.key))
	);

	$: cellPad = density === 'compact' ? 'px-2.5 py-1' : 'px-3 py-2';
	$: headPad = density === 'compact' ? 'px-2.5 py-1.5' : 'px-3 py-2.5';
	$: rowText = density === 'compact' ? 'text-xs' : 'text-sm';

	const formatK4miNotes = (notes: unknown): string => {
		if (!Array.isArray(notes) || notes.length === 0) return '';
		return notes
			.map((n) => {
				if (typeof n === 'string') return n;
				if (n && typeof n === 'object' && 'text' in n)
					return String((n as { text?: unknown }).text ?? '');
				return '';
			})
			.filter(Boolean)
			.join(' · ');
	};

	const loadCards = async () => {
		loading = true;
		try {
			const res = await getBusinessCards(localStorage.token, {
				q: searchQuery || undefined,
				company: companyFilter || undefined,
				status: statusFilter || undefined,
				needs_review: needsReviewOnly ? true : undefined,
				sort_by: sortBy || undefined,
				sort_dir: sortBy ? sortDir : undefined,
				limit: perPage,
				offset: (page - 1) * perPage
			});
			cards = res.business_cards ?? [];
			total = res.total ?? 0;

			if (selectedCard) {
				const refreshed = cards.find((c) => c.id === selectedCard!.id);
				if (refreshed) selectedCard = refreshed;
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const loadCompanies = async () => {
		try {
			companies = await getBusinessCardCompanies(localStorage.token);
		} catch {
			companies = [];
		}
	};

	const handleSort = (key: string) => {
		if (sortBy === key) {
			if (sortDir === 'asc') {
				sortDir = 'desc';
			} else {
				sortBy = '';
				sortDir = 'asc';
			}
		} else {
			sortBy = key;
			sortDir = 'asc';
		}
		page = 1;
		loadCards();
	};

	const debouncedSearch = () => {
		clearTimeout(searchDebounce);
		searchDebounce = setTimeout(() => {
			page = 1;
			loadCards();
		}, 300);
	};

	const handleDelete = (card: BusinessCard) => {
		deleteTarget = card;
		showDeleteConfirm = true;
	};

	const confirmDelete = async () => {
		if (!deleteTarget) return;
		try {
			await deleteBusinessCard(localStorage.token, deleteTarget.id);
			cards = cards.filter((c) => c.id !== deleteTarget!.id);
			total = Math.max(0, total - 1);
			if (selectedCard?.id === deleteTarget.id) selectedCard = null;
			toast.success($i18n.t('Deleted'));
		} catch (err) {
			toast.error(`${err}`);
		}
		deleteTarget = null;
		showDeleteConfirm = false;
	};

	const handleReprocess = async (card: BusinessCard) => {
		if (!card.k4mi_document_id) return;
		reprocessingIds = new Set([...reprocessingIds, card.id]);
		try {
			await reprocessBusinessCard(localStorage.token, card.id);
			toast.success($i18n.t('Re-extraction queued'));
			setTimeout(loadCards, 1500);
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			reprocessingIds.delete(card.id);
			reprocessingIds = new Set(reprocessingIds);
		}
	};

	const handleSync = async (card: BusinessCard) => {
		if (!card.k4mi_document_id) return;
		syncingIds = new Set([...syncingIds, card.id]);
		try {
			await syncBusinessCardFromK4mi(localStorage.token, card.id);
			toast.success($i18n.t('Syncing from K4mi…'));
			setTimeout(loadCards, 1500);
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			syncingIds.delete(card.id);
			syncingIds = new Set(syncingIds);
		}
	};

	const k4miHref = (id: number | null) =>
		id ? `${K4MI_BASE_URL}/documents/${id}/details` : '#';

	const openCard = (card: BusinessCard) => {
		selectedCard = card;
	};

	const onCardUpdated = (updated: BusinessCard) => {
		const idx = cards.findIndex((c) => c.id === updated.id);
		if (idx !== -1) cards[idx] = updated;
		cards = [...cards];
		selectedCard = updated;
	};

	const toggleDensity = () => {
		density = density === 'compact' ? 'comfortable' : 'compact';
		try {
			localStorage.setItem('bc-table-density', density);
		} catch {}
	};

	const toggleColumn = (key: string) => {
		if (visibleColumns.has(key)) {
			visibleColumns.delete(key);
		} else {
			visibleColumns.add(key);
		}
		visibleColumns = new Set(visibleColumns);
		try {
			localStorage.setItem(
				'bc-table-columns',
				JSON.stringify(Array.from(visibleColumns))
			);
		} catch {}
	};

	const resetColumns = () => {
		visibleColumns = new Set(COLUMNS.map((c) => String(c.key)));
		try {
			localStorage.setItem(
				'bc-table-columns',
				JSON.stringify(Array.from(visibleColumns))
			);
		} catch {}
	};

	const handleClickOutside = (e: MouseEvent) => {
		if (!columnsMenuOpen) return;
		const target = e.target as HTMLElement;
		if (!target.closest('[data-columns-menu]')) {
			columnsMenuOpen = false;
		}
	};

	onMount(() => {
		try {
			const d = localStorage.getItem('bc-table-density');
			if (d === 'compact' || d === 'comfortable') density = d;
		} catch {}

		try {
			const stored = localStorage.getItem('bc-table-columns');
			if (stored) {
				visibleColumns = new Set(JSON.parse(stored));
			} else {
				visibleColumns = new Set(COLUMNS.map((c) => String(c.key)));
			}
		} catch {
			visibleColumns = new Set(COLUMNS.map((c) => String(c.key)));
		}

		const url = $pageStore.url;
		if (url.searchParams.get('needs_review') === 'true') needsReviewOnly = true;
		const company = url.searchParams.get('company');
		if (company) companyFilter = company;

		loadCards();
		loadCompanies();
	});

	$: page, loadCards();
	$: needsReviewOnly, statusFilter, companyFilter, (page = 1, loadCards());
</script>

<svelte:window on:click={handleClickOutside} />

<div class="py-3">
	<div class="flex flex-wrap items-center gap-2 mb-3">
		<input
			type="search"
			class="px-3 py-1.5 text-sm rounded-full bg-gray-50 dark:bg-gray-850 outline-none focus:ring-1 focus:ring-gray-400"
			placeholder={$i18n.t('Search name, company, email, notes…')}
			bind:value={searchQuery}
			on:input={debouncedSearch}
		/>

		<select
			class="px-3 py-1.5 text-sm rounded-full bg-gray-50 dark:bg-gray-850 outline-none"
			bind:value={companyFilter}
		>
			<option value="">{$i18n.t('All companies')}</option>
			{#each companies as c}
				<option value={c}>{c}</option>
			{/each}
		</select>

		<select
			class="px-3 py-1.5 text-sm rounded-full bg-gray-50 dark:bg-gray-850 outline-none"
			bind:value={statusFilter}
		>
			<option value="">{$i18n.t('Any status')}</option>
			<option value="completed">{$i18n.t('Completed')}</option>
			<option value="processing">{$i18n.t('Processing')}</option>
			<option value="pending">{$i18n.t('Pending')}</option>
			<option value="failed">{$i18n.t('Failed')}</option>
		</select>

		<label class="flex items-center gap-1.5 text-sm">
			<input type="checkbox" bind:checked={needsReviewOnly} />
			{$i18n.t('Needs review only')}
		</label>

		<div class="ml-auto flex items-center gap-1.5">
			<span class="text-xs text-gray-500 dark:text-gray-400">
				{total} {$i18n.t('cards')}
			</span>

			<Tooltip content={density === 'compact' ? $i18n.t('Comfortable density') : $i18n.t('Compact density')}>
				<button
					class="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
					on:click={toggleDensity}
					aria-label={$i18n.t('Toggle density')}
				>
					{#if density === 'compact'}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="size-4"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="12" x2="21" y2="12" /><line
								x1="3"
								y1="18"
								x2="21"
								y2="18"
							/></svg
						>
					{:else}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="size-4"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><line x1="3" y1="4" x2="21" y2="4" /><line x1="3" y1="8" x2="21" y2="8" /><line
								x1="3"
								y1="12"
								x2="21"
								y2="12"
							/><line x1="3" y1="16" x2="21" y2="16" /><line
								x1="3"
								y1="20"
								x2="21"
								y2="20"
							/></svg
						>
					{/if}
				</button>
			</Tooltip>

			<div class="relative" data-columns-menu>
				<Tooltip content={$i18n.t('Show/hide columns')}>
					<button
						class="p-1.5 rounded-md text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
						on:click={() => (columnsMenuOpen = !columnsMenuOpen)}
						aria-label={$i18n.t('Show/hide columns')}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="size-4"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><rect x="3" y="3" width="18" height="18" rx="2" /><line
								x1="9"
								y1="3"
								x2="9"
								y2="21"
							/><line x1="15" y1="3" x2="15" y2="21" /></svg
						>
					</button>
				</Tooltip>

				{#if columnsMenuOpen}
					<div
						class="absolute right-0 top-full mt-1 z-30 w-56 rounded-md bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 shadow-lg py-1.5"
					>
						<div
							class="px-3 py-1 text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400"
						>
							{$i18n.t('Visible columns')}
						</div>
						{#each COLUMNS as col}
							<label
								class="flex items-center gap-2 px-3 py-1.5 text-sm hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer {col.alwaysVisible
									? 'opacity-60 cursor-not-allowed'
									: ''}"
							>
								<input
									type="checkbox"
									checked={col.alwaysVisible || visibleColumns.has(String(col.key))}
									disabled={col.alwaysVisible}
									on:change={() => !col.alwaysVisible && toggleColumn(String(col.key))}
								/>
								<span class="text-gray-700 dark:text-gray-200">{$i18n.t(col.label)}</span>
							</label>
						{/each}
						<div class="border-t border-gray-100 dark:border-gray-800 mt-1.5 pt-1.5">
							<button
								class="w-full text-left px-3 py-1.5 text-sm text-blue-600 dark:text-blue-400 hover:bg-gray-50 dark:hover:bg-gray-800"
								on:click={resetColumns}
							>
								{$i18n.t('Show all')}
							</button>
						</div>
					</div>
				{/if}
			</div>
		</div>
	</div>

	{#if loading && cards.length === 0}
		<div class="flex justify-center py-16"><Spinner /></div>
	{:else if cards.length === 0}
		<div class="text-center py-16 text-sm text-gray-500 dark:text-gray-400">
			{$i18n.t('No business cards found.')}
			<div class="mt-2 text-xs">
				{$i18n.t('Tag a K4mi document with')} <code>business_card</code> {$i18n.t('to extract one.')}
			</div>
		</div>
	{:else}
		<div class="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
			<table class="w-full {rowText} border-collapse">
				<thead
					class="text-left text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 bg-gray-50/80 dark:bg-gray-900/60 sticky top-0 z-10 backdrop-blur"
				>
					<tr>
						{#each shownColumns as col}
							<th
								class="{headPad} font-medium whitespace-nowrap border-b border-gray-200 dark:border-gray-800"
								style="min-width: {col.minW}"
							>
								{#if col.sortable}
									<button
										class="inline-flex items-center gap-1 hover:text-gray-800 dark:hover:text-gray-200 transition"
										on:click={() => handleSort(col.key)}
									>
										<span>{$i18n.t(col.label)}</span>
										{#if sortBy === col.key}
											<span class="text-gray-400">{sortDir === 'asc' ? '↑' : '↓'}</span>
										{/if}
									</button>
								{:else}
									{$i18n.t(col.label)}
								{/if}
							</th>
						{/each}
						<th
							class="{headPad} font-medium whitespace-nowrap border-b border-gray-200 dark:border-gray-800"
							style="width: 5.5rem"
						>
							{$i18n.t('Status')}
						</th>
						<th
							class="{headPad} font-medium whitespace-nowrap text-right border-b border-gray-200 dark:border-gray-800"
							style="width: 9rem"
						>
							{$i18n.t('Actions')}
						</th>
					</tr>
				</thead>
				<tbody>
					{#each cards as card, idx (card.id)}
						<tr
							class="group border-b border-gray-100 dark:border-gray-900 transition-colors cursor-pointer {idx %
								2 ===
							0
								? 'bg-white dark:bg-transparent'
								: 'bg-gray-50/40 dark:bg-gray-900/20'} hover:bg-blue-50/40 dark:hover:bg-blue-500/5"
							on:click={() => openCard(card)}
							on:keydown={(e) => {
								if (e.key === 'Enter' || e.key === ' ') {
									e.preventDefault();
									openCard(card);
								}
							}}
							role="button"
							tabindex="0"
						>
							{#each shownColumns as col}
								<td class="{cellPad} align-top" style="min-width: {col.minW}">
									{#if col.readonly && col.key === 'k4mi_notes'}
										{@const noteText = formatK4miNotes(card.k4mi_notes)}
										{#if noteText}
											<Tooltip content={noteText}>
												<span
													class="text-xs leading-snug text-gray-600 dark:text-gray-300 line-clamp-2"
												>
													{noteText}
												</span>
											</Tooltip>
										{:else}
											<span class="text-xs text-gray-400 dark:text-gray-600">—</span>
										{/if}
									{:else if (card as any)[col.key]}
										<span class="block truncate text-gray-800 dark:text-gray-200">
											{(card as any)[col.key]}
										</span>
									{:else}
										<span class="text-gray-400 dark:text-gray-600">—</span>
									{/if}
								</td>
							{/each}
							<td class="{cellPad} align-top whitespace-nowrap">
								{#if card.processing_status === 'processing' || card.processing_status === 'pending'}
									<Tooltip content={$i18n.t('Processing')}>
										<span
											class="inline-flex items-center justify-center size-6 rounded-full bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
											aria-label={$i18n.t('Processing')}
										>
											<Spinner className="size-3" />
										</span>
									</Tooltip>
								{:else if card.processing_status === 'failed'}
									<Tooltip content={$i18n.t('Failed')}>
										<span
											class="inline-flex items-center justify-center size-6 rounded-full bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-300"
											aria-label={$i18n.t('Failed')}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												class="size-3.5"
												viewBox="0 0 24 24"
												fill="none"
												stroke="currentColor"
												stroke-width="2.5"
												stroke-linecap="round"
												stroke-linejoin="round"
												><line x1="18" y1="6" x2="6" y2="18" /><line
													x1="6"
													y1="6"
													x2="18"
													y2="18"
												/></svg
											>
										</span>
									</Tooltip>
								{:else if card.needs_review}
									<Tooltip content={$i18n.t('Needs review')}>
										<span
											class="inline-flex items-center justify-center size-6 rounded-full bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
											aria-label={$i18n.t('Needs review')}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												class="size-3.5"
												viewBox="0 0 24 24"
												fill="none"
												stroke="currentColor"
												stroke-width="2.5"
												stroke-linecap="round"
												stroke-linejoin="round"
												><path d="M12 9v4" /><path d="M12 17h.01" /><path
													d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"
												/></svg
											>
										</span>
									</Tooltip>
								{:else}
									<Tooltip content={$i18n.t('OK')}>
										<span
											class="inline-flex items-center justify-center size-6 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300"
											aria-label={$i18n.t('OK')}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												class="size-3.5"
												viewBox="0 0 24 24"
												fill="none"
												stroke="currentColor"
												stroke-width="2.5"
												stroke-linecap="round"
												stroke-linejoin="round"
												><polyline points="20 6 9 17 4 12" /></svg
											>
										</span>
									</Tooltip>
								{/if}
							</td>
							<td class="{cellPad} align-top whitespace-nowrap text-right">
								<div
									class="inline-flex items-center gap-2 text-gray-500 dark:text-gray-400 transition"
								>
									{#if card.k4mi_document_id}
										<Tooltip content={$i18n.t('Open in K4mi')}>
											<a
												class="hover:text-gray-700 dark:hover:text-gray-200 transition p-1 -m-1"
												href={k4miHref(card.k4mi_document_id)}
												target="_blank"
												rel="noopener noreferrer"
												aria-label={$i18n.t('Open in K4mi')}
												on:click|stopPropagation
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													class="size-4"
													viewBox="0 0 24 24"
													fill="none"
													stroke="currentColor"
													stroke-width="2"
													stroke-linecap="round"
													stroke-linejoin="round"
													><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline
														points="15 3 21 3 21 9"
													/><line x1="10" y1="14" x2="21" y2="3" /></svg
												>
											</a>
										</Tooltip>
										<Tooltip content={$i18n.t('Sync notes & metadata from K4mi')}>
											<button
												class="hover:text-blue-600 dark:hover:text-blue-300 transition p-1 -m-1 disabled:opacity-40 disabled:cursor-not-allowed"
												on:click|stopPropagation={() => handleSync(card)}
												disabled={syncingIds.has(card.id)}
												aria-label={$i18n.t('Sync from K4mi')}
											>
												{#if syncingIds.has(card.id)}
													<Spinner className="size-4" />
												{:else}
													<svg
														xmlns="http://www.w3.org/2000/svg"
														class="size-4"
														viewBox="0 0 24 24"
														fill="none"
														stroke="currentColor"
														stroke-width="2"
														stroke-linecap="round"
														stroke-linejoin="round"
														><polyline points="23 4 23 10 17 10" /><polyline
															points="1 20 1 14 7 14"
														/><path
															d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"
														/></svg
													>
												{/if}
											</button>
										</Tooltip>
										<Tooltip content={$i18n.t('Re-extract from the document')}>
											<button
												class="hover:text-indigo-600 dark:hover:text-indigo-300 transition p-1 -m-1 disabled:opacity-40 disabled:cursor-not-allowed"
												on:click|stopPropagation={() => handleReprocess(card)}
												disabled={reprocessingIds.has(card.id)}
												aria-label={$i18n.t('Reprocess')}
											>
												{#if reprocessingIds.has(card.id)}
													<Spinner className="size-4" />
												{:else}
													<svg
														xmlns="http://www.w3.org/2000/svg"
														class="size-4"
														viewBox="0 0 24 24"
														fill="none"
														stroke="currentColor"
														stroke-width="2"
														stroke-linecap="round"
														stroke-linejoin="round"
														><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg
													>
												{/if}
											</button>
										</Tooltip>
									{/if}
									<Tooltip content={$i18n.t('Delete')}>
										<button
											class="hover:text-red-600 dark:hover:text-red-400 transition p-1 -m-1"
											on:click|stopPropagation={() => handleDelete(card)}
											aria-label={$i18n.t('Delete')}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												class="size-4"
												viewBox="0 0 24 24"
												fill="none"
												stroke="currentColor"
												stroke-width="2"
												stroke-linecap="round"
												stroke-linejoin="round"
												><polyline points="3 6 5 6 21 6" /><path
													d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"
												/><line x1="10" y1="11" x2="10" y2="17" /><line
													x1="14"
													y1="11"
													x2="14"
													y2="17"
												/></svg
											>
										</button>
									</Tooltip>
								</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<div class="mt-3">
			<Pagination bind:page count={total} {perPage} />
		</div>
	{/if}
</div>

{#if selectedCard}
	<BusinessCardDetailDrawer
		card={selectedCard}
		on:close={() => (selectedCard = null)}
		on:updated={(e) => onCardUpdated(e.detail)}
		on:deleteRequest={(e) => {
			selectedCard = null;
			handleDelete(e.detail);
		}}
	/>
{/if}

<ConfirmDialog
	bind:show={showDeleteConfirm}
	title={$i18n.t('Delete business card?')}
	message={deleteTarget
		? `${$i18n.t('Permanently delete the card for')} "${deleteTarget.full_name}"?`
		: ''}
	on:confirm={confirmDelete}
	on:cancel={() => {
		deleteTarget = null;
		showDeleteConfirm = false;
	}}
/>
