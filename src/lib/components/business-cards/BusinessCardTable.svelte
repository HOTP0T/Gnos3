<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { page as pageStore } from '$app/stores';
	import { toast } from 'svelte-sonner';

	import {
		getBusinessCards,
		getBusinessCardCompanies,
		updateBusinessCard,
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

	let editingCell: { id: number; field: string } | null = null;
	let editValue: string = '';

	let showDeleteConfirm = false;
	let deleteTarget: BusinessCard | null = null;
	let reprocessingIds: Set<number> = new Set();
	let syncingIds: Set<number> = new Set();

	const COLUMNS: Array<{
		key: keyof BusinessCard;
		label: string;
		sortable: boolean;
		readonly?: boolean;
		minW: string;
	}> = [
		{ key: 'full_name', label: 'Name', sortable: true, minW: '12rem' },
		{ key: 'job_title', label: 'Job title', sortable: false, minW: '10rem' },
		{ key: 'company_name', label: 'Company', sortable: true, minW: '10rem' },
		{ key: 'email', label: 'Email', sortable: true, minW: '14rem' },
		{ key: 'phone', label: 'Phone', sortable: false, minW: '8rem' },
		{ key: 'mobile', label: 'Mobile', sortable: false, minW: '8rem' },
		{ key: 'website', label: 'Website', sortable: false, minW: '10rem' },
		{ key: 'k4mi_notes', label: 'Notes', sortable: false, readonly: true, minW: '16rem' }
	];

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

	const startEdit = (card: BusinessCard, field: string) => {
		editingCell = { id: card.id, field };
		const val = (card as any)[field];
		editValue = val == null ? '' : String(val);
	};

	const cancelEdit = () => {
		editingCell = null;
		editValue = '';
	};

	const saveEdit = async () => {
		if (!editingCell) return;
		const { id, field } = editingCell;
		const newVal = editValue.trim() || null;

		const card = cards.find((c) => c.id === id);
		if (card && (card as any)[field] === newVal) {
			cancelEdit();
			return;
		}

		try {
			const updated = await updateBusinessCard(localStorage.token, id, { [field]: newVal });
			const idx = cards.findIndex((c) => c.id === id);
			if (idx !== -1) cards[idx] = updated;
			cards = [...cards];
			toast.success($i18n.t('Saved'));
		} catch (err) {
			toast.error(`${err}`);
		}
		cancelEdit();
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

	onMount(() => {
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

		<div class="ml-auto text-xs text-gray-500 dark:text-gray-400">
			{total} {$i18n.t('cards')}
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
			<table class="w-full text-sm border-collapse">
				<thead class="text-left text-xs uppercase tracking-wide text-gray-500 dark:text-gray-400 bg-gray-50/70 dark:bg-gray-900/40">
					<tr>
						{#each COLUMNS as col}
							<th
								class="px-3 py-2.5 font-medium whitespace-nowrap border-b border-gray-200 dark:border-gray-800"
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
							class="px-3 py-2.5 font-medium whitespace-nowrap border-b border-gray-200 dark:border-gray-800"
							style="min-width: 7.5rem"
						>
							{$i18n.t('Status')}
						</th>
						<th
							class="px-3 py-2.5 font-medium whitespace-nowrap text-right border-b border-gray-200 dark:border-gray-800 sticky right-0 bg-gray-50/70 dark:bg-gray-900/40"
							style="min-width: 9rem"
						>
							{$i18n.t('Actions')}
						</th>
					</tr>
				</thead>
				<tbody>
					{#each cards as card, idx (card.id)}
						<tr
							class="group border-b border-gray-100 dark:border-gray-900 transition-colors {idx %
								2 ===
							0
								? 'bg-white dark:bg-transparent'
								: 'bg-gray-50/40 dark:bg-gray-900/20'} hover:bg-gray-100/70 dark:hover:bg-gray-800/40"
						>
							{#each COLUMNS as col}
								<td class="px-3 py-2 align-top" style="min-width: {col.minW}">
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
									{:else if editingCell && editingCell.id === card.id && editingCell.field === col.key}
										<input
											class="w-full px-1.5 py-1 text-sm rounded bg-white dark:bg-gray-900 outline-none ring-1 ring-blue-400 dark:ring-blue-500"
											bind:value={editValue}
											on:blur={saveEdit}
											on:keydown={(e) => {
												if (e.key === 'Enter') saveEdit();
												else if (e.key === 'Escape') cancelEdit();
											}}
											autofocus
										/>
									{:else}
										<button
											class="text-left w-full hover:bg-gray-100 dark:hover:bg-gray-800 rounded px-1.5 py-1 transition truncate block"
											on:click={() => startEdit(card, col.key)}
										>
											{#if (card as any)[col.key]}
												<span class="text-gray-800 dark:text-gray-200">{(card as any)[col.key]}</span>
											{:else}
												<span class="text-gray-400 dark:text-gray-600">—</span>
											{/if}
										</button>
									{/if}
								</td>
							{/each}
							<td class="px-3 py-2 align-top whitespace-nowrap">
								{#if card.processing_status === 'processing' || card.processing_status === 'pending'}
									<span
										class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
									>
										<Spinner className="size-3" />
										{$i18n.t('Processing')}
									</span>
								{:else if card.processing_status === 'failed'}
									<span
										class="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-300"
									>
										{$i18n.t('Failed')}
									</span>
								{:else if card.needs_review}
									<span
										class="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
									>
										{$i18n.t('Needs review')}
									</span>
								{:else}
									<span
										class="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300"
									>
										{$i18n.t('OK')}
									</span>
								{/if}
							</td>
							<td
								class="px-3 py-2 align-top whitespace-nowrap text-right sticky right-0 bg-inherit"
							>
								<div
									class="inline-flex items-center gap-2 text-gray-400 dark:text-gray-500 opacity-70 group-hover:opacity-100 transition"
								>
									{#if card.k4mi_document_id}
										<Tooltip content={$i18n.t('Open in K4mi')}>
											<a
												class="hover:text-gray-700 dark:hover:text-gray-200 transition p-1 -m-1"
												href={k4miHref(card.k4mi_document_id)}
												target="_blank"
												rel="noopener noreferrer"
												aria-label={$i18n.t('Open in K4mi')}
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
												on:click={() => handleSync(card)}
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
												on:click={() => handleReprocess(card)}
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
											on:click={() => handleDelete(card)}
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
