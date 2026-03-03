<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { page as pageStore } from '$app/stores';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';
	import { browser } from '$app/environment';

	import { getInvoices, updateInvoice, deleteInvoice } from '$lib/apis/invoices';

	import Pagination from '$lib/components/common/Pagination.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Badge from '$lib/components/common/Badge.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import FilterBar from '$lib/components/invoices/FilterBar.svelte';
	import ExportButton from '$lib/components/invoices/ExportButton.svelte';
	import DocumentPreviewModal from '$lib/components/invoices/DocumentPreviewModal.svelte';

	const i18n = getContext('i18n');

	// Data
	let invoices: any[] = [];
	let total = 0;
	let loading = true;

	// Pagination
	let page = 1;
	let perPage = 50;

	// Filters
	let filters: Record<string, any> = {};
	let showFilters = false;

	// Sorting — not sent to backend, done client-side for the current page
	// (Backend always sorts by invoice_date desc)

	// Editing
	let editingCell: { id: number; field: string } | null = null;
	let editValue: string = '';

	// Expanded rows
	let expandedRows: Set<number> = new Set();

	// Delete confirmation
	let showDeleteConfirm = false;
	let deleteTarget: any = null;

	// Document preview
	let showPreview = false;
	let previewInvoice: any = null;

	// Search
	let searchQuery = '';
	let searchDebounce: ReturnType<typeof setTimeout>;

	const EDITABLE_COLUMNS = [
		{ key: 'vendor_name', label: 'Vendor', type: 'text' },
		{ key: 'invoice_number', label: 'Invoice #', type: 'text' },
		{ key: 'invoice_date', label: 'Date', type: 'date' },
		{ key: 'due_date', label: 'Due Date', type: 'date' },
		{ key: 'currency', label: 'Cur', type: 'text', maxLength: 3 },
		{ key: 'subtotal', label: 'Subtotal', type: 'number', align: 'right' },
		{ key: 'tax_amount', label: 'Tax', type: 'number', align: 'right' },
		{ key: 'total_amount', label: 'Total', type: 'number', align: 'right' },
		{ key: 'amount_paid', label: 'Paid', type: 'number', align: 'right' },
		{ key: 'balance_due', label: 'Balance', type: 'number', align: 'right' },
		{ key: 'payment_terms', label: 'Terms', type: 'text' },
		{ key: 'po_number', label: 'PO #', type: 'text' },
		{ key: 'description', label: 'Description', type: 'text' }
	];

	const loadInvoices = async () => {
		loading = true;
		try {
			const params: Record<string, any> = {
				...filters,
				limit: perPage,
				offset: (page - 1) * perPage
			};
			if (searchQuery) {
				params.vendor = searchQuery;
			}
			const res = await getInvoices(localStorage.token, params);
			if (res) {
				invoices = res.invoices;
				total = res.total;
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const handleFilter = (e: CustomEvent) => {
		filters = e.detail;
		page = 1;
		loadInvoices();
	};

	const handleSearch = () => {
		clearTimeout(searchDebounce);
		searchDebounce = setTimeout(() => {
			page = 1;
			loadInvoices();
		}, 300);
	};

	// Inline editing
	const startEdit = (invoice: any, field: string) => {
		editingCell = { id: invoice.id, field };
		const val = invoice[field];
		if (val === null || val === undefined) {
			editValue = '';
		} else if (field === 'invoice_date' || field === 'due_date') {
			editValue = val ? dayjs(val).format('YYYY-MM-DD') : '';
		} else {
			editValue = String(val);
		}
	};

	const saveEdit = async () => {
		if (!editingCell) return;

		const { id, field } = editingCell;
		const invoice = invoices.find((inv) => inv.id === id);
		if (!invoice) return;

		// Build update payload
		let updateVal: any = editValue;
		const col = EDITABLE_COLUMNS.find((c) => c.key === field);

		if (col?.type === 'number') {
			updateVal = editValue === '' ? null : parseFloat(editValue);
		} else if (col?.type === 'date') {
			updateVal = editValue === '' ? null : editValue;
		} else {
			updateVal = editValue === '' ? null : editValue;
		}

		// Skip if value hasn't changed
		const oldVal = invoice[field];
		const oldStr = oldVal === null || oldVal === undefined ? '' : String(oldVal);
		const newStr = updateVal === null ? '' : String(updateVal);
		if (oldStr === newStr || (col?.type === 'date' && dayjs(oldStr).format('YYYY-MM-DD') === newStr)) {
			editingCell = null;
			return;
		}

		try {
			const res = await updateInvoice(localStorage.token, id, { [field]: updateVal });
			if (res) {
				// Update local data
				const idx = invoices.findIndex((inv) => inv.id === id);
				if (idx !== -1) {
					invoices[idx] = res;
					invoices = invoices;
				}
				toast.success($i18n.t('Updated'));
			}
		} catch (err) {
			toast.error(`${err}`);
		}

		editingCell = null;
	};

	const cancelEdit = () => {
		editingCell = null;
	};

	const handleKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Enter') {
			saveEdit();
		} else if (e.key === 'Escape') {
			cancelEdit();
		}
	};

	// Delete
	const confirmDelete = (invoice: any) => {
		deleteTarget = invoice;
		showDeleteConfirm = true;
	};

	const handleDelete = async () => {
		if (!deleteTarget) return;
		try {
			await deleteInvoice(localStorage.token, deleteTarget.id);
			toast.success($i18n.t('Invoice deleted'));
			if (invoices.length === 1 && page > 1) {
				page -= 1;
			}
			loadInvoices();
		} catch (err) {
			toast.error(`${err}`);
		}
		deleteTarget = null;
	};

	// Expand/collapse
	const toggleExpand = (id: number) => {
		if (expandedRows.has(id)) {
			expandedRows.delete(id);
		} else {
			expandedRows.add(id);
		}
		expandedRows = expandedRows;
	};

	// Document preview
	const openPreview = (invoice: any) => {
		previewInvoice = invoice;
		showPreview = true;
	};

	// Cell display helpers
	const formatCurrency = (val: any) => {
		if (val === null || val === undefined) return '-';
		return parseFloat(val).toLocaleString(undefined, {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2
		});
	};

	const formatDate = (val: any) => {
		if (!val) return '-';
		return dayjs(val).format('YYYY-MM-DD');
	};

	const displayValue = (invoice: any, col: any) => {
		const val = invoice[col.key];
		if (val === null || val === undefined) return '-';
		if (col.type === 'number') return formatCurrency(val);
		if (col.type === 'date') return formatDate(val);
		return String(val);
	};

	let mounted = false;

	// React to page changes only after initial load
	$: if (mounted && page) {
		loadInvoices();
	}

	onMount(() => {
		// Parse URL query params for initial filters (e.g., ?needs_review=true&vendor=...)
		if (browser) {
			const urlParams = $pageStore.url.searchParams;
			const urlVendor = urlParams.get('vendor');
			const urlNeedsReview = urlParams.get('needs_review');

			if (urlVendor) {
				searchQuery = urlVendor;
			}
			if (urlNeedsReview === 'true') {
				filters = { ...filters, needs_review: true };
				showFilters = true;
			}
		}

		loadInvoices().then(() => {
			mounted = true;
		});
	});
</script>

<ConfirmDialog
	bind:show={showDeleteConfirm}
	on:confirm={handleDelete}
	title={$i18n.t('Delete Invoice')}
	message={$i18n.t('Are you sure you want to delete this invoice? This action cannot be undone.')}
/>

<DocumentPreviewModal bind:show={showPreview} invoice={previewInvoice} />

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Invoices')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{total}
			</div>
		</div>

		<div class="flex gap-1">
			<div class="flex w-full space-x-2">
				<!-- Search -->
				<div class="flex flex-1">
					<div class="self-center ml-1 mr-3">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="2"
							stroke="currentColor"
							class="size-3.5"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								d="m21 21-5.197-5.197m0 0A7.5 7.5 0 1 0 5.196 5.196a7.5 7.5 0 0 0 10.607 10.607Z"
							/>
						</svg>
					</div>
					<input
						class="w-full text-sm pr-4 py-1 rounded-r-xl outline-hidden bg-transparent dark:text-gray-200"
						bind:value={searchQuery}
						on:input={handleSearch}
						placeholder={$i18n.t('Search vendors...')}
					/>
				</div>

				<!-- Filter toggle -->
				<Tooltip content={$i18n.t('Filters')}>
					<button
						class="p-2 rounded-xl hover:bg-gray-100 dark:bg-gray-900 dark:hover:bg-gray-850 transition font-medium text-sm"
						on:click={() => {
							showFilters = !showFilters;
						}}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="2"
							stroke="currentColor"
							class="size-3.5"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 0 1-.659 1.591l-5.432 5.432a2.25 2.25 0 0 0-.659 1.591v2.927a2.25 2.25 0 0 1-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 0 0-.659-1.591L3.659 7.409A2.25 2.25 0 0 1 3 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0 1 12 3Z"
							/>
						</svg>
					</button>
				</Tooltip>

				<!-- Export -->
				<ExportButton {invoices} />
			</div>
		</div>
	</div>

	<!-- Filters -->
	<FilterBar bind:show={showFilters} on:filter={handleFilter} />

	<!-- Table -->
	{#if loading && invoices.length === 0}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if invoices.length === 0}
		<div class="flex justify-center my-10 text-sm text-gray-500">
			{$i18n.t('No invoices found')}
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table
				class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto max-w-full"
			>
				<thead
					class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200"
				>
					<tr class="border-b-[1.5px] border-gray-50 dark:border-gray-850/30">
						{#each EDITABLE_COLUMNS as col}
							<th
								scope="col"
								class="px-2.5 py-2 select-none whitespace-nowrap {col.align === 'right'
									? 'text-right'
									: ''}"
							>
								{$i18n.t(col.label)}
							</th>
						{/each}
						<th scope="col" class="px-2.5 py-2 select-none">{$i18n.t('Status')}</th>
						<th scope="col" class="px-2.5 py-2 select-none"
							>{$i18n.t('Confidence')}</th
						>
						<th scope="col" class="px-2.5 py-2 select-none text-right"
							>{$i18n.t('Actions')}</th
						>
					</tr>
				</thead>
				<tbody>
					{#each invoices as invoice (invoice.id)}
						<tr
							class="bg-white dark:bg-gray-900 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
						>
							{#each EDITABLE_COLUMNS as col}
								<td
									class="px-3 py-1 {col.align === 'right'
										? 'text-right'
										: ''} whitespace-nowrap cursor-pointer"
									on:click={() => startEdit(invoice, col.key)}
								>
									{#if editingCell?.id === invoice.id && editingCell?.field === col.key}
										{#if col.type === 'date'}
											<input
												class="w-full text-xs bg-transparent outline-hidden border-b border-blue-500 dark:text-gray-200"
												type="date"
												bind:value={editValue}
												on:keydown={handleKeydown}
												on:blur={saveEdit}
												autofocus
											/>
										{:else if col.type === 'number'}
											<input
												class="w-full text-xs bg-transparent outline-hidden border-b border-blue-500 dark:text-gray-200 {col.align ===
												'right'
													? 'text-right'
													: ''}"
												type="number"
												step="0.01"
												bind:value={editValue}
												on:keydown={handleKeydown}
												on:blur={saveEdit}
												autofocus
											/>
										{:else}
											<input
												class="w-full text-xs bg-transparent outline-hidden border-b border-blue-500 dark:text-gray-200"
												type="text"
												maxlength={col.maxLength}
												bind:value={editValue}
												on:keydown={handleKeydown}
												on:blur={saveEdit}
												autofocus
											/>
										{/if}
									{:else}
										<span class="dark:text-gray-300">
											{displayValue(invoice, col)}
										</span>
									{/if}
								</td>
							{/each}

							<!-- Status -->
							<td class="px-3 py-1">
								<Badge
									type={invoice.processing_status === 'completed'
										? 'success'
										: invoice.processing_status === 'failed'
											? 'error'
											: 'muted'}
									content={invoice.processing_status}
								/>
							</td>

							<!-- Confidence -->
							<td class="px-3 py-1">
								{#if invoice.confidence_score !== null}
									<Badge
										type={parseFloat(invoice.confidence_score) >= 0.7
											? 'success'
											: parseFloat(invoice.confidence_score) >= 0.4
												? 'warning'
												: 'error'}
										content={`${(parseFloat(invoice.confidence_score) * 100).toFixed(0)}%`}
									/>
								{:else}
									<span class="text-gray-400">-</span>
								{/if}
							</td>

							<!-- Actions -->
							<td class="px-3 py-1 text-right whitespace-nowrap">
								<div class="flex items-center justify-end gap-1">
									<!-- Preview -->
									<Tooltip content={$i18n.t('Preview')}>
										<button
											class="p-1 hover:bg-gray-100 dark:hover:bg-gray-850 rounded-lg transition"
											on:click={() => openPreview(invoice)}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												fill="none"
												viewBox="0 0 24 24"
												stroke-width="2"
												stroke="currentColor"
												class="size-3.5"
											>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z"
												/>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z"
												/>
											</svg>
										</button>
									</Tooltip>

									<!-- Expand -->
									<Tooltip content={$i18n.t('Details')}>
										<button
											class="p-1 hover:bg-gray-100 dark:hover:bg-gray-850 rounded-lg transition"
											on:click={() => toggleExpand(invoice.id)}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												fill="none"
												viewBox="0 0 24 24"
												stroke-width="2"
												stroke="currentColor"
												class="size-3.5 transition-transform {expandedRows.has(
													invoice.id
												)
													? 'rotate-180'
													: ''}"
											>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="m19.5 8.25-7.5 7.5-7.5-7.5"
												/>
											</svg>
										</button>
									</Tooltip>

									<!-- Delete -->
									<Tooltip content={$i18n.t('Delete')}>
										<button
											class="p-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition text-red-500"
											on:click={() => confirmDelete(invoice)}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												fill="none"
												viewBox="0 0 24 24"
												stroke-width="2"
												stroke="currentColor"
												class="size-3.5"
											>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
												/>
											</svg>
										</button>
									</Tooltip>
								</div>
							</td>
						</tr>

						<!-- Expanded Detail Row -->
						{#if expandedRows.has(invoice.id)}
							<tr class="bg-gray-50/50 dark:bg-gray-850/30">
								<td colspan={EDITABLE_COLUMNS.length + 3} class="px-4 py-3">
									<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
										<!-- Left: Raw extraction & errors -->
										<div class="space-y-3">
											{#if invoice.extraction_raw}
												<div>
													<div class="text-xs font-medium text-gray-500 mb-1">
														{$i18n.t('Raw Extraction')}
													</div>
													<pre
														class="text-xs bg-gray-50 dark:bg-gray-850 rounded-lg p-3 overflow-auto max-h-48 dark:text-gray-300">{JSON.stringify(
															invoice.extraction_raw,
															null,
															2
														)}</pre>
												</div>
											{/if}

											{#if invoice.extraction_errors?.length}
												<div>
													<div class="text-xs font-medium text-gray-500 mb-1">
														{$i18n.t('Extraction Errors')}
													</div>
													<div class="space-y-1">
														{#each invoice.extraction_errors as error}
															<Badge type="error" content={error} />
														{/each}
													</div>
												</div>
											{/if}
										</div>

										<!-- Right: Line items, notes, tags -->
										<div class="space-y-3">
											{#if invoice.line_items?.length}
												<div>
													<div class="text-xs font-medium text-gray-500 mb-1">
														{$i18n.t('Line Items')}
													</div>
													<table
														class="w-full text-xs text-left text-gray-500 dark:text-gray-400"
													>
														<thead>
															<tr
																class="border-b border-gray-100 dark:border-gray-850"
															>
																<th class="py-1 pr-2"
																	>{$i18n.t('Description')}</th
																>
																<th class="py-1 pr-2 text-right"
																	>{$i18n.t('Qty')}</th
																>
																<th class="py-1 pr-2 text-right"
																	>{$i18n.t('Price')}</th
																>
																<th class="py-1 text-right"
																	>{$i18n.t('Amount')}</th
																>
															</tr>
														</thead>
														<tbody>
															{#each invoice.line_items as item}
																<tr>
																	<td class="py-0.5 pr-2 dark:text-gray-300"
																		>{item.description ??
																			'-'}</td
																	>
																	<td
																		class="py-0.5 pr-2 text-right dark:text-gray-300"
																		>{item.quantity ?? '-'}</td
																	>
																	<td
																		class="py-0.5 pr-2 text-right dark:text-gray-300"
																		>{item.unit_price != null
																			? formatCurrency(
																					item.unit_price
																				)
																			: '-'}</td
																	>
																	<td
																		class="py-0.5 text-right dark:text-gray-300"
																		>{item.amount != null
																			? formatCurrency(
																					item.amount
																				)
																			: '-'}</td
																	>
																</tr>
															{/each}
														</tbody>
													</table>
												</div>
											{/if}

											{#if invoice.k4mi_notes?.length}
												<div>
													<div class="text-xs font-medium text-gray-500 mb-1">
														{$i18n.t('Notes')}
													</div>
													<div class="space-y-1">
														{#each invoice.k4mi_notes as note}
															<div
																class="text-xs bg-gray-50 dark:bg-gray-850 rounded-lg p-2 dark:text-gray-300"
															>
																{note.text || note.note || ''}
															</div>
														{/each}
													</div>
												</div>
											{/if}

											{#if invoice.k4mi_tags?.length}
												<div>
													<div class="text-xs font-medium text-gray-500 mb-1">
														{$i18n.t('Tags')}
													</div>
													<div class="flex flex-wrap gap-1">
														{#each invoice.k4mi_tags as tag}
															<span
																class="px-1.5 py-0.5 rounded-xl bg-gray-100 dark:bg-gray-850 text-xs dark:text-gray-200"
															>
																{tag}
															</span>
														{/each}
													</div>
												</div>
											{/if}

											<div class="flex gap-3 text-xs text-gray-400">
												<span
													>K4mi ID: {invoice.k4mi_document_id}</span
												>
												{#if invoice.extraction_model}
													<span
														>Model: {invoice.extraction_model}</span
													>
												{/if}
												{#if invoice.user_corrected}
													<Badge
														type="info"
														content={$i18n.t('User Corrected')}
													/>
												{/if}
											</div>
										</div>
									</div>
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		</div>

		<!-- Pagination -->
		<div class="flex items-center justify-between mt-2">
			<div class="flex items-center gap-2">
				<span class="text-xs text-gray-500">{$i18n.t('Per page')}:</span>
				<select
					class="text-xs bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2 py-1"
					bind:value={perPage}
					on:change={() => {
						page = 1;
						loadInvoices();
					}}
				>
					<option value={25}>25</option>
					<option value={50}>50</option>
					<option value={100}>100</option>
				</select>
			</div>

			<Pagination bind:page count={total} {perPage} />
		</div>
	{/if}
</div>
