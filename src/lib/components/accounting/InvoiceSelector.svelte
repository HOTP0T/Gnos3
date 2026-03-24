<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import { getInvoiceList } from '$lib/apis/accounting';
	import { K4MI_BASE_URL } from '$lib/constants';

	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;

	let loading = false;
	let invoices: any[] = [];
	let total = 0;
	let search = '';
	let searchDebounce: ReturnType<typeof setTimeout>;
	let offset = 0;
	const limit = 20;

	// Filter state
	let statusFilter: string = 'all'; // 'all' | 'review' | 'ready'
	let sortBy: string = 'date_desc'; // 'date_desc' | 'amount_desc' | 'vendor_asc'

	// Preview popover state
	let previewInvoice: any = null;
	let previewPos = { x: 0, y: 0 };

	const sortOptions: Record<string, { sort_by: string; sort_dir: string }> = {
		date_desc: { sort_by: 'invoice_date', sort_dir: 'desc' },
		amount_desc: { sort_by: 'total_amount', sort_dir: 'desc' },
		vendor_asc: { sort_by: 'vendor_name', sort_dir: 'asc' }
	};

	const loadInvoices = async (resetOffset = true) => {
		if (resetOffset) offset = 0;
		loading = true;
		try {
			const params: Record<string, any> = { limit, offset };
			if (search) params.q = search;

			// Status filter
			if (statusFilter === 'review') {
				params.needs_review = true;
			} else if (statusFilter === 'ready') {
				params.needs_review = false;
			}

			// Sort
			const sort = sortOptions[sortBy] ?? sortOptions.date_desc;
			params.sort_by = sort.sort_by;
			params.sort_dir = sort.sort_dir;

			const res = await getInvoiceList(params);
			if (res && res.invoices) {
				invoices = res.invoices;
				total = res.total ?? invoices.length;
			} else if (Array.isArray(res)) {
				invoices = res;
				total = res.length;
			} else {
				invoices = [];
				total = 0;
			}
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load invoices')}: ${err}`);
		}
		loading = false;
	};

	const handleSearch = () => {
		clearTimeout(searchDebounce);
		searchDebounce = setTimeout(() => loadInvoices(), 300);
	};

	const selectInvoice = (invoice: any) => {
		dispatch('select', invoice);
		show = false;
	};

	const formatCurrency = (val: any) => {
		if (val === null || val === undefined) return '-';
		return parseFloat(val).toLocaleString(undefined, {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2
		});
	};

	const truncate = (text: string | null | undefined, maxLen: number) => {
		if (!text) return '-';
		return text.length > maxLen ? text.slice(0, maxLen) + '...' : text;
	};

	const showPreview = (e: MouseEvent, invoice: any) => {
		e.stopPropagation();
		const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
		// Position popover to the left of the icon, vertically centered
		previewPos = {
			x: rect.left,
			y: rect.top + rect.height / 2
		};
		previewInvoice = previewInvoice?.id === invoice.id ? null : invoice;
	};

	const closePreview = () => {
		previewInvoice = null;
	};

	const getLineItemsCount = (invoice: any): number => {
		if (!invoice.line_items) return 0;
		if (Array.isArray(invoice.line_items)) return invoice.line_items.length;
		if (typeof invoice.line_items === 'object' && invoice.line_items.items) {
			return Array.isArray(invoice.line_items.items) ? invoice.line_items.items.length : 0;
		}
		return 0;
	};

	$: if (show) {
		previewInvoice = null;
		loadInvoices();
	}
</script>

{#if show}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		class="fixed inset-0 bg-black/60 flex items-center justify-center z-[99999]"
		on:mousedown={() => {
			previewInvoice = null;
			show = false;
		}}
	>
		<div
			class="bg-white dark:bg-gray-950 rounded-2xl w-[60rem] max-w-[95vw] max-h-[80vh] flex flex-col shadow-2xl border border-gray-200 dark:border-gray-800"
			on:mousedown|stopPropagation
		>
			<!-- Header -->
			<div
				class="px-5 py-4 border-b border-gray-100 dark:border-gray-800 flex items-center justify-between"
			>
				<h2 class="text-lg font-medium dark:text-gray-200">
					{$i18n.t('Select Invoice')}
				</h2>
				<button
					class="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition"
					on:click={() => (show = false)}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						stroke-width="2"
						stroke="currentColor"
						class="size-5 dark:text-gray-300"
					>
						<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
					</svg>
				</button>
			</div>

			<!-- Search + Filters -->
			<div class="px-5 py-3 border-b border-gray-100 dark:border-gray-800 space-y-2">
				<input
					type="text"
					class="w-full text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 px-3 py-2 outline-hidden focus:border-blue-500 transition"
					placeholder={$i18n.t('Search by vendor, invoice number, client, description, or amount...')}
					bind:value={search}
					on:input={handleSearch}
				/>
				<div class="flex items-center gap-3">
					<!-- Status filter -->
					<div class="flex items-center gap-1.5">
						<span class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Status')}:</span>
						<select
							class="text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 px-2 py-1.5 outline-hidden focus:border-blue-500 transition"
							bind:value={statusFilter}
							on:change={() => loadInvoices()}
						>
							<option value="all">{$i18n.t('All')}</option>
							<option value="review">{$i18n.t('Needs Review')}</option>
							<option value="ready">{$i18n.t('Ready')}</option>
						</select>
					</div>

					<!-- Sort by -->
					<div class="flex items-center gap-1.5">
						<span class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Sort')}:</span>
						<select
							class="text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 px-2 py-1.5 outline-hidden focus:border-blue-500 transition"
							bind:value={sortBy}
							on:change={() => loadInvoices()}
						>
							<option value="date_desc">{$i18n.t('Date (newest)')}</option>
							<option value="amount_desc">{$i18n.t('Amount (highest)')}</option>
							<option value="vendor_asc">{$i18n.t('Vendor (A-Z)')}</option>
						</select>
					</div>

					<!-- Result count -->
					{#if !loading && total > 0}
						<span class="text-xs text-gray-400 ml-auto">
							{total}
							{$i18n.t('invoices')}
						</span>
					{/if}
				</div>
			</div>

			<!-- Invoice List -->
			<div class="flex-1 overflow-y-auto px-5 py-2 relative">
				{#if loading}
					<div class="flex justify-center py-8">
						<Spinner className="size-5" />
					</div>
				{:else if invoices.length === 0}
					<div class="text-center py-8 text-gray-400 text-sm">
						{$i18n.t('No invoices found')}
					</div>
				{:else}
					<table class="w-full text-sm text-left">
						<thead
							class="text-xs text-gray-500 dark:text-gray-400 uppercase border-b border-gray-200 dark:border-gray-700"
						>
							<tr>
								<th class="py-2 pr-2">{$i18n.t('Invoice #')}</th>
								<th class="py-2 pr-2">{$i18n.t('Vendor')}</th>
								<th class="py-2 pr-2">{$i18n.t('Client')}</th>
								<th class="py-2 pr-2">{$i18n.t('Description')}</th>
								<th class="py-2 pr-2">{$i18n.t('Date')}</th>
								<th class="py-2 pr-2">{$i18n.t('Due')}</th>
								<th class="py-2 pr-2 text-right">{$i18n.t('Amount')}</th>
								<th class="py-2 pr-2">{$i18n.t('Cur.')}</th>
								<th class="py-2 pr-2">{$i18n.t('Status')}</th>
								<th class="py-2 w-16"></th>
							</tr>
						</thead>
						<tbody>
							{#each invoices as invoice}
								<tr
									class="border-b border-gray-100 dark:border-gray-800 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition cursor-pointer"
									on:click={() => selectInvoice(invoice)}
								>
									<td class="py-2 pr-2 font-mono dark:text-gray-300 whitespace-nowrap">
										{invoice.invoice_number ?? `#${invoice.id}`}
										{#if invoice.k4mi_document_id}
											<a
												href="{K4MI_BASE_URL}/documents/{invoice.k4mi_document_id}/details"
												target="_blank"
												rel="noopener"
												class="ml-1 text-blue-500 hover:text-blue-700 inline-block align-middle"
												title={$i18n.t('Open in K4mi')}
												on:click|stopPropagation
											>
												<svg
													xmlns="http://www.w3.org/2000/svg"
													fill="none"
													viewBox="0 0 24 24"
													stroke-width="1.5"
													stroke="currentColor"
													class="w-3 h-3"
													><path
														stroke-linecap="round"
														stroke-linejoin="round"
														d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25"
													/></svg
												>
											</a>
										{/if}
									</td>
									<td
										class="py-2 pr-2 dark:text-gray-300 max-w-[120px] truncate"
										title={invoice.vendor_name ?? ''}
									>
										{invoice.vendor_name ?? '-'}
									</td>
									<td
										class="py-2 pr-2 dark:text-gray-300 max-w-[100px] truncate"
										title={invoice.client_name ?? ''}
									>
										{invoice.client_name ?? '-'}
									</td>
									<td
										class="py-2 pr-2 text-gray-500 dark:text-gray-400 max-w-[120px] truncate"
										title={invoice.description ?? ''}
									>
										{truncate(invoice.description, 30)}
									</td>
									<td class="py-2 pr-2 text-gray-500 whitespace-nowrap">
										{invoice.invoice_date
											? dayjs(invoice.invoice_date).format('YYYY-MM-DD')
											: '-'}
									</td>
									<td class="py-2 pr-2 text-gray-500 whitespace-nowrap">
										{invoice.due_date
											? dayjs(invoice.due_date).format('YYYY-MM-DD')
											: '-'}
									</td>
									<td class="py-2 pr-2 text-right font-medium dark:text-gray-300 whitespace-nowrap">
										{formatCurrency(invoice.total_amount)}
									</td>
									<td class="py-2 pr-2 text-gray-400 text-[10px] uppercase whitespace-nowrap">
										{invoice.currency ?? 'USD'}
									</td>
									<td class="py-2 pr-2">
										{#if invoice.needs_review}
											<span class="text-xs text-yellow-600 dark:text-yellow-400"
												>{$i18n.t('Review')}</span
											>
										{:else}
											<span class="text-xs text-green-600 dark:text-green-400"
												>{$i18n.t('Ready')}</span
											>
										{/if}
									</td>
									<td class="py-2 text-right whitespace-nowrap">
										<!-- Info / preview button -->
										<button
											class="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition mr-1 relative"
											title={$i18n.t('Invoice details')}
											on:click|stopPropagation={(e) => showPreview(e, invoice)}
										>
											<svg
												xmlns="http://www.w3.org/2000/svg"
												fill="none"
												viewBox="0 0 24 24"
												stroke-width="1.5"
												stroke="currentColor"
												class="w-3.5 h-3.5 text-gray-500 dark:text-gray-400"
											>
												<path
													stroke-linecap="round"
													stroke-linejoin="round"
													d="m11.25 11.25.041-.02a.75.75 0 0 1 1.063.852l-.708 2.836a.75.75 0 0 0 1.063.853l.041-.021M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9-3.75h.008v.008H12V8.25Z"
												/>
											</svg>
										</button>
										<!-- Select button -->
										<button
											class="px-2.5 py-1 text-xs font-medium rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition"
											on:click|stopPropagation={() => selectInvoice(invoice)}
										>
											{$i18n.t('Select')}
										</button>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>

					<!-- Pagination -->
					{#if total > limit}
						<div
							class="flex justify-between items-center mt-3 pt-3 border-t border-gray-100 dark:border-gray-800"
						>
							<div class="text-xs text-gray-400">
								{$i18n.t('Showing')}
								{offset + 1}-{offset + invoices.length}
								{$i18n.t('of')}
								{total}
							</div>
							<div class="flex gap-2">
								<button
									class="px-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition disabled:opacity-40"
									on:click={() => {
										offset = Math.max(0, offset - limit);
										loadInvoices(false);
									}}
									disabled={offset === 0}
								>
									{$i18n.t('Previous')}
								</button>
								<button
									class="px-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition disabled:opacity-40"
									on:click={() => {
										offset += limit;
										loadInvoices(false);
									}}
									disabled={offset + invoices.length >= total}
								>
									{$i18n.t('Next')}
								</button>
							</div>
						</div>
					{/if}
				{/if}
			</div>
		</div>

		<!-- Preview Popover (rendered outside the modal scroll container) -->
		{#if previewInvoice}
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<!-- svelte-ignore a11y-no-static-element-interactions -->
			<div
				class="fixed bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-xl shadow-2xl p-4 w-[22rem] z-[100000] text-sm"
				style="top: {Math.min(previewPos.y, window.innerHeight - 360)}px; left: {Math.max(8, previewPos.x - 360)}px;"
				on:mousedown|stopPropagation
			>
				<div class="flex items-center justify-between mb-3">
					<h3 class="font-medium dark:text-gray-200 text-sm">
						{$i18n.t('Invoice Details')}
					</h3>
					<button
						class="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition"
						on:click|stopPropagation={closePreview}
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="2"
							stroke="currentColor"
							class="w-4 h-4 dark:text-gray-400"
						>
							<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
						</svg>
					</button>
				</div>
				<div class="space-y-1.5 text-xs">
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Invoice #')}</span>
						<span class="dark:text-gray-200 font-mono">
							{previewInvoice.invoice_number ?? `#${previewInvoice.id}`}
						</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Vendor')}</span>
						<span class="dark:text-gray-200 text-right max-w-[160px] truncate">
							{previewInvoice.vendor_name ?? '-'}
						</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Client')}</span>
						<span class="dark:text-gray-200 text-right max-w-[160px] truncate">
							{previewInvoice.client_name ?? '-'}
						</span>
					</div>
					{#if previewInvoice.description}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400 shrink-0 mr-2"
								>{$i18n.t('Description')}</span
							>
							<span
								class="dark:text-gray-200 text-right max-w-[180px] truncate"
								title={previewInvoice.description}
							>
								{previewInvoice.description}
							</span>
						</div>
					{/if}
					<div class="border-t border-gray-100 dark:border-gray-800 my-1.5"></div>
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Date')}</span>
						<span class="dark:text-gray-200">
							{previewInvoice.invoice_date
								? dayjs(previewInvoice.invoice_date).format('YYYY-MM-DD')
								: '-'}
						</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Due Date')}</span>
						<span class="dark:text-gray-200">
							{previewInvoice.due_date
								? dayjs(previewInvoice.due_date).format('YYYY-MM-DD')
								: '-'}
						</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Amount')}</span>
						<span class="dark:text-gray-200 font-medium">
							{formatCurrency(previewInvoice.total_amount)}
							<span class="text-gray-400 text-[10px] uppercase ml-0.5"
								>{previewInvoice.currency ?? 'USD'}</span
							>
						</span>
					</div>
					{#if previewInvoice.subtotal !== null && previewInvoice.subtotal !== undefined}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Subtotal')}</span>
							<span class="dark:text-gray-200">{formatCurrency(previewInvoice.subtotal)}</span>
						</div>
					{/if}
					{#if previewInvoice.tax_amount !== null && previewInvoice.tax_amount !== undefined}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400">
								{$i18n.t('Tax')}
								{#if previewInvoice.tax_inclusive}
									<span class="text-[10px] text-gray-400">({$i18n.t('incl.')})</span>
								{/if}
							</span>
							<span class="dark:text-gray-200"
								>{formatCurrency(previewInvoice.tax_amount)}</span
							>
						</div>
					{/if}
					<div class="border-t border-gray-100 dark:border-gray-800 my-1.5"></div>
					{#if previewInvoice.po_number}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400">{$i18n.t('PO #')}</span>
							<span class="dark:text-gray-200 font-mono">{previewInvoice.po_number}</span>
						</div>
					{/if}
					{#if previewInvoice.business_unit}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Business Unit')}</span>
							<span class="dark:text-gray-200">{previewInvoice.business_unit}</span>
						</div>
					{/if}
					{#if previewInvoice.payment_terms}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Payment Terms')}</span>
							<span class="dark:text-gray-200">{previewInvoice.payment_terms}</span>
						</div>
					{/if}
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Line Items')}</span>
						<span class="dark:text-gray-200">{getLineItemsCount(previewInvoice)}</span>
					</div>
					<div class="flex justify-between">
						<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Status')}</span>
						<span>
							{#if previewInvoice.needs_review}
								<span class="text-yellow-600 dark:text-yellow-400"
									>{$i18n.t('Needs Review')}</span
								>
							{:else}
								<span class="text-green-600 dark:text-green-400">{$i18n.t('Ready')}</span>
							{/if}
						</span>
					</div>
					{#if previewInvoice.confidence_score !== null && previewInvoice.confidence_score !== undefined}
						<div class="flex justify-between">
							<span class="text-gray-500 dark:text-gray-400">{$i18n.t('Confidence')}</span>
							<span class="dark:text-gray-200"
								>{Math.round(parseFloat(previewInvoice.confidence_score) * 100)}%</span
							>
						</div>
					{/if}

					<!-- K4mi document link -->
					{#if previewInvoice.k4mi_document_id}
						<div class="border-t border-gray-100 dark:border-gray-800 my-1.5"></div>
						<a
							href="{K4MI_BASE_URL}/documents/{previewInvoice.k4mi_document_id}/details"
							target="_blank"
							rel="noopener"
							class="flex items-center gap-1.5 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition mt-1"
							on:click|stopPropagation
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
								class="w-3.5 h-3.5"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25"
								/>
							</svg>
							{$i18n.t('View full document in K4mi')}
						</a>
					{/if}
				</div>
			</div>
		{/if}
	</div>
{/if}
