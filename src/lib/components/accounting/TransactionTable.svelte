<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { page as pageStore } from '$app/stores';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';
	import { browser } from '$app/environment';

	import {
		getTransactions,
		deleteTransaction,
		postTransaction,
		voidTransaction,
		getAccounts,
		bulkPostTransactions,
		bulkDeleteDrafts,
		getAccountingAiStatus,
		aiValidateTransaction
	} from '$lib/apis/accounting';
	import { getInvoice } from '$lib/apis/invoices';
	import { convertAmount } from '$lib/utils/currency';

	import Pagination from '$lib/components/common/Pagination.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import TransactionFormModal from '$lib/components/accounting/TransactionFormModal.svelte';
	import DocumentPreviewModal from '$lib/components/invoices/DocumentPreviewModal.svelte';

	const i18n = getContext('i18n');
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');

	export let companyId: number;

	// ─── Data ────────────────────────────────────────────────────────────────────
	let transactions: any[] = [];
	let total = 0;
	let loading = true;
	let accounts: any[] = [];

	// ─── Pagination ──────────────────────────────────────────────────────────────
	let currentPage = 1;
	const perPage = 20;

	// ─── Filters ─────────────────────────────────────────────────────────────────
	let filterType = '';
	let filterStatus = '';
	let filterDateFrom = '';
	let filterDateTo = '';
	let filterUnbalanced = false;
	let searchQuery = '';
	let searchDebounce: ReturnType<typeof setTimeout>;

	// ─── Expanded rows ───────────────────────────────────────────────────────────
	let expandedRows: Set<number> = new Set();

	// ─── AI validation ───────────────────────────────────────────────────────────
	let aiAvailable = false;
	let validatingId: number | null = null;

	// ─── Bulk selection ──────────────────────────────────────────────────────────
	let selectedIds: Set<number> = new Set();
	$: filteredTransactions = filterUnbalanced ? transactions.filter((t) => isUnbalanced(t)) : transactions;
	$: draftTransactions = filteredTransactions.filter((t) => t.status === 'draft');
	$: allDraftsSelected = draftTransactions.length > 0 && draftTransactions.every((t) => selectedIds.has(t.id));

	const toggleSelectAll = () => {
		if (allDraftsSelected) selectedIds = new Set();
		else selectedIds = new Set(draftTransactions.map((t) => t.id));
	};
	const toggleSelect = (id: number) => {
		if (selectedIds.has(id)) selectedIds.delete(id);
		else selectedIds.add(id);
		selectedIds = selectedIds;
	};

	const handleBulkPost = async () => {
		if (selectedIds.size === 0) return;
		try {
			const result = await bulkPostTransactions({ company_id: companyId });
			toast.success(`${result?.posted ?? 0} ${$i18n.t('entries posted')}`);
			selectedIds = new Set();
			await loadTransactions();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	const handleBulkDelete = async () => {
		if (selectedIds.size === 0) return;
		try {
			const result = await bulkDeleteDrafts(companyId, [...selectedIds]);
			toast.success(`${result?.deleted ?? 0} ${$i18n.t('drafts deleted')}`);
			selectedIds = new Set();
			await loadTransactions();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	// ─── Modal state ─────────────────────────────────────────────────────────────
	let showFormModal = false;
	let editTransaction: any = null;

	// ─── Invoice preview ─────────────────────────────────────────────────────────
	let showPreview = false;
	let previewInvoice: any = null;
	let previewLoading = false;

	const openInvoicePreview = async (invoiceId: number) => {
		previewLoading = true;
		try {
			const inv = await getInvoice('', invoiceId);
			previewInvoice = inv;
			showPreview = true;
		} catch (err) {
			toast.error($i18n.t('Failed to load invoice'));
		}
		previewLoading = false;
	};

	// ─── Delete confirmation ─────────────────────────────────────────────────────
	let showDeleteConfirm = false;
	let deleteTarget: any = null;

	// ─── Post/Void confirmation ──────────────────────────────────────────────────
	let showPostConfirm = false;
	let postTarget: any = null;
	let showVoidConfirm = false;
	let voidTarget: any = null;

	// ─── Badge helpers ───────────────────────────────────────────────────────────
	const STATUS_CLASSES: Record<string, string> = {
		draft: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
		posted: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
		voided: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
	};

	const TYPE_CLASSES: Record<string, string> = {
		invoice: 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300',
		bill: 'bg-orange-100 text-orange-800 dark:bg-orange-900 dark:text-orange-300',
		payment_in: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300',
		payment_out: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300',
		others: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
		payment: 'bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300',
		journal: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300',
		adjustment: 'bg-cyan-100 text-cyan-800 dark:bg-cyan-900 dark:text-cyan-300'
	};

	const TYPE_LABELS: Record<string, string> = {
		invoice: 'Invoice',
		bill: 'Bill',
		payment_in: 'Payment In',
		payment_out: 'Payment Out',
		others: 'Others',
		payment: 'Payment',
		journal: 'Journal',
		adjustment: 'Adjustment'
	};

	// ─── Formatting helpers ──────────────────────────────────────────────────────
	const formatCurrency = (val: any) => {
		if (val === null || val === undefined) return '-';
		return parseFloat(val).toLocaleString(undefined, {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2
		});
	};

	const isUnbalanced = (txn: any): boolean => {
		if (!txn.lines || txn.lines.length < 2) return true;
		const totalDebit = txn.lines.reduce((s: number, l: any) => s + parseFloat(String(l.debit ?? 0)), 0);
		const totalCredit = txn.lines.reduce((s: number, l: any) => s + parseFloat(String(l.credit ?? 0)), 0);
		return Math.abs(totalDebit - totalCredit) > 0.005;
	};

	const formatDate = (val: any) => {
		if (!val) return '-';
		return dayjs(val).format('YYYY-MM-DD');
	};

	// ─── Currency conversion ─────────────────────────────────────────────────────
	$: nativeCurrency = $companyCurrencyCtx || 'EUR';

	// Try to get native currency from company data when loaded
	const _trySetNativeCurrency = (currency: string) => {
		if (currency) nativeCurrency = currency;
	};

	function cvt(amount: any, date?: string): { display: string; original: string; hasRate: boolean } {
		const num = typeof amount === 'string' ? parseFloat(amount) : (amount ?? 0);
		if (!num || !$displayCurrency || $displayCurrency === nativeCurrency) {
			return { display: '', original: '', hasRate: true };
		}
		const result = convertAmount(num, nativeCurrency, $displayCurrency, ($exchangeRates ?? []), date);
		return {
			display: result.hasRate ? result.converted.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : '',
			original: num.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}),
			hasRate: result.hasRate,
		};
	}

	$: isConverting = $displayCurrency && $displayCurrency !== nativeCurrency;

	// ─── Data loading ────────────────────────────────────────────────────────────
	const loadTransactions = async () => {
		loading = true;
		try {
			const params: Record<string, any> = {
				limit: perPage,
				offset: (currentPage - 1) * perPage,
				company_id: companyId
			};
			if (filterType) params.type = filterType;
			if (filterStatus) params.status = filterStatus;
			if (filterDateFrom) params.date_from = filterDateFrom;
			if (filterDateTo) params.date_to = filterDateTo;
			if (searchQuery) params.search = searchQuery;

			// Check for invoice_id filter from URL
			if (browser) {
				const urlInvoiceId = $pageStore.url.searchParams.get('invoice_id');
				if (urlInvoiceId) params.invoice_id = parseInt(urlInvoiceId);
			}

			const res = await getTransactions(params);
			if (Array.isArray(res)) {
				transactions = res;
				total = res.length;
			} else if (res && res.transactions) {
				transactions = res.transactions;
				total = res.total ?? res.transactions.length;
			} else {
				transactions = [];
				total = 0;
			}
		} catch (err: any) {
			toast.error($i18n.t('Failed to load transactions') + ': ' + (err?.detail ?? err));
		}
		loading = false;
	};

	const loadAccounts = async () => {
		try {
			const res = await getAccounts({ company_id: companyId });
			accounts = Array.isArray(res) ? res : res?.accounts ?? [];
		} catch {
			accounts = [];
		}
	};

	// ─── Search debounce ─────────────────────────────────────────────────────────
	const handleSearch = () => {
		clearTimeout(searchDebounce);
		searchDebounce = setTimeout(() => {
			currentPage = 1;
			loadTransactions();
		}, 300);
	};

	// ─── Filter changes ──────────────────────────────────────────────────────────
	const handleFilterChange = () => {
		currentPage = 1;
		loadTransactions();
	};

	// ─── Expand/collapse ─────────────────────────────────────────────────────────
	const toggleExpand = (id: number) => {
		if (expandedRows.has(id)) {
			expandedRows.delete(id);
		} else {
			expandedRows.add(id);
		}
		expandedRows = expandedRows;
	};

	// ─── Actions ─────────────────────────────────────────────────────────────────
	const openNewEntry = () => {
		editTransaction = null;
		showFormModal = true;
	};

	const openEdit = (txn: any) => {
		editTransaction = txn;
		showFormModal = true;
	};

	const confirmPost = (txn: any) => {
		postTarget = txn;
		showPostConfirm = true;
	};

	const handlePost = async () => {
		if (!postTarget) return;
		try {
			await postTransaction(postTarget.id);
			toast.success($i18n.t('Transaction posted'));
			loadTransactions();
		} catch (err: any) {
			toast.error($i18n.t('Failed to post') + ': ' + (err?.detail ?? err?.message ?? String(err)));
		}
		postTarget = null;
	};

	const confirmVoid = (txn: any) => {
		voidTarget = txn;
		showVoidConfirm = true;
	};

	const handleVoid = async () => {
		if (!voidTarget) return;
		try {
			const result = await voidTransaction(voidTarget.id);
			const draftId = result?.correction_draft?.id;
			if (draftId) {
				toast.success($i18n.t('Entry voided. Correction draft') + ` #${draftId} ` + $i18n.t('created — edit and re-post when ready.'));
			} else {
				toast.success($i18n.t('Transaction voided'));
			}
			loadTransactions();
		} catch (err: any) {
			toast.error($i18n.t('Failed to void') + ': ' + (err?.detail ?? err));
		}
		voidTarget = null;
	};

	const confirmDelete = (txn: any) => {
		deleteTarget = txn;
		showDeleteConfirm = true;
	};

	const handleDelete = async () => {
		if (!deleteTarget) return;
		try {
			await deleteTransaction(deleteTarget.id);
			toast.success($i18n.t('Transaction deleted'));
			if (transactions.length === 1 && currentPage > 1) {
				currentPage -= 1;
			}
			loadTransactions();
		} catch (err: any) {
			toast.error($i18n.t('Failed to delete') + ': ' + (err?.detail ?? err));
		}
		deleteTarget = null;
	};

	const handleFormSave = () => {
		loadTransactions();
	};

	// ─── Pagination reactivity ───────────────────────────────────────────────────
	let mounted = false;

	$: if (mounted && currentPage) {
		loadTransactions();
	}

	// ─── Lifecycle ───────────────────────────────────────────────────────────────
	async function handleAiValidate(txnId: number) {
		validatingId = txnId;
		try {
			const result = await aiValidateTransaction(txnId);
			if (result.is_valid) {
				toast.success($i18n.t('Entry is valid'));
			} else {
				const msgs = [...result.issues, ...result.warnings].join('; ');
				toast.warning(msgs || $i18n.t('Issues found'));
			}
		} catch {
			toast.error($i18n.t('AI validation failed'));
		} finally {
			validatingId = null;
		}
	}

	onMount(() => {
		loadAccounts();
		loadTransactions().then(() => {
			mounted = true;
		});
		getAccountingAiStatus()
			.then((s) => (aiAvailable = s.available))
			.catch(() => {});
	});
</script>

<!-- Delete confirmation -->
<ConfirmDialog
	bind:show={showDeleteConfirm}
	on:confirm={handleDelete}
	title={$i18n.t('Delete Transaction')}
	message={$i18n.t('Are you sure you want to delete this draft transaction? This action cannot be undone.')}
/>

<!-- Post confirmation -->
<ConfirmDialog
	bind:show={showPostConfirm}
	on:confirm={handlePost}
	title={$i18n.t('Post Transaction')}
	message={$i18n.t('Posting this transaction will make it permanent. It can only be reversed by voiding. Continue?')}
/>

<!-- Void confirmation -->
<ConfirmDialog
	bind:show={showVoidConfirm}
	on:confirm={handleVoid}
	title={$i18n.t('Void Transaction')}
	message={$i18n.t('Voiding this transaction will create a reversal entry. Continue?')}
/>

<!-- Form modal -->
<TransactionFormModal
	bind:show={showFormModal}
	transaction={editTransaction}
	{accounts}
	{companyId}
	on:save={handleFormSave}
/>

<!-- Invoice preview modal -->
<DocumentPreviewModal
	bind:show={showPreview}
	invoice={previewInvoice}
	onUpdate={(updated) => { previewInvoice = updated; }}
/>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Transactions')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{total}
			</div>
		</div>

		<div class="flex gap-1 items-center">
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
					placeholder={$i18n.t('Search transactions...')}
				/>
			</div>

			<!-- New Entry button -->
			<button
				class="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-xl bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium transition whitespace-nowrap"
				on:click={openNewEntry}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-3.5"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
				</svg>
				{$i18n.t('New Entry')}
			</button>
		</div>
	</div>

	<!-- Filter Bar -->
	<div
		class="flex flex-wrap items-center gap-2 py-2 px-0.5 text-xs"
	>
		<!-- Type filter -->
		<select
			class="text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-900 px-2.5 py-1.5 outline-hidden dark:text-gray-300"
			bind:value={filterType}
			on:change={handleFilterChange}
		>
			<option value="">{$i18n.t('All Types')}</option>
			<option value="invoice">{$i18n.t('Invoice')}</option>
			<option value="bill">{$i18n.t('Bill')}</option>
			<option value="payment_in">{$i18n.t('Payment In')}</option>
			<option value="payment_out">{$i18n.t('Payment Out')}</option>
			<option value="others">{$i18n.t('Others')}</option>
			<option value="adjustment">{$i18n.t('Adjustment')}</option>
		</select>

		<!-- Status filter -->
		<select
			class="text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-900 px-2.5 py-1.5 outline-hidden dark:text-gray-300"
			bind:value={filterStatus}
			on:change={handleFilterChange}
		>
			<option value="">{$i18n.t('All Statuses')}</option>
			<option value="draft">{$i18n.t('Draft')}</option>
			<option value="posted">{$i18n.t('Posted')}</option>
			<option value="voided">{$i18n.t('Voided')}</option>
		</select>

		<!-- Date From -->
		<div class="flex items-center gap-1">
			<span class="text-gray-500 dark:text-gray-400">{$i18n.t('From')}:</span>
			<input
				type="date"
				class="text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-900 px-2 py-1.5 outline-hidden dark:text-gray-300"
				bind:value={filterDateFrom}
				on:change={handleFilterChange}
			/>
		</div>

		<!-- Date To -->
		<div class="flex items-center gap-1">
			<span class="text-gray-500 dark:text-gray-400">{$i18n.t('To')}:</span>
			<input
				type="date"
				class="text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-900 px-2 py-1.5 outline-hidden dark:text-gray-300"
				bind:value={filterDateTo}
				on:change={handleFilterChange}
			/>
		</div>

		<!-- Unbalanced filter -->
		<button
			class="text-xs px-2.5 py-1.5 rounded-lg border transition {filterUnbalanced
				? 'border-amber-400 bg-amber-50 text-amber-700 dark:border-amber-500 dark:bg-amber-900/30 dark:text-amber-300 font-medium'
				: 'border-gray-200 dark:border-gray-700 text-gray-500 dark:text-gray-400 hover:border-amber-300 dark:hover:border-amber-600'}"
			on:click={() => { filterUnbalanced = !filterUnbalanced; }}
		>
			{$i18n.t('Unbalanced')}
		</button>

		<!-- Clear filters -->
		{#if filterType || filterStatus || filterDateFrom || filterDateTo || searchQuery || filterUnbalanced}
			<button
				class="text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 underline"
				on:click={() => {
					filterType = '';
					filterStatus = '';
					filterDateFrom = '';
					filterDateTo = '';
					filterUnbalanced = false;
					searchQuery = '';
					handleFilterChange();
				}}
			>
				{$i18n.t('Clear filters')}
			</button>
		{/if}
	</div>

	<!-- Table -->
	{#if loading && transactions.length === 0}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if filteredTransactions.length === 0}
		<div class="flex justify-center my-10 text-sm text-gray-500">
			{$i18n.t(filterUnbalanced ? 'No unbalanced transactions found' : 'No transactions found')}
		</div>
	{:else}
		<!-- Bulk Action Bar -->
		{#if selectedIds.size > 0}
			<div class="flex items-center gap-3 mb-2 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm">
				<span class="font-medium text-blue-700 dark:text-blue-300">{selectedIds.size} {$i18n.t('selected')}</span>
				<button class="px-2 py-0.5 text-xs rounded bg-green-600 text-white hover:bg-green-700 transition" on:click={handleBulkPost}>{$i18n.t('Post Selected')}</button>
				<button class="px-2 py-0.5 text-xs rounded bg-red-600 text-white hover:bg-red-700 transition" on:click={handleBulkDelete}>{$i18n.t('Delete Selected')}</button>
				<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300" on:click={() => { selectedIds = new Set(); }}>{$i18n.t('Clear')}</button>
			</div>
		{/if}

		<div class="overflow-x-auto">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead
					class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
				>
					<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
						<th class="px-2 py-2 w-8"><input type="checkbox" checked={allDraftsSelected} on:change={toggleSelectAll} class="rounded" /></th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('#')}</th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('Date')}</th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('Type')}</th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('Status')}</th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('Reference')}</th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('Description')}</th>
						<th class="px-3 py-2 whitespace-nowrap text-right">{$i18n.t('Total')}</th>
						<th class="px-3 py-2 whitespace-nowrap">{$i18n.t('Invoice')}</th>
						<th class="px-3 py-2 whitespace-nowrap text-right">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each filteredTransactions as txn (txn.id)}
						<!-- Main row -->
						<tr
							class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition cursor-pointer"
							on:click={() => toggleExpand(txn.id)}
						>
							<!-- Checkbox -->
							<td class="px-2 py-2" on:click|stopPropagation>
								{#if txn.status === 'draft'}
									<input type="checkbox" checked={selectedIds.has(txn.id)} on:change={() => toggleSelect(txn.id)} class="rounded" />
								{/if}
							</td>
							<!-- Entry # -->
							<td class="px-3 py-2 whitespace-nowrap text-gray-400 font-mono text-[10px]">
								{txn.id}
							</td>

							<!-- Date -->
							<td class="px-3 py-2 whitespace-nowrap">
								{formatDate(txn.transaction_date)}
							</td>

							<!-- Type badge -->
							<td class="px-3 py-2">
								<span
									class="inline-block px-2 py-0.5 rounded-lg text-xs font-medium {TYPE_CLASSES[
										txn.transaction_type
									] ?? TYPE_CLASSES.others}"
								>
									{TYPE_LABELS[txn.transaction_type] ?? txn.transaction_type}
								</span>
							</td>

							<!-- Status badge -->
							<td class="px-3 py-2">
								<div class="flex items-center gap-1">
									<span
										class="inline-block px-2 py-0.5 rounded-lg text-xs font-medium uppercase {STATUS_CLASSES[
											txn.status
										] ?? STATUS_CLASSES.draft}"
									>
										{txn.status}
									</span>
									{#if isUnbalanced(txn)}
										<Tooltip content={$i18n.t('Missing double entry — debits and credits do not balance')}>
											<span class="inline-flex text-amber-500">
												<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
													stroke-width="2" stroke="currentColor" class="size-3.5">
													<path stroke-linecap="round" stroke-linejoin="round"
														d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
												</svg>
											</span>
										</Tooltip>
									{/if}
								</div>
							</td>

							<!-- Reference -->
							<td class="px-3 py-2 whitespace-nowrap max-w-[140px] overflow-hidden text-ellipsis">
								{txn.reference ?? '-'}
							</td>

							<!-- Description -->
							<td class="px-3 py-2 max-w-[220px] overflow-hidden text-ellipsis whitespace-nowrap">
								<Tooltip content={txn.description ?? ''}>
									<span>{txn.description ?? '-'}</span>
								</Tooltip>
							</td>

							<!-- Total -->
							<td class="px-3 py-2 text-right whitespace-nowrap font-medium">
								{#key $displayCurrency}
								{#if isConverting}
									{@const c = cvt(txn.total, txn.transaction_date)}
									{#if c.hasRate}
										<span class="font-medium">{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
										<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
									{:else}
										<span>{c.original} {nativeCurrency}</span>
										<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
									{/if}
								{:else}
									{formatCurrency(txn.total)}
								{/if}
								{/key}
							</td>

							<!-- Invoice link -->
							<td class="px-3 py-2 whitespace-nowrap">
								{#if txn.invoice_id}
									<button
										class="text-blue-600 dark:text-blue-400 hover:underline hover:text-blue-800 dark:hover:text-blue-200 transition flex items-center gap-0.5"
										on:click|stopPropagation={() => openInvoicePreview(txn.invoice_id)}
										title={$i18n.t('Preview invoice')}
									>
										<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3 h-3"><path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" /><path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" /></svg>
										#{txn.invoice_id}
									</button>
								{:else}
									<span class="text-gray-400">-</span>
								{/if}
							</td>

							<!-- Actions -->
							<td class="px-3 py-2 text-right whitespace-nowrap">
								<div
									class="flex items-center justify-end gap-1"
									on:click|stopPropagation
								>
									<!-- Post (draft only) -->
									{#if txn.status === 'draft'}
										<Tooltip content={$i18n.t('Post')}>
											<button
												class="p-1 rounded-lg hover:bg-green-50 dark:hover:bg-green-900/20 text-green-600 dark:text-green-400 transition"
												on:click={() => confirmPost(txn)}
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
														d="m4.5 12.75 6 6 9-13.5"
													/>
												</svg>
											</button>
										</Tooltip>
									{/if}

									<!-- AI Validate (draft only) -->
									{#if aiAvailable && txn.status === 'draft'}
										<Tooltip content={$i18n.t('AI Validate')}>
											<button
												class="p-1 rounded-lg hover:bg-purple-50 dark:hover:bg-purple-900/20 text-purple-600 dark:text-purple-400 transition disabled:opacity-50"
												disabled={validatingId === txn.id}
												on:click={() => handleAiValidate(txn.id)}
											>
												<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5">
													<path stroke-linecap="round" stroke-linejoin="round" d="M9.813 15.904 9 18.75l-.813-2.846a4.5 4.5 0 0 0-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 0 0 3.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 0 0 3.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 0 0-3.09 3.09ZM18.259 8.715 18 9.75l-.259-1.035a3.375 3.375 0 0 0-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 0 0 2.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 0 0 2.455 2.456L21.75 6l-1.036.259a3.375 3.375 0 0 0-2.455 2.456Z" />
												</svg>
											</button>
										</Tooltip>
									{/if}

									<!-- Void (posted only) -->
									{#if txn.status === 'posted'}
										<Tooltip content={$i18n.t('Void')}>
											<button
												class="p-1 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 text-red-500 transition"
												on:click={() => confirmVoid(txn)}
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
														d="M18.364 18.364A9 9 0 0 0 5.636 5.636m12.728 12.728A9 9 0 0 1 5.636 5.636m12.728 12.728L5.636 5.636"
													/>
												</svg>
											</button>
										</Tooltip>
									{/if}

									<!-- Edit (draft only) -->
									{#if txn.status === 'draft'}
										<Tooltip content={$i18n.t('Edit')}>
											<button
												class="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition"
												on:click={() => openEdit(txn)}
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
														d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L10.582 16.07a4.5 4.5 0 0 1-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 0 1 1.13-1.897l8.932-8.931Zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0 1 15.75 21H5.25A2.25 2.25 0 0 1 3 18.75V8.25A2.25 2.25 0 0 1 5.25 6H10"
													/>
												</svg>
											</button>
										</Tooltip>
									{/if}

									<!-- View (posted/voided) -->
									{#if txn.status === 'posted' || txn.status === 'voided'}
										<Tooltip content={$i18n.t('View')}>
											<button
												class="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition"
												on:click={() => openEdit(txn)}
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
									{/if}

									<!-- Delete (draft only) -->
									{#if txn.status === 'draft'}
										<Tooltip content={$i18n.t('Delete')}>
											<button
												class="p-1 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 text-red-500 transition"
												on:click={() => confirmDelete(txn)}
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
									{/if}

									<!-- Expand chevron -->
									<button
										class="p-1 hover:bg-gray-100 dark:hover:bg-gray-850 rounded-lg transition"
										on:click={() => toggleExpand(txn.id)}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											viewBox="0 0 24 24"
											stroke-width="2"
											stroke="currentColor"
											class="size-3.5 transition-transform {expandedRows.has(txn.id)
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
								</div>
							</td>
						</tr>

						<!-- Expanded Detail Row: debit/credit lines -->
						{#if expandedRows.has(txn.id)}
							<tr class="bg-gray-50/50 dark:bg-gray-850/30">
								<td colspan="9" class="px-4 py-3">
									<div class="space-y-3">
										{#if txn.lines?.length}
											<div>
												<div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5">
													{$i18n.t('Journal Lines')}
												</div>
												<table
													class="w-full text-xs text-left text-gray-500 dark:text-gray-400"
												>
													<thead>
														<tr
															class="border-b border-gray-200 dark:border-gray-700"
														>
															<th class="py-1.5 pr-2">{$i18n.t('Account Code')}</th>
															<th class="py-1.5 pr-2">{$i18n.t('Account Name')}</th>
															<th class="py-1.5 pr-2 text-right">{$i18n.t('Debit')}</th>
															<th class="py-1.5 pr-2 text-right">{$i18n.t('Credit')}</th>
															<th class="py-1.5">{$i18n.t('Description')}</th>
														</tr>
													</thead>
													<tbody>
														{#each txn.lines as line}
															<tr class="border-b border-gray-100 dark:border-gray-800">
																<td class="py-1 pr-2 dark:text-gray-300 font-mono">
																	{line.account_code ?? '-'}
																</td>
																<td class="py-1 pr-2 dark:text-gray-300">
																	{line.account_name ?? '-'}
																</td>
																<td class="py-1 pr-2 text-right dark:text-gray-300">
																	{#if parseFloat(String(line.debit ?? 0)) > 0}
																		{#key $displayCurrency}
																		{#if isConverting}
																			{@const c = cvt(line.debit, txn.transaction_date)}
																			{#if c.hasRate}
																				<span>{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
																				<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
																			{:else}
																				<span>{c.original} {nativeCurrency}</span>
																				<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
																			{/if}
																		{:else}
																			{formatCurrency(line.debit)}
																		{/if}
																		{/key}
																	{:else}
																		-
																	{/if}
																</td>
																<td class="py-1 pr-2 text-right dark:text-gray-300">
																	{#if parseFloat(String(line.credit ?? 0)) > 0}
																		{#key $displayCurrency}
																		{#if isConverting}
																			{@const c = cvt(line.credit, txn.transaction_date)}
																			{#if c.hasRate}
																				<span>{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
																				<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
																			{:else}
																				<span>{c.original} {nativeCurrency}</span>
																				<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
																			{/if}
																		{:else}
																			{formatCurrency(line.credit)}
																		{/if}
																		{/key}
																	{:else}
																		-
																	{/if}
																</td>
																<td class="py-1 dark:text-gray-300">
																	{line.description ?? '-'}
																</td>
															</tr>
														{/each}
														<!-- Totals row -->
														{#each [{ d: txn.lines.reduce((s, l) => s + parseFloat(String(l.debit ?? 0)), 0), c: txn.lines.reduce((s, l) => s + parseFloat(String(l.credit ?? 0)), 0) }] as totals}
															{@const unbalanced = Math.abs(totals.d - totals.c) > 0.005}
															<tr class="border-t border-gray-300 dark:border-gray-600 font-semibold {unbalanced ? 'text-red-600 dark:text-red-400' : 'text-gray-700 dark:text-gray-300'}">
																<td class="py-1.5 pr-2" colspan="2">{$i18n.t('Total')}</td>
																<td class="py-1.5 pr-2 text-right">{formatCurrency(totals.d)}</td>
																<td class="py-1.5 pr-2 text-right">{formatCurrency(totals.c)}</td>
																<td class="py-1.5">
																	{#if unbalanced}
																		<span class="text-red-500 text-[10px]">{$i18n.t('Unbalanced')}: {formatCurrency(Math.abs(totals.d - totals.c))}</span>
																	{/if}
																</td>
															</tr>
														{/each}
													</tbody>
												</table>
											</div>
										{:else}
											<div class="text-xs text-gray-400">
												{$i18n.t('No line details available')}
											</div>
										{/if}

										<!-- Extra metadata -->
										<div class="flex flex-wrap gap-4 text-xs text-gray-400">
											{#if txn.currency && txn.currency !== 'USD'}
												<span>{$i18n.t('Currency')}: {txn.currency}</span>
											{/if}
											{#if txn.exchange_rate && txn.exchange_rate !== 1}
												<span>{$i18n.t('Exchange Rate')}: {txn.exchange_rate}</span>
											{/if}
											{#if txn.notes}
												<span>{$i18n.t('Notes')}: {txn.notes}</span>
											{/if}
											{#if txn.payment_id}
												<span>{$i18n.t('Payment')} #{txn.payment_id}</span>
											{/if}
											<span>{$i18n.t('Created')}: {formatDate(txn.created_at)}</span>
											{#if txn.updated_at && txn.updated_at !== txn.created_at}
												<span>{$i18n.t('Updated')}: {formatDate(txn.updated_at)}</span>
											{/if}
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
			<div class="text-xs text-gray-500">
				{$i18n.t('Showing')}
				{Math.min((currentPage - 1) * perPage + 1, total)}–{Math.min(
					currentPage * perPage,
					total
				)}
				{$i18n.t('of')}
				{total}
			</div>

			<Pagination bind:page={currentPage} count={total} {perPage} />
		</div>
	{/if}
</div>
