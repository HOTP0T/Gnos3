<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { toast } from 'svelte-sonner';

	import {
		getCompanyInvoices,
		getUnassignedInvoices,
		assignInvoiceToCompany,
		unassignInvoice,
		bulkAssignInvoices,
		bulkUnassignInvoices,
		confirmInvoiceCategory,
		getAccounts,
		getBankAccounts,
		getBankStatements,
		matchBankStatement,
		getTransactions
	} from '$lib/apis/accounting';
	import { INVOICE_API_BASE_URL, K4MI_BASE_URL } from '$lib/constants';
	import { convertAmount } from '$lib/utils/currency';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import PaymentFormModal from '$lib/components/accounting/PaymentFormModal.svelte';
	import DocumentPreviewModal from '$lib/components/invoices/DocumentPreviewModal.svelte';

	const i18n = getContext('i18n');
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');

	export let companyId: number;

	let loading = true;
	let companyInvoices: any[] = [];
	let companyTotal = 0;
	let unassignedInvoices: any[] = [];
	let unassignedTotal = 0;

	// Filters
	let searchCompany = '';
	let searchUnassigned = '';
	let showUnassigned = true;
	let filterReview: 'all' | 'review' | 'ready' = 'all';

	// Advanced filters
	let dateFrom = '';
	let dateTo = '';
	let minAmount: string = '';
	let maxAmount: string = '';
	let showCompanyFilters = false;
	let showUnassignedFilters = false;

	// Pagination
	let companyPage = 0;
	let unassignedPage = 0;
	let perPage = 50;

	// Sort (server-side)
	let companySortBy = 'invoice_date';
	let companySortDir: 'asc' | 'desc' = 'desc';
	let unassignedSortBy = 'invoice_date';
	let unassignedSortDir: 'asc' | 'desc' = 'desc';

	// Selection
	let selectedCompanyIds: Set<number> = new Set();
	let selectedUnassignedIds: Set<number> = new Set();

	// Expanded rows
	let expandedId: number | null = null;

	// Document preview modal
	let showPreview = false;
	let previewInvoice: any = null;

	const openPreview = (inv: any) => {
		previewInvoice = inv;
		showPreview = true;
	};

	const handlePreviewUpdate = (updated: any) => {
		// Refresh the invoice in both lists
		const ci = companyInvoices.findIndex((i) => i.id === updated.id);
		if (ci >= 0) { companyInvoices[ci] = updated; companyInvoices = companyInvoices; }
		const ui = unassignedInvoices.findIndex((i) => i.id === updated.id);
		if (ui >= 0) { unassignedInvoices[ui] = updated; unassignedInvoices = unassignedInvoices; }
	};

	// Stats
	let totalAmount = 0;
	let reviewCount = 0;

	// AI activity indicator
	let aiActivity = '';
	let aiProcessingIds: Set<number> = new Set();

	// Account lookup for categorization display
	let accountMap: Record<string, string> = {};
	let accountsList: any[] = [];
	const loadAccounts = async () => {
		try {
			const res = await getAccounts({ company_id: companyId });
			const accts = res?.accounts ?? res ?? [];
			if (Array.isArray(accts)) {
				accountsList = accts;
				for (const a of accts) accountMap[a.code] = a.name;
			}
		} catch {}
	};
	const accountLabel = (code: string | null) => {
		if (!code) return '';
		return accountMap[code] ? `${code} ${accountMap[code]}` : code;
	};

	// Inline edit: patch invoice fields
	const patchInvoice = async (invoiceId: number, fields: Record<string, any>) => {
		const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/${invoiceId}`, {
			method: 'PATCH',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(fields)
		});
		if (!res.ok) throw await res.json();
		return res.json();
	};

	const saveField = async (invoiceId: number, field: string, value: string) => {
		try {
			let parsed: any = value || null;
			if (['total_amount', 'subtotal', 'tax_amount'].includes(field) && value) {
				parsed = parseFloat(value);
				if (isNaN(parsed)) parsed = null;
			}
			await patchInvoice(invoiceId, { [field]: parsed });
			toast.success($i18n.t('Saved'));
			const idx = companyInvoices.findIndex((i) => i.id === invoiceId);
			if (idx >= 0) companyInvoices[idx][field] = value;
			companyInvoices = companyInvoices;
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	// Payment modal state
	let showPaymentModal = false;
	let paymentPrefill: any = null;

	// Bank reconciliation state for pay-and-match flow
	let bankMatchCandidates: any[] = [];
	let bankMatchLoading = false;
	let selectedBankLineId: number | null = null;
	let paymentInvoiceId: number | null = null;

	const suggestPayment = async (inv: any) => {
		const isSale = (inv.vendor_name || '').toLowerCase().includes('ma soci');
		paymentPrefill = {
			payment_date: new Date().toISOString().slice(0, 10),
			amount: parseFloat(inv.total_amount) || 0,
			currency: inv.currency || 'EUR',
			direction: isSale ? 'inbound' : 'outbound',
			method: 'bank_transfer',
			payer: isSale ? (inv.client_name || '') : '',
			payee: isSale ? '' : (inv.vendor_name || ''),
			reference: `PAY-${inv.invoice_number || inv.id}`,
			invoice_id: inv.id
		};
		paymentInvoiceId = inv.id;
		selectedBankLineId = null;
		bankMatchCandidates = [];
		showPaymentModal = true;

		// Fetch unmatched bank lines that could match this payment amount
		bankMatchLoading = true;
		try {
			const bankAccts = await getBankAccounts(companyId) ?? [];
			if (bankAccts.length > 0) {
				const allLines: any[] = [];
				for (const ba of bankAccts) {
					const lines = await getBankStatements(ba.id, { status: 'unmatched' }) ?? [];
					for (const line of lines) {
						line._bankAccountName = ba.name;
						line._bankAccountCurrency = ba.currency || '';
					}
					allLines.push(...lines);
				}
				// Filter to lines with similar amounts (within 1% tolerance)
				const targetAmount = Math.abs(parseFloat(inv.total_amount) || 0);
				if (targetAmount > 0) {
					const tolerance = targetAmount * 0.01;
					bankMatchCandidates = allLines.filter((line: any) => {
						const lineAmount = Math.abs(parseFloat(line.amount) || 0);
						return Math.abs(lineAmount - targetAmount) <= tolerance;
					});
				} else {
					bankMatchCandidates = [];
				}
			}
		} catch (err) {
			// Non-blocking: bank match is optional
			console.warn('Could not fetch bank statements for matching:', err);
		}
		bankMatchLoading = false;
	};

	const selectBankLine = (line: any) => {
		selectedBankLineId = line.id;
		// Pre-fill payment date from the bank line date
		if (paymentPrefill && line.transaction_date) {
			paymentPrefill = { ...paymentPrefill, payment_date: line.transaction_date };
		}
	};

	const deselectBankLine = () => {
		selectedBankLineId = null;
		// Reset payment date to today
		if (paymentPrefill) {
			paymentPrefill = { ...paymentPrefill, payment_date: new Date().toISOString().slice(0, 10) };
		}
	};

	const handlePaymentSaved = async () => {
		// If a bank line was selected, auto-match it to the payment's transaction
		if (selectedBankLineId && paymentInvoiceId) {
			try {
				// Find the transaction created for this payment (by invoice_id, most recent)
				const txnRes = await getTransactions({ company_id: companyId, invoice_id: paymentInvoiceId, limit: 1 });
				const txns = txnRes?.transactions ?? txnRes ?? [];
				if (Array.isArray(txns) && txns.length > 0) {
					// Pick the last (most recent) payment-type transaction
					const paymentTxn = txns.find((t: any) => t.transaction_type === 'payment') ?? txns[0];
					await matchBankStatement(selectedBankLineId, paymentTxn.id);
					toast.success($i18n.t('Payment recorded and bank line matched'));
				}
			} catch (err) {
				console.warn('Bank matching failed:', err);
				toast.error($i18n.t('Payment recorded but bank matching failed'));
			}
		}
		selectedBankLineId = null;
		bankMatchCandidates = [];
		paymentInvoiceId = null;
		await reloadAll();
	};

	// Account picker state
	let accountPickerInvoiceId: number | null = null;

	const handleConfirmCategory = async (inv: any) => {
		const code = inv.suggested_account_code;
		if (!code) return;
		try {
			await confirmInvoiceCategory(inv.id, code);
			toast.success($i18n.t('Account confirmed'));
			await reloadAll();
		} catch (err) { toast.error(`${err}`); }
	};

	const handleManualAccountSelect = async (invoiceId: number, accountCode: string) => {
		if (!accountCode) return;
		aiProcessingIds.add(invoiceId);
		aiProcessingIds = aiProcessingIds;
		aiActivity = $i18n.t('AI is creating journal entry...');
		try {
			const result = await confirmInvoiceCategory(invoiceId, accountCode);
			const txnId = result?.transaction_id;
			if (txnId) {
				toast.success($i18n.t('Account confirmed — Draft entry') + ` #${txnId} ` + $i18n.t('created'));
			} else {
				toast.success($i18n.t('Account confirmed'));
			}
			accountPickerInvoiceId = null;
			await reloadAll();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		aiProcessingIds.delete(invoiceId);
		aiProcessingIds = aiProcessingIds;
		aiActivity = '';
	};

	// Client-side review filter (quick local filter, no server round-trip)
	$: filteredCompany = companyInvoices.filter((inv) => {
		if (filterReview === 'review' && !inv.needs_review) return false;
		if (filterReview === 'ready' && inv.needs_review) return false;
		return true;
	});

	$: allCompanySelected = filteredCompany.length > 0 && filteredCompany.every((i) => selectedCompanyIds.has(i.id));
	$: allUnassignedSelected = unassignedInvoices.length > 0 && unassignedInvoices.every((i) => selectedUnassignedIds.has(i.id));

	const loadCompanyInvoices = async () => {
		try {
			const res = await getCompanyInvoices(companyId, {
				q: searchCompany || undefined,
				date_from: dateFrom || undefined,
				date_to: dateTo || undefined,
				min_amount: minAmount ? parseFloat(minAmount) : undefined,
				max_amount: maxAmount ? parseFloat(maxAmount) : undefined,
				sort_by: companySortBy,
				sort_dir: companySortDir,
				limit: perPage,
				offset: companyPage * perPage
			});
			companyInvoices = res.invoices ?? [];
			companyTotal = res.total ?? 0;
			totalAmount = companyInvoices.reduce((s, i) => s + (parseFloat(i.total_amount) || 0), 0);
			reviewCount = companyInvoices.filter((i) => i.needs_review).length;
			selectedCompanyIds = new Set();
		} catch (err) { toast.error(`${err}`); }
	};

	const loadUnassignedInvoices = async () => {
		try {
			const res = await getUnassignedInvoices({
				q: searchUnassigned || undefined,
				date_from: dateFrom || undefined,
				date_to: dateTo || undefined,
				min_amount: minAmount ? parseFloat(minAmount) : undefined,
				max_amount: maxAmount ? parseFloat(maxAmount) : undefined,
				sort_by: unassignedSortBy,
				sort_dir: unassignedSortDir,
				limit: perPage,
				offset: unassignedPage * perPage
			});
			unassignedInvoices = res.invoices ?? [];
			unassignedTotal = res.total ?? 0;
			selectedUnassignedIds = new Set();
		} catch (err) { toast.error(`${err}`); }
	};

	const reloadAll = async () => {
		await Promise.all([loadCompanyInvoices(), showUnassigned ? loadUnassignedInvoices() : Promise.resolve()]);
	};

	const handleBulkAssign = async () => {
		if (selectedUnassignedIds.size === 0) return;
		const count = selectedUnassignedIds.size;
		aiActivity = $i18n.t('AI is categorizing') + ` ${count} ` + $i18n.t('invoice(s) and creating journal entries...');
		try {
			await bulkAssignInvoices(companyId, [...selectedUnassignedIds]);
			toast.success($i18n.t(`${count} invoice(s) assigned — AI categorized and created draft entries`));
			await reloadAll();
		} catch (err) { toast.error(`${err}`); }
		aiActivity = '';
	};

	const handleBulkUnassign = async () => {
		if (selectedCompanyIds.size === 0) return;
		try {
			await bulkUnassignInvoices(companyId, [...selectedCompanyIds]);
			toast.success($i18n.t(`${selectedCompanyIds.size} invoice(s) removed`));
			await reloadAll();
		} catch (err) { toast.error(`${err}`); }
	};

	const toggleSort = (field: string, panel: 'company' | 'unassigned') => {
		if (panel === 'company') {
			if (companySortBy === field) companySortDir = companySortDir === 'asc' ? 'desc' : 'asc';
			else { companySortBy = field; companySortDir = 'asc'; }
			companyPage = 0;
			loadCompanyInvoices();
		} else {
			if (unassignedSortBy === field) unassignedSortDir = unassignedSortDir === 'asc' ? 'desc' : 'asc';
			else { unassignedSortBy = field; unassignedSortDir = 'asc'; }
			unassignedPage = 0;
			loadUnassignedInvoices();
		}
	};

	const toggleSelectAllCompany = () => {
		if (allCompanySelected) selectedCompanyIds = new Set();
		else selectedCompanyIds = new Set(filteredCompany.map((i) => i.id));
		selectedCompanyIds = selectedCompanyIds;
	};

	const toggleSelectAllUnassigned = () => {
		if (allUnassignedSelected) selectedUnassignedIds = new Set();
		else selectedUnassignedIds = new Set(unassignedInvoices.map((i) => i.id));
		selectedUnassignedIds = selectedUnassignedIds;
	};

	const toggleCompanySelect = (id: number) => {
		if (selectedCompanyIds.has(id)) selectedCompanyIds.delete(id);
		else selectedCompanyIds.add(id);
		selectedCompanyIds = selectedCompanyIds;
	};

	const toggleUnassignedSelect = (id: number) => {
		if (selectedUnassignedIds.has(id)) selectedUnassignedIds.delete(id);
		else selectedUnassignedIds.add(id);
		selectedUnassignedIds = selectedUnassignedIds;
	};

	let searchTimer: ReturnType<typeof setTimeout>;
	const onSearchCompany = () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { companyPage = 0; loadCompanyInvoices(); }, 300); };
	let searchTimer2: ReturnType<typeof setTimeout>;
	const onSearchUnassigned = () => { clearTimeout(searchTimer2); searchTimer2 = setTimeout(() => { unassignedPage = 0; loadUnassignedInvoices(); }, 300); };

	const onCompanyFilterChange = () => { companyPage = 0; loadCompanyInvoices(); };
	const onUnassignedFilterChange = () => { unassignedPage = 0; loadUnassignedInvoices(); };

	const clearCompanyFilters = () => {
		dateFrom = '';
		dateTo = '';
		minAmount = '';
		maxAmount = '';
		companySortBy = 'invoice_date';
		companySortDir = 'desc';
		companyPage = 0;
		loadCompanyInvoices();
	};

	const clearUnassignedFilters = () => {
		dateFrom = '';
		dateTo = '';
		minAmount = '';
		maxAmount = '';
		unassignedSortBy = 'invoice_date';
		unassignedSortDir = 'desc';
		unassignedPage = 0;
		loadUnassignedInvoices();
	};

	const setCompanySortPreset = (value: string) => {
		switch (value) {
			case 'date_desc': companySortBy = 'invoice_date'; companySortDir = 'desc'; break;
			case 'date_asc': companySortBy = 'invoice_date'; companySortDir = 'asc'; break;
			case 'amount_desc': companySortBy = 'total_amount'; companySortDir = 'desc'; break;
			case 'amount_asc': companySortBy = 'total_amount'; companySortDir = 'asc'; break;
			case 'vendor_asc': companySortBy = 'vendor_name'; companySortDir = 'asc'; break;
			case 'vendor_desc': companySortBy = 'vendor_name'; companySortDir = 'desc'; break;
		}
		companyPage = 0;
		loadCompanyInvoices();
	};

	const setUnassignedSortPreset = (value: string) => {
		switch (value) {
			case 'date_desc': unassignedSortBy = 'invoice_date'; unassignedSortDir = 'desc'; break;
			case 'date_asc': unassignedSortBy = 'invoice_date'; unassignedSortDir = 'asc'; break;
			case 'amount_desc': unassignedSortBy = 'total_amount'; unassignedSortDir = 'desc'; break;
			case 'amount_asc': unassignedSortBy = 'total_amount'; unassignedSortDir = 'asc'; break;
			case 'vendor_asc': unassignedSortBy = 'vendor_name'; unassignedSortDir = 'asc'; break;
			case 'vendor_desc': unassignedSortBy = 'vendor_name'; unassignedSortDir = 'desc'; break;
		}
		unassignedPage = 0;
		loadUnassignedInvoices();
	};

	$: companySortPreset = `${companySortBy === 'total_amount' ? 'amount' : companySortBy === 'vendor_name' ? 'vendor' : 'date'}_${companySortDir}`;
	$: unassignedSortPreset = `${unassignedSortBy === 'total_amount' ? 'amount' : unassignedSortBy === 'vendor_name' ? 'vendor' : 'date'}_${unassignedSortDir}`;

	// ─── Currency conversion ────────────────────────────────────────────────────
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

	const fmt = (amount: string | number | null, cur: string = 'USD'): string => {
		if (!amount) return '-';
		const n = typeof amount === 'string' ? parseFloat(amount) : amount;
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const sortIcon = (field: string, currentField: string, currentDir: string): string => {
		if (field !== currentField) return '';
		return currentDir === 'asc' ? ' \u25B2' : ' \u25BC';
	};

	onMount(async () => { await Promise.all([loadCompanyInvoices(), loadAccounts(), loadUnassignedInvoices()]); loading = false; });
</script>

{#if loading}
	<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
{:else}
	<div class="py-3 space-y-4">
		<!-- AI Activity Banner -->
		{#if aiActivity}
			<div class="flex items-center gap-2 px-4 py-2.5 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/50 rounded-xl text-sm text-blue-700 dark:text-blue-300 animate-pulse">
				<svg class="animate-spin h-4 w-4 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
				<span>{aiActivity}</span>
			</div>
		{/if}

		<!-- Stats -->
		<div class="grid grid-cols-3 gap-3">
			<div class="bg-white dark:bg-gray-900 rounded-xl p-3 border border-gray-100/30 dark:border-gray-850/30">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Company Invoices')}</div>
				<div class="text-xl font-medium dark:text-gray-200">{companyTotal}</div>
			</div>
			<div class="bg-white dark:bg-gray-900 rounded-xl p-3 border border-gray-100/30 dark:border-gray-850/30">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Total Amount')}</div>
				<div class="text-xl font-medium dark:text-gray-200">
					{#key $displayCurrency}
					{#if isConverting}
						{@const c = cvt(totalAmount)}
						{#if c.hasRate}
							<span>{c.display} <span class="text-xs text-gray-400">{$displayCurrency}</span></span>
							<div class="text-[10px] text-gray-400">{c.original} {nativeCurrency}</div>
						{:else}
							<span>{c.original} {nativeCurrency}</span>
						{/if}
					{:else}
						{fmt(totalAmount)} <span class="text-sm text-gray-400">{nativeCurrency}</span>
					{/if}
					{/key}
				</div>
			</div>
			<div class="bg-white dark:bg-gray-900 rounded-xl p-3 border {reviewCount > 0 ? 'border-yellow-200 dark:border-yellow-800/50' : 'border-gray-100/30 dark:border-gray-850/30'}">
				<div class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Needs Review')}</div>
				<div class="text-xl font-medium {reviewCount > 0 ? 'text-yellow-600 dark:text-yellow-400' : 'dark:text-gray-200'}">{reviewCount}</div>
			</div>
		</div>

		<!-- Company Invoices Panel -->
		<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30">
			<div class="flex justify-between items-center mb-3 gap-2 flex-wrap">
				<div class="text-sm font-medium dark:text-gray-200">{$i18n.t('Company Invoices')}</div>
				<div class="flex items-center gap-2">
					<!-- Review filter -->
					<select bind:value={filterReview} class="text-xs rounded-lg px-2 py-1 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300">
						<option value="all">{$i18n.t('All')}</option>
						<option value="review">{$i18n.t('Needs Review')}</option>
						<option value="ready">{$i18n.t('Ready')}</option>
					</select>
					<!-- Page size -->
					<select bind:value={perPage} on:change={() => { companyPage = 0; loadCompanyInvoices(); }} class="text-xs rounded-lg px-2 py-1 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300">
						<option value={20}>20</option>
						<option value={50}>50</option>
						<option value={100}>100</option>
					</select>
					<button class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition" on:click={() => { showUnassigned = !showUnassigned; if (showUnassigned) loadUnassignedInvoices(); }}>
						{showUnassigned ? $i18n.t('Hide Unassigned') : $i18n.t('Browse Unassigned')}
					</button>
				</div>
			</div>

			<input type="text" placeholder={$i18n.t('Search by vendor, invoice #, client...')} bind:value={searchCompany} on:input={onSearchCompany} class="w-full mb-2 px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />

			<!-- Filter toggle -->
			<div class="flex items-center gap-2 mb-2">
				<button class="text-xs px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition" on:click={() => showCompanyFilters = !showCompanyFilters}>
					{$i18n.t('Filters')} {showCompanyFilters ? '\u25B2' : '\u25BC'}
				</button>
				{#if dateFrom || dateTo || minAmount || maxAmount}
					<span class="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">{$i18n.t('Active')}</span>
				{/if}
			</div>

			{#if showCompanyFilters}
				<div class="grid grid-cols-2 md:grid-cols-5 gap-2 mb-2 p-2 bg-gray-50 dark:bg-gray-850 rounded-lg">
					<div>
						<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('From')}</label>
						<input type="date" bind:value={dateFrom} on:change={onCompanyFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
					</div>
					<div>
						<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('To')}</label>
						<input type="date" bind:value={dateTo} on:change={onCompanyFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
					</div>
					<div>
						<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('Min Amount')}</label>
						<input type="number" step="0.01" placeholder="0.00" bind:value={minAmount} on:change={onCompanyFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
					</div>
					<div>
						<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('Max Amount')}</label>
						<input type="number" step="0.01" placeholder="0.00" bind:value={maxAmount} on:change={onCompanyFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
					</div>
					<div>
						<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('Sort')}</label>
						<select value={companySortPreset} on:change={(e) => setCompanySortPreset(e.target.value)} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200">
							<option value="date_desc">{$i18n.t('Date (newest)')}</option>
							<option value="date_asc">{$i18n.t('Date (oldest)')}</option>
							<option value="amount_desc">{$i18n.t('Amount (highest)')}</option>
							<option value="amount_asc">{$i18n.t('Amount (lowest)')}</option>
							<option value="vendor_asc">{$i18n.t('Vendor (A-Z)')}</option>
							<option value="vendor_desc">{$i18n.t('Vendor (Z-A)')}</option>
						</select>
					</div>
				</div>
				<div class="flex justify-end mb-2">
					<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition" on:click={clearCompanyFilters}>{$i18n.t('Clear Filters')}</button>
				</div>
			{/if}

			<!-- Bulk bar -->
			{#if selectedCompanyIds.size > 0}
				<div class="flex items-center gap-3 mb-2 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg text-sm">
					<span class="font-medium text-blue-700 dark:text-blue-300">{selectedCompanyIds.size} {$i18n.t('selected')}</span>
					<button class="px-2 py-0.5 text-xs rounded bg-red-600 text-white hover:bg-red-700 transition" on:click={handleBulkUnassign}>{$i18n.t('Unassign Selected')}</button>
					<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300" on:click={() => { selectedCompanyIds = new Set(); }}>{$i18n.t('Clear')}</button>
				</div>
			{/if}

			{#if filteredCompany.length === 0}
				<div class="text-center text-sm text-gray-500 py-6">{$i18n.t('No invoices assigned to this company')}</div>
			{:else}
				<div class="overflow-x-auto">
					<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
						<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
							<tr class="border-b border-gray-50 dark:border-gray-850/30">
								<th class="px-2 py-2 w-8"><input type="checkbox" checked={allCompanySelected} on:change={toggleSelectAllCompany} class="rounded" /></th>
								<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('invoice_number', 'company')}>{$i18n.t('Invoice #')}{sortIcon('invoice_number', companySortBy, companySortDir)}</th>
								<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('vendor_name', 'company')}>{$i18n.t('Vendor')}{sortIcon('vendor_name', companySortBy, companySortDir)}</th>
								<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('client_name', 'company')}>{$i18n.t('Client')}{sortIcon('client_name', companySortBy, companySortDir)}</th>
								<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('invoice_date', 'company')}>{$i18n.t('Date')}{sortIcon('invoice_date', companySortBy, companySortDir)}</th>
								<th class="px-2 py-2 text-right cursor-pointer select-none" on:click={() => toggleSort('total_amount', 'company')}>{$i18n.t('Amount')}{sortIcon('total_amount', companySortBy, companySortDir)}</th>
								<th class="px-2 py-2">{$i18n.t('Status')}</th>
								<th class="px-2 py-2">{$i18n.t('Account')}</th>
										<th class="px-2 py-2">{$i18n.t('Journal')}</th>
								<th class="px-2 py-2">{$i18n.t('Payment')}</th>
								<th class="px-2 py-2"></th>
							</tr>
						</thead>
						<tbody>
							{#each filteredCompany as inv (inv.id)}
								<tr class="text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition border-b border-gray-50/50 dark:border-gray-850/30 cursor-pointer" on:click={() => { expandedId = expandedId === inv.id ? null : inv.id; }}>
									<td class="px-2 py-1.5" on:click|stopPropagation><input type="checkbox" checked={selectedCompanyIds.has(inv.id)} on:change={() => toggleCompanySelect(inv.id)} class="rounded" /></td>
									<td class="px-2 py-1.5 font-mono">{#if inv.k4mi_document_id}<a href="{K4MI_BASE_URL}/documents/{inv.k4mi_document_id}/details" target="_blank" rel="noopener" class="text-blue-600 dark:text-blue-400 hover:underline" title={$i18n.t('Open in K4mi')} on:click|stopPropagation>{inv.invoice_number ?? '-'}</a>{:else}{inv.invoice_number ?? '-'}{/if}</td>
									<td class="px-2 py-1.5 max-w-[130px] truncate">{inv.vendor_name ?? '-'}</td>
									<td class="px-2 py-1.5 max-w-[130px] truncate">{inv.client_name ?? '-'}</td>
									<td class="px-2 py-1.5">{inv.invoice_date ?? '-'}</td>
									<td class="px-2 py-1.5 text-right font-mono">
										{#key $displayCurrency}
										{#if isConverting}
											{@const c = cvt(inv.total_amount, inv.invoice_date)}
											{#if c.hasRate}
												<span class="font-medium">{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
												<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
											{:else}
												<span>{c.original} {nativeCurrency}</span>
												<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
											{/if}
										{:else}
											{fmt(inv.total_amount)} <span class="text-[10px] text-gray-400 ml-0.5">{inv.currency || nativeCurrency}</span>
										{/if}
										{/key}
									</td>
									<td class="px-2 py-1.5">
										{#if inv.needs_review}<span class="text-yellow-600 dark:text-yellow-400 font-medium">Review</span>
										{:else}<span class="text-green-600 dark:text-green-400">Ready</span>{/if}
									</td>
									<td class="px-2 py-1.5 relative" on:click|stopPropagation>
										{#if accountPickerInvoiceId === inv.id}
											<select
												class="text-[10px] w-36 rounded px-1 py-0.5 border border-blue-400 bg-white dark:bg-gray-900 dark:text-gray-200 outline-none"
												on:change={(e) => handleManualAccountSelect(inv.id, e.currentTarget.value)}
												on:blur={() => { accountPickerInvoiceId = null; }}
											>
												<option value="">{$i18n.t('Select account...')}</option>
												{#each accountsList as acct}
													<option value={acct.code}>{acct.code} — {acct.name}</option>
												{/each}
											</select>
										{:else if inv.final_account_code}
											<button class="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 hover:bg-green-200 dark:hover:bg-green-800/50 transition cursor-pointer" title="{accountLabel(inv.final_account_code)} — {$i18n.t('click to change')}" on:click={() => { accountPickerInvoiceId = inv.id; }}>{inv.final_account_code}</button>
										{:else if inv.suggested_account_code}
											<button class="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800/50 transition" title="{$i18n.t('AI suggestion')}: {accountLabel(inv.suggested_account_code)}" on:click={() => { accountPickerInvoiceId = inv.id; }}>{inv.suggested_account_code} ?</button>
										{:else}
											<button class="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 transition cursor-pointer" title={$i18n.t('Click to assign account')} on:click={() => { accountPickerInvoiceId = inv.id; }}>{$i18n.t('Assign')}</button>
										{/if}
									</td>
									<td class="px-2 py-1.5" on:click|stopPropagation>
										{#if inv.transaction_id}
											<span class="text-[10px] px-1.5 py-0.5 rounded {inv.transaction_status === 'posted' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : inv.transaction_status === 'draft' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400' : 'bg-gray-100 dark:bg-gray-800 text-gray-500'}" title="{inv.transaction_status === 'draft' ? $i18n.t('Draft — review and post in Entries tab') : $i18n.t('Posted')}">
												{inv.transaction_status === 'draft' ? $i18n.t('Draft') : $i18n.t('Posted')}
											</span>
										{:else}
											<span class="text-gray-300 dark:text-gray-600 text-[10px]">—</span>
										{/if}
									</td>
									<td class="px-2 py-1.5" on:click|stopPropagation>
									{#if inv.payment_status === 'paid'}
										<span class="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">{$i18n.t('Paid')}</span>
									{:else if inv.transaction_id}
										<button class="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-800/50 transition" on:click={() => suggestPayment(inv)}>
											{$i18n.t('Pay')}
										</button>
									{:else}
										<span class="text-gray-300 dark:text-gray-600 text-[10px]">—</span>
									{/if}
								</td>
								<td class="px-2 py-1.5 whitespace-nowrap" on:click|stopPropagation>
										<button
											class="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 mr-1"
											title={$i18n.t('Preview')}
											on:click={() => openPreview(inv)}
										>
											<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5">
												<path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
												<path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
											</svg>
										</button>
										<button class="text-xs text-red-500 hover:text-red-700" on:click={() => { unassignInvoice(companyId, inv.id).then(() => { toast.success($i18n.t('Removed')); reloadAll(); }).catch((e) => toast.error(`${e}`)); }}>{$i18n.t('Remove')}</button>
									</td>
								</tr>
								{#if expandedId === inv.id}
									<tr class="bg-gray-50/50 dark:bg-gray-850/30">
										<td colspan="12" class="px-4 py-3 text-xs">
										<div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Vendor')}</label>
												<input type="text" value={inv.vendor_name ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'vendor_name', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Client')}</label>
												<input type="text" value={inv.client_name ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'client_name', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Invoice #')}</label>
												<input type="text" value={inv.invoice_number ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'invoice_number', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Invoice Date')}</label>
												<input type="date" value={inv.invoice_date ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'invoice_date', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Due Date')}</label>
												<input type="date" value={inv.due_date ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'due_date', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Total Amount')}</label>
												<input type="number" step="0.01" value={inv.total_amount ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'total_amount', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Subtotal')}</label>
												<input type="number" step="0.01" value={inv.subtotal ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'subtotal', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Tax Amount')}</label>
												<input type="number" step="0.01" value={inv.tax_amount ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'tax_amount', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Currency')}</label>
												<input type="text" maxlength="3" value={inv.currency ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition uppercase" on:blur={(e) => saveField(inv.id, 'currency', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Description')}</label>
												<input type="text" value={inv.description ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'description', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('PO#')}</label>
												<input type="text" value={inv.po_number ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'po_number', e.currentTarget.value)} />
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-0.5">{$i18n.t('Business Unit')}</label>
												<input type="text" value={inv.business_unit ?? ''} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 dark:text-gray-200 focus:outline-none focus:border-blue-500 transition" on:blur={(e) => saveField(inv.id, 'business_unit', e.currentTarget.value)} />
											</div>
										</div>
									</td>
									</tr>
								{/if}
							{/each}
						</tbody>
					</table>
				</div>
				{#if companyTotal > perPage}
					<div class="flex justify-between items-center mt-3 text-xs text-gray-500">
						<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={companyPage === 0} on:click={() => { companyPage--; loadCompanyInvoices(); }}>{$i18n.t('Previous')}</button>
						<span>{$i18n.t('Page')} {companyPage + 1} / {Math.ceil(companyTotal / perPage)}</span>
						<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={(companyPage + 1) * perPage >= companyTotal} on:click={() => { companyPage++; loadCompanyInvoices(); }}>{$i18n.t('Next')}</button>
					</div>
				{/if}
			{/if}
		</div>

		<!-- Unassigned Panel -->
		{#if showUnassigned}
			<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30">
				<div class="text-sm font-medium dark:text-gray-200 mb-3">{$i18n.t('Unassigned Invoices')} ({unassignedTotal})</div>

				<input type="text" placeholder={$i18n.t('Search unassigned...')} bind:value={searchUnassigned} on:input={onSearchUnassigned} class="w-full mb-2 px-3 py-1.5 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />

				<!-- Filter toggle -->
				<div class="flex items-center gap-2 mb-2">
					<button class="text-xs px-2 py-1 rounded-lg border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition" on:click={() => showUnassignedFilters = !showUnassignedFilters}>
						{$i18n.t('Filters')} {showUnassignedFilters ? '\u25B2' : '\u25BC'}
					</button>
					{#if dateFrom || dateTo || minAmount || maxAmount}
						<span class="text-[10px] px-1.5 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">{$i18n.t('Active')}</span>
					{/if}
				</div>

				{#if showUnassignedFilters}
					<div class="grid grid-cols-2 md:grid-cols-5 gap-2 mb-2 p-2 bg-gray-50 dark:bg-gray-850 rounded-lg">
						<div>
							<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('From')}</label>
							<input type="date" bind:value={dateFrom} on:change={onUnassignedFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
						</div>
						<div>
							<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('To')}</label>
							<input type="date" bind:value={dateTo} on:change={onUnassignedFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
						</div>
						<div>
							<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('Min Amount')}</label>
							<input type="number" step="0.01" placeholder="0.00" bind:value={minAmount} on:change={onUnassignedFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
						</div>
						<div>
							<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('Max Amount')}</label>
							<input type="number" step="0.01" placeholder="0.00" bind:value={maxAmount} on:change={onUnassignedFilterChange} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200 focus:outline-none focus:border-blue-500" />
						</div>
						<div>
							<label class="text-[10px] text-gray-500 dark:text-gray-400 block mb-0.5">{$i18n.t('Sort')}</label>
							<select value={unassignedSortPreset} on:change={(e) => setUnassignedSortPreset(e.target.value)} class="w-full text-xs px-2 py-1 rounded border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-200">
								<option value="date_desc">{$i18n.t('Date (newest)')}</option>
								<option value="date_asc">{$i18n.t('Date (oldest)')}</option>
								<option value="amount_desc">{$i18n.t('Amount (highest)')}</option>
								<option value="amount_asc">{$i18n.t('Amount (lowest)')}</option>
								<option value="vendor_asc">{$i18n.t('Vendor (A-Z)')}</option>
								<option value="vendor_desc">{$i18n.t('Vendor (Z-A)')}</option>
							</select>
						</div>
					</div>
					<div class="flex justify-end mb-2">
						<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition" on:click={clearUnassignedFilters}>{$i18n.t('Clear Filters')}</button>
					</div>
				{/if}

				{#if selectedUnassignedIds.size > 0}
					<div class="flex items-center gap-3 mb-2 px-3 py-2 bg-green-50 dark:bg-green-900/20 rounded-lg text-sm">
						<span class="font-medium text-green-700 dark:text-green-300">{selectedUnassignedIds.size} {$i18n.t('selected')}</span>
						<button class="px-2 py-0.5 text-xs rounded bg-blue-600 text-white hover:bg-blue-700 transition" on:click={handleBulkAssign}>{$i18n.t('Assign Selected')}</button>
						<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300" on:click={() => { selectedUnassignedIds = new Set(); }}>{$i18n.t('Clear')}</button>
					</div>
				{/if}

				{#if unassignedInvoices.length === 0}
					<div class="text-center text-sm text-gray-500 py-6">{$i18n.t('No unassigned invoices')}</div>
				{:else}
					<div class="overflow-x-auto">
						<table class="w-full text-sm text-left text-gray-500 dark:text-gray-400 table-auto">
							<thead class="text-xs text-gray-800 uppercase bg-transparent dark:text-gray-200">
								<tr class="border-b border-gray-50 dark:border-gray-850/30">
									<th class="px-2 py-2 w-8"><input type="checkbox" checked={allUnassignedSelected} on:change={toggleSelectAllUnassigned} class="rounded" /></th>
									<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('invoice_number', 'unassigned')}>{$i18n.t('Invoice #')}{sortIcon('invoice_number', unassignedSortBy, unassignedSortDir)}</th>
									<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('vendor_name', 'unassigned')}>{$i18n.t('Vendor')}{sortIcon('vendor_name', unassignedSortBy, unassignedSortDir)}</th>
									<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('client_name', 'unassigned')}>{$i18n.t('Client')}{sortIcon('client_name', unassignedSortBy, unassignedSortDir)}</th>
									<th class="px-2 py-2 cursor-pointer select-none" on:click={() => toggleSort('invoice_date', 'unassigned')}>{$i18n.t('Date')}{sortIcon('invoice_date', unassignedSortBy, unassignedSortDir)}</th>
									<th class="px-2 py-2 text-right cursor-pointer select-none" on:click={() => toggleSort('total_amount', 'unassigned')}>{$i18n.t('Amount')}{sortIcon('total_amount', unassignedSortBy, unassignedSortDir)}</th>
									<th class="px-2 py-2"></th>
								</tr>
							</thead>
							<tbody>
								{#each unassignedInvoices as inv (inv.id)}
									<tr class="text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition border-b border-gray-50/50 dark:border-gray-850/30">
										<td class="px-2 py-1.5"><input type="checkbox" checked={selectedUnassignedIds.has(inv.id)} on:change={() => toggleUnassignedSelect(inv.id)} class="rounded" /></td>
										<td class="px-2 py-1.5 font-mono">{#if inv.k4mi_document_id}<a href="{K4MI_BASE_URL}/documents/{inv.k4mi_document_id}/details" target="_blank" rel="noopener" class="text-blue-600 dark:text-blue-400 hover:underline" title={$i18n.t('Open in K4mi')} on:click|stopPropagation>{inv.invoice_number ?? '-'}</a>{:else}{inv.invoice_number ?? '-'}{/if}</td>
										<td class="px-2 py-1.5 max-w-[130px] truncate">{inv.vendor_name ?? '-'}</td>
										<td class="px-2 py-1.5 max-w-[130px] truncate">{inv.client_name ?? '-'}</td>
										<td class="px-2 py-1.5">{inv.invoice_date ?? '-'}</td>
										<td class="px-2 py-1.5 text-right font-mono">
											{#key $displayCurrency}
											{#if isConverting}
												{@const c = cvt(inv.total_amount, inv.invoice_date)}
												{#if c.hasRate}
													<span class="font-medium">{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
													<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
												{:else}
													<span>{c.original} {nativeCurrency}</span>
													<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
												{/if}
											{:else}
												{fmt(inv.total_amount)} <span class="text-[10px] text-gray-400 ml-0.5">{inv.currency || nativeCurrency}</span>
											{/if}
											{/key}
										</td>
										<td class="px-2 py-1.5 whitespace-nowrap">
											<button
												class="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 mr-1"
												title={$i18n.t('Preview')}
												on:click|stopPropagation={() => openPreview(inv)}
											>
												<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5">
													<path stroke-linecap="round" stroke-linejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
													<path stroke-linecap="round" stroke-linejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
												</svg>
											</button>
											<button class="px-2 py-0.5 text-xs font-medium rounded bg-blue-600 text-white hover:bg-blue-700 transition" on:click={() => { assignInvoiceToCompany(companyId, inv.id).then(() => { toast.success($i18n.t('Assigned')); reloadAll(); }).catch((e) => toast.error(`${e}`)); }}>{$i18n.t('Assign')}</button>
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
					{#if unassignedTotal > perPage}
						<div class="flex justify-between items-center mt-3 text-xs text-gray-500">
							<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={unassignedPage === 0} on:click={() => { unassignedPage--; loadUnassignedInvoices(); }}>{$i18n.t('Previous')}</button>
							<span>{$i18n.t('Page')} {unassignedPage + 1} / {Math.ceil(unassignedTotal / perPage)}</span>
							<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={(unassignedPage + 1) * perPage >= unassignedTotal} on:click={() => { unassignedPage++; loadUnassignedInvoices(); }}>{$i18n.t('Next')}</button>
						</div>
					{/if}
				{/if}
			</div>
		{/if}
	</div>

	<!-- Bank Match Panel (shown above PaymentFormModal when paying an invoice) -->
	{#if showPaymentModal}
		<div class="fixed inset-0 z-[50001] flex items-start justify-center pt-[5vh] pointer-events-none">
			<div class="pointer-events-auto w-full max-w-lg mx-4 mb-2">
				{#if bankMatchLoading}
					<div class="bg-white dark:bg-gray-900 rounded-xl border border-blue-200 dark:border-blue-800/50 p-3 shadow-lg flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
						<Spinner className="size-4" />
						<span>{$i18n.t('Searching for matching bank statement lines...')}</span>
					</div>
				{:else if bankMatchCandidates.length > 0}
					<div class="bg-white dark:bg-gray-900 rounded-xl border border-blue-200 dark:border-blue-800/50 p-3 shadow-lg space-y-2">
						<div class="text-xs font-medium text-gray-700 dark:text-gray-300">{$i18n.t('Match with bank statement line:')}</div>
						<div class="space-y-1 max-h-40 overflow-y-auto">
							{#each bankMatchCandidates as line}
								<button
									class="w-full text-left px-3 py-2 rounded-lg text-xs flex items-center justify-between gap-2 transition {selectedBankLineId === line.id ? 'bg-blue-100 dark:bg-blue-900/40 border border-blue-400 dark:border-blue-600' : 'bg-gray-50 dark:bg-gray-850 hover:bg-gray-100 dark:hover:bg-gray-800 border border-transparent'}"
									on:click={() => selectedBankLineId === line.id ? deselectBankLine() : selectBankLine(line)}
								>
									<div class="flex items-center gap-3 min-w-0">
										<span class="text-gray-500 dark:text-gray-400 whitespace-nowrap">{line.transaction_date}</span>
										<span class="truncate text-gray-700 dark:text-gray-300">{line.description ?? '—'}</span>
										{#if line._bankAccountName}
											<span class="text-[10px] px-1.5 py-0.5 rounded bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400 whitespace-nowrap">{line._bankAccountName}</span>
										{/if}
									</div>
									<span class="font-mono whitespace-nowrap {parseFloat(line.amount) < 0 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}">
										{parseFloat(line.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
										{#if line._bankAccountCurrency}
											<span class="text-gray-400 dark:text-gray-500 text-[10px] ml-0.5">{line._bankAccountCurrency}</span>
										{/if}
									</span>
								</button>
							{/each}
						</div>
						{#if selectedBankLineId}
							<div class="text-[10px] text-blue-600 dark:text-blue-400">{$i18n.t('Bank line selected — will be auto-matched after payment is saved')}</div>
						{/if}
					</div>
				{:else if !bankMatchLoading}
					<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-200 dark:border-gray-800 p-3 shadow-lg text-xs text-gray-500 dark:text-gray-400 italic">
						{$i18n.t('No matching bank lines found — record payment manually')}
					</div>
				{/if}
			</div>
		</div>
	{/if}

	<PaymentFormModal
		bind:show={showPaymentModal}
		accounts={[]}
		{companyId}
		prefill={paymentPrefill}
		on:save={handlePaymentSaved}
	/>

	<DocumentPreviewModal
		bind:show={showPreview}
		invoice={previewInvoice}
		onUpdate={handlePreviewUpdate}
	/>
{/if}
