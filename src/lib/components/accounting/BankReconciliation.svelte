<script lang="ts">
	import { onMount, onDestroy, getContext, tick } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';
	import {
		getBankAccounts,
		createBankAccount,
		getBankStatements,
		importBankStatement,
		autoMatchBankStatements,
		matchBankStatement,
		unmatchBankStatement,
		getAccounts,
		getExchangeRates,
		editBankStatementLine,
		getTransactions,
		downloadBankStatementTemplate,
		createMatchGroup
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import PaymentFormModal from '$lib/components/accounting/PaymentFormModal.svelte';
	import TransactionFormModal from '$lib/components/accounting/TransactionFormModal.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let bankAccounts: any[] = [];
	let accounts: any[] = [];
	let selectedBankId: number | null = null;
	let statements: any[] = [];
	let loading = true;
	let statementsLoading = false;

	// New bank account form
	let showNewBank = false;
	let newBankName = '';
	let newBankAccountId: number | null = null;

	// Filter
	let statusFilter = '';
	let searchQuery = '';
	let dateFrom = '';
	let dateTo = '';
	let directionFilter: '' | 'credit' | 'debit' = '';

	// Import modal
	let showImportModal = false;
	let importCurrency = '';
	let importFileInput: HTMLInputElement;
	let importing = false;

	// Inline reference editing
	let editingRefLineId: number | null = null;
	let editRefValue = '';

	// Auto-reconcile loading
	let autoReconciling = false;

	// Expandable matched details row
	let expandedLineId: number | null = null;

	// Match popover — unified entry list with optional multi-select
	let matchingLineId: number | null = null;
	let matchSearchQuery = '';
	let matchCandidates: any[] = [];
	let matchLoading = false;
	let matchMultiSelect = false; // toggle for multi-select mode
	let matchSelectedIds: Set<number> = new Set(); // selected transaction IDs for multi-match
	let matchCreating = false;

	// Multi-BSL selection for N:1 matching
	let selectedBslIds: Set<number> = new Set();
	let showMultiMatchModal = false;
	let multiMatchInvoices: any[] = [];
	let multiMatchAllocations: Map<number, number> = new Map();
	let multiMatchLoading = false;
	let multiMatchCreating = false;

	// Create entry modal
	let showCreateEntryModal = false;
	let createEntryBslId: number | null = null;

	// Payment modal state
	let showPaymentModal = false;
	let paymentPrefill: any = null;
	let payingLineId: number | null = null;

	const handlePay = (line: any) => {
		const amount = Math.abs(parseFloat(line.amount) || 0);
		const isInbound = parseFloat(line.amount) > 0;
		payingLineId = line.id;
		paymentPrefill = {
			payment_date: line.transaction_date || new Date().toISOString().slice(0, 10),
			amount,
			currency: selectedBankCurrency || 'EUR',
			direction: isInbound ? 'inbound' : 'outbound',
			method: 'bank_transfer',
			payer: isInbound ? (line.description || '') : '',
			payee: isInbound ? '' : (line.description || ''),
			reference: line.reference || line.description || '',
			invoice_id: line.matched_txn_invoice_id || null,
			_bank_statement_line_id: line.id
		};
		showPaymentModal = true;
	};

	const handlePaymentSaved = async () => {
		toast.success($i18n.t('Payment recorded'));
		showPaymentModal = false;
		payingLineId = null;
		await loadStatements();
	};

	const CURRENCIES =['EUR', 'USD', 'GBP', 'CNY', 'JPY', 'CHF', 'CAD', 'AUD', 'HKD', 'SGD', 'SEK', 'NOK', 'DKK', 'NZD', 'KRW', 'INR', 'BRL', 'ZAR', 'MXN', 'PLN', 'CZK', 'TRY', 'THB', 'TWD', 'MAD', 'XOF'];

	// Display currency conversion
	let displayCurrency = '';
	let exchangeRates: any[] = [];

	const loadExchangeRates = async () => {
		try {
			const data = await getExchangeRates({ company_id: companyId });
			exchangeRates = Array.isArray(data) ? data : [];
		} catch { exchangeRates = []; }
	};

	const convertAmount = (amount: number, fromCurrency: string, date: string): { converted: number | null; rate: number | null } => {
		if (!displayCurrency || displayCurrency === fromCurrency || !fromCurrency) return { converted: null, rate: null };
		const candidates = exchangeRates.filter(r =>
			(r.from_currency === fromCurrency && r.to_currency === displayCurrency) ||
			(r.from_currency === displayCurrency && r.to_currency === fromCurrency)
		);
		if (candidates.length === 0) return { converted: null, rate: null };
		const sorted = [...candidates].sort((a, b) => {
			const da = Math.abs(new Date(a.effective_date).getTime() - new Date(date).getTime());
			const db = Math.abs(new Date(b.effective_date).getTime() - new Date(date).getTime());
			return da - db;
		});
		const best = sorted[0];
		let rate: number;
		if (best.from_currency === fromCurrency && best.to_currency === displayCurrency) {
			rate = parseFloat(best.rate);
		} else {
			rate = 1 / parseFloat(best.rate);
		}
		return { converted: amount * rate, rate };
	};

	const load = async () => {
		loading = true;
		try {
			const [baData, acctData] = await Promise.all([
				getBankAccounts(companyId),
				getAccounts({ company_id: companyId }),
			]);
			bankAccounts = baData ?? [];
			const accts = acctData?.accounts ?? acctData ?? [];
			accounts = Array.isArray(accts) ? accts : [];
			if (bankAccounts.length > 0 && !selectedBankId) {
				selectedBankId = bankAccounts[0].id;
				await loadStatements();
			}
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		loading = false;
	};

	const loadStatements = async () => {
		if (!selectedBankId) return;
		statementsLoading = true;
		try {
			const params: Record<string, any> = {};
			if (statusFilter) params.status = statusFilter;
			statements = await getBankStatements(selectedBankId, params) ?? [];
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		statementsLoading = false;
	};

	onMount(async () => {
		await Promise.all([load(), loadExchangeRates()]);
	});

	const handleCreateBank = async () => {
		if (!newBankName) return;
		try {
			await createBankAccount(companyId, { name: newBankName, account_id: newBankAccountId });
			toast.success($i18n.t('Bank account created'));
			showNewBank = false;
			newBankName = '';
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	// Import modal handlers
	const openImportModal = () => {
		importCurrency = selectedBankCurrency || 'EUR';
		showImportModal = true;
	};

	const handleImport = async () => {
		if (!selectedBankId || !importFileInput?.files?.[0]) {
			toast.error($i18n.t('Please select a CSV file'));
			return;
		}
		importing = true;
		try {
			const currency = importCurrency || selectedBankCurrency || undefined;
			const res = await importBankStatement(selectedBankId, importFileInput.files[0], currency);
			toast.success($i18n.t(`Imported ${res.imported} lines`));
			showImportModal = false;
			await loadStatements();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		importing = false;
	};

	// Auto-reconcile
	const handleAutoReconcile = async () => {
		if (!selectedBankId) return;
		autoReconciling = true;
		try {
			const res = await autoMatchBankStatements(selectedBankId);
			toast.success($i18n.t(`Matched ${res.matched} lines, ${res.remaining} remaining`));
			await loadStatements();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		autoReconciling = false;
	};

	const handleUnmatch = async (lineId: number) => {
		const line = statements.find((s: any) => s.id === lineId);
		if (line?.payment_id) {
			const ok = confirm($i18n.t('This line has a recorded payment. Unmatching will keep the payment but break the reconciliation link. Continue?'));
			if (!ok) return;
		}
		try {
			await unmatchBankStatement(lineId);
			await loadStatements();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	// Reference inline editing
	const startEditRef = (line: any) => {
		editingRefLineId = line.id;
		editRefValue = line.reference ?? '';
	};

	const saveEditRef = async () => {
		if (editingRefLineId === null) return;
		const original = statements.find(s => s.id === editingRefLineId);
		if (!original) return;
		if (editRefValue !== (original.reference ?? '')) {
			try {
				await editBankStatementLine(editingRefLineId, { reference: editRefValue });
				toast.success($i18n.t('Reference updated'));
				await loadStatements();
			} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		}
		editingRefLineId = null;
	};

	const handleRefKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') editingRefLineId = null;
		if (e.key === 'Enter') saveEditRef();
	};

	// Manual match popover — unified entry list with optional multi-select
	const openMatchPopover = async (line: any) => {
		matchingLineId = line.id;
		matchSearchQuery = '';
		matchMultiSelect = false;
		matchSelectedIds = new Set();
		matchLoading = true;
		matchCandidates = [];
		try {
			const res = await getTransactions({ company_id: companyId, status: 'posted', limit: 100 });
			const all = res?.transactions ?? res ?? [];
			const txns = Array.isArray(all) ? all : [];

			// Sum allocated amounts per transaction across all BSLs' match groups
			const txnAllocated = new Map<number, number>();
			for (const s of statements) {
				if (s.match_groups) {
					for (const mg of s.match_groups) {
						if (mg.transactions) {
							for (const t of mg.transactions) {
								txnAllocated.set(t.transaction_id, (txnAllocated.get(t.transaction_id) || 0) + parseFloat(t.allocated_amount || 0));
							}
						}
					}
				}
			}

			const bankLine = statements.find(s => s.id === matchingLineId);
			const bankAmount = bankLine ? Math.abs(parseFloat(bankLine.amount)) : 0;
			const bankAmountRaw = bankLine ? parseFloat(bankLine.amount) : 0;
			const isCredit = bankAmountRaw > 0;
			const bankDesc = (bankLine?.description ?? '').toLowerCase();
			const bankRef = (bankLine?.reference ?? '').toLowerCase();

			matchCandidates = txns
				.map((t: any) => {
					const totalDebit = (t.lines ?? []).reduce((sum: number, l: any) => sum + parseFloat(l.debit || 0), 0);
					const allocated = txnAllocated.get(t.id) || 0;
					const remaining = totalDebit - allocated;
					const amountMatch = bankAmount > 0 && Math.abs(remaining - bankAmount) < 0.02;
					const txnType = (t.transaction_type ?? '').toLowerCase();
					const directionMatch = isCredit
						? ['invoice', 'sale', 'payment_in', 'others'].includes(txnType)
						: ['bill', 'purchase', 'payment_out', 'others'].includes(txnType);
					const txnDesc = (t.description ?? '').toLowerCase();
					const txnRef = (t.reference ?? '').toLowerCase();
					let nameScore = 0;
					const bankWords = (bankDesc + ' ' + bankRef).split(/\s+/).filter((w: string) => w.length > 2);
					const txnWords = (txnDesc + ' ' + txnRef).split(/\s+/).filter((w: string) => w.length > 2);
					if (bankWords.length > 0 && txnWords.length > 0) {
						const matches = bankWords.filter((bw: string) => txnWords.some((tw: string) => tw.includes(bw) || bw.includes(tw)));
						nameScore = matches.length / Math.max(bankWords.length, 1);
					}
					const bankDate = bankLine?.transaction_date ?? '';
					const txnDate = t.transaction_date ?? '';
					const dateDiff = bankDate && txnDate ? Math.abs(new Date(bankDate).getTime() - new Date(txnDate).getTime()) / 86400000 : 999;
					let score = 0;
					if (amountMatch) score += 0.40;
					if (directionMatch) score += 0.20;
					score += nameScore * 0.25;
					score += Math.max(0, (1 - dateDiff / 60)) * 0.15;
					return { ...t, _totalAmount: totalDebit, _allocated: allocated, _remaining: remaining, _amountMatch: amountMatch, _directionMatch: directionMatch, _nameScore: nameScore, _dateDiff: dateDiff, _score: score };
				})
				.filter((t: any) => t._remaining > 0.01) // hide fully-covered entries
				.sort((a: any, b: any) => b._score - a._score);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
			matchCandidates = [];
		}
		matchLoading = false;
	};

	// Toggle entry selection in multi-select mode
	const toggleMatchEntry = (txnId: number) => {
		if (matchSelectedIds.has(txnId)) {
			matchSelectedIds.delete(txnId);
		} else {
			matchSelectedIds.add(txnId);
		}
		matchSelectedIds = new Set(matchSelectedIds);
	};

	// Single-select: match one entry to the BSL
	const handleMatchSelect = async (transactionId: number) => {
		if (matchingLineId === null) return;
		try {
			await matchBankStatement(matchingLineId, transactionId);
			toast.success($i18n.t('Statement line matched'));
			matchingLineId = null;
			await loadStatements();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	// Multi-select: match multiple entries to the BSL
	const handleMatchMultiple = async () => {
		if (matchingLineId === null || matchSelectedIds.size === 0) return;
		matchCreating = true;
		try {
			const selectedTxns = matchCandidates.filter((t: any) => matchSelectedIds.has(t.id));
			const txnAllocations = selectedTxns.map((t: any) => ({
				transaction_id: t.id,
				allocated_amount: t._totalAmount,
			}));
			const bslAmount = txnAllocations.reduce((s: number, a: any) => s + a.allocated_amount, 0);
			await createMatchGroup(companyId, {
				bsl_allocations: [{ bank_statement_line_id: matchingLineId, allocated_amount: bslAmount }],
				transaction_allocations: txnAllocations,
			});

			toast.success($i18n.t(`Matched ${matchSelectedIds.size} entries`));
			matchingLineId = null;
			matchSelectedIds = new Set();
			matchMultiSelect = false;
			await loadStatements();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		matchCreating = false;
	};

	$: matchSelectedTotal = Array.from(matchSelectedIds).reduce((sum, id) => {
		const txn = matchCandidates.find((t: any) => t.id === id);
		return sum + (txn?._totalAmount ?? 0);
	}, 0);

	$: filteredMatchCandidates = matchCandidates.filter((item: any) => {
		if (!matchSearchQuery) return true;
		const q = matchSearchQuery.toLowerCase();
		return (item.reference ?? '').toLowerCase().includes(q)
			|| (item.description ?? '').toLowerCase().includes(q)
			|| String(item._totalAmount ?? item.total ?? '').includes(q)
			|| String(item.id).includes(q);
	}).slice(0, 20);

	// Multi-BSL selection helpers
	const toggleBslSelection = (lineId: number) => {
		if (selectedBslIds.has(lineId)) {
			selectedBslIds.delete(lineId);
		} else {
			selectedBslIds.add(lineId);
		}
		selectedBslIds = new Set(selectedBslIds);
	};

	$: selectedBslTotal = Array.from(selectedBslIds).reduce((sum, id) => {
		const line = statements.find((s: any) => s.id === id);
		return sum + (line ? Math.abs(parseFloat(line.amount)) - parseFloat(line.allocated_total || 0) : 0);
	}, 0);

	const openMultiMatchModal = async () => {
		if (selectedBslIds.size < 1) return;
		showMultiMatchModal = true;
		multiMatchAllocations = new Map();
		multiMatchLoading = true;
		try {
			const res = await getTransactions({ company_id: companyId, status: 'posted', limit: 200 });
			const all = res?.transactions ?? res ?? [];

			// Sum allocated amounts per transaction across all match groups
			const txnAllocated = new Map<number, number>();
			for (const s of statements) {
				if (s.match_groups) {
					for (const mg of s.match_groups) {
						if (mg.transactions) {
							for (const t of mg.transactions) {
								txnAllocated.set(t.transaction_id, (txnAllocated.get(t.transaction_id) || 0) + parseFloat(t.allocated_amount || 0));
							}
						}
					}
				}
			}

			multiMatchInvoices = (Array.isArray(all) ? all : [])
				.map((t: any) => {
					const totalDebit = (t.lines ?? []).reduce((sum: number, l: any) => sum + parseFloat(l.debit || 0), 0);
					const allocated = txnAllocated.get(t.id) || 0;
					const remaining = totalDebit - allocated;
					return { ...t, _totalAmount: totalDebit, _remaining: remaining };
				})
				.filter((t: any) => t._remaining > 0.01);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
			multiMatchInvoices = [];
		}
		multiMatchLoading = false;
	};

	$: multiMatchAllocTotal = Array.from(multiMatchAllocations.values()).reduce((s, v) => s + v, 0);
	$: multiMatchRemaining = selectedBslTotal - multiMatchAllocTotal;

	const toggleMultiMatchEntry = (txn: any) => {
		if (multiMatchAllocations.has(txn.id)) {
			multiMatchAllocations.delete(txn.id);
		} else {
			const remaining = selectedBslTotal - multiMatchAllocTotal;
			multiMatchAllocations.set(txn.id, Math.min(txn._totalAmount, remaining));
		}
		multiMatchAllocations = new Map(multiMatchAllocations);
	};

	const handleMultiMatchCreate = async () => {
		if (selectedBslIds.size === 0 || multiMatchAllocations.size === 0) return;
		multiMatchCreating = true;
		try {
			const bslAllocs = Array.from(selectedBslIds).map(id => {
				const line = statements.find((s: any) => s.id === id);
				const unalloc = line ? Math.abs(parseFloat(line.amount)) - parseFloat(line.allocated_total || 0) : 0;
				return { bank_statement_line_id: id, allocated_amount: unalloc };
			});
			const txnAllocs = Array.from(multiMatchAllocations.entries()).map(([txn_id, amt]) => ({
				transaction_id: txn_id,
				allocated_amount: amt,
			}));
			await createMatchGroup(companyId, {
				bsl_allocations: bslAllocs,
				transaction_allocations: txnAllocs,
			});
			toast.success($i18n.t('Match group created'));
			showMultiMatchModal = false;
			selectedBslIds = new Set();
			multiMatchAllocations = new Map();
			await loadStatements();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		multiMatchCreating = false;
	};

	// Close match popover on outside click
	const handleDocumentClick = (e: MouseEvent) => {
		if (matchingLineId !== null) {
			const target = e.target as HTMLElement;
			if (!target.closest('.match-popover') && !target.closest('.match-trigger')) {
				matchingLineId = null;
			}
		}
	};

	onMount(() => {
		document.addEventListener('click', handleDocumentClick);
	});
	onDestroy(() => {
		document.removeEventListener('click', handleDocumentClick);
	});

	// Create entry from unmatched line — opens the full TransactionFormModal
	const openCreateEntry = (line: any) => {
		createEntryBslId = line.id;
		showCreateEntryModal = true;
	};

	const handleCreateEntrySave = async () => {
		showCreateEntryModal = false;
		createEntryBslId = null;
		await loadStatements();
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '—';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const fmtSigned = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const statusColor = (s: string) => {
		switch (s) {
			case 'auto_matched': return 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400';
			case 'manual_matched': return 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400';
			case 'partial_matched': return 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400';
			case 'excluded': return 'bg-gray-100 dark:bg-gray-800 text-gray-500';
			default: return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400';
		}
	};

	const statusLabel = (s: string) => {
		switch (s) {
			case 'auto_matched': return 'Auto';
			case 'manual_matched': return 'Matched';
			case 'partial_matched': return 'Partial';
			case 'excluded': return 'Excluded';
			default: return 'Unmatched';
		}
	};

	$: matchedCount = statements.filter(s => ['auto_matched', 'manual_matched'].includes(s.match_status)).length;
	$: partialCount = statements.filter(s => s.match_status === 'partial_matched').length;
	$: unmatchedCount = statements.filter(s => s.match_status === 'unmatched').length;
	$: selectedBankCurrency = bankAccounts.find(ba => ba.id === selectedBankId)?.currency || '';
	$: if (selectedBankCurrency && !displayCurrency) displayCurrency = selectedBankCurrency;
	$: showConverted = displayCurrency && displayCurrency !== selectedBankCurrency;

	// Summary card computations
	$: bankBalance = statements.length > 0
		? statements.reduce((sum, s) => sum + (parseFloat(s.amount) || 0), 0)
		: 0;
	$: glBalance = (() => {
		// Approximate: sum of matched transaction amounts
		// In a real scenario, this would come from the API, but we approximate from statement data
		return statements
			.filter(s => s.match_status !== 'unmatched')
			.reduce((sum, s) => sum + (parseFloat(s.amount) || 0), 0);
	})();
	$: reconDifference = bankBalance - glBalance;
	$: isReconciled = Math.abs(reconDifference) < 0.01;

	// Client-side search & filter on loaded statements
	$: filteredStatements = statements.filter((s: any) => {
		// Text search
		if (searchQuery) {
			const q = searchQuery.toLowerCase();
			const match = (s.description ?? '').toLowerCase().includes(q)
				|| (s.reference ?? '').toLowerCase().includes(q)
				|| String(s.amount ?? '').includes(q)
				|| (s.transaction_date ?? '').includes(q);
			if (!match) return false;
		}
		// Date range
		if (dateFrom && s.transaction_date < dateFrom) return false;
		if (dateTo && s.transaction_date > dateTo) return false;
		// Direction
		if (directionFilter === 'credit' && parseFloat(s.amount) <= 0) return false;
		if (directionFilter === 'debit' && parseFloat(s.amount) >= 0) return false;
		return true;
	});

	// Import modal portal
	let importModalEl: HTMLDivElement | null = null;
	let importMounted = false;
	onMount(() => { importMounted = true; });

	$: if (importMounted) {
		if (showImportModal && importModalEl) {
			document.body.appendChild(importModalEl);
			document.body.style.overflow = 'hidden';
		} else if (importModalEl) {
			try { document.body.removeChild(importModalEl); } catch {}
			document.body.style.overflow = 'unset';
		}
	}

	const handleGlobalKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') {
			if (showMultiMatchModal) showMultiMatchModal = false;
			else if (showImportModal) showImportModal = false;
			else if (showCreateEntryModal) showCreateEntryModal = false;
		}
	};

	onMount(() => {
		window.addEventListener('keydown', handleGlobalKeydown);
	});
	onDestroy(() => {
		window.removeEventListener('keydown', handleGlobalKeydown);
		if (importModalEl) try { document.body.removeChild(importModalEl); } catch {}
	});
</script>

<!-- Import Statement Modal -->
{#if showImportModal}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		bind:this={importModalEl}
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-[50000] overflow-hidden overscroll-contain"
		in:fade={{ duration: 10 }}
		on:mousedown={() => { showImportModal = false; }}
	>
		<div
			class="m-auto max-w-full w-[28rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[90dvh] shadow-3xl border border-white dark:border-gray-900 overflow-y-auto"
			in:flyAndScale
			on:mousedown={(e) => { e.stopPropagation(); }}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{$i18n.t('Import Statement')}
				</div>

				<div class="space-y-4">
					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
							{$i18n.t('Bank Statement (CSV or Excel)')} *
						</label>
						<input
							bind:this={importFileInput}
							type="file"
							accept=".csv,.xlsx,.xls"
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition file:mr-3 file:py-1 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-blue-50 file:text-blue-700 dark:file:bg-blue-900/30 dark:file:text-blue-300"
						/>
					</div>

					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
							{$i18n.t('Currency')}
						</label>
						<select
							bind:value={importCurrency}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						>
							{#each CURRENCIES as c}
								<option value={c}>{c}</option>
							{/each}
						</select>
						<p class="text-[10px] text-gray-400 dark:text-gray-500 mt-1">{$i18n.t('Defaults to the bank account currency')}</p>
					</div>

					<button
						type="button"
						class="w-full flex items-center justify-center gap-1.5 px-3 py-2 text-sm rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-medium transition"
						on:click={downloadBankStatementTemplate}
					>
						<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-4">
							<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" />
						</svg>
						{$i18n.t('Download Sample Template (.xlsx)')}
					</button>
				</div>

				<div class="mt-6 flex justify-between gap-1.5">
					<button
						class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
						on:click={() => { showImportModal = false; }}
						type="button"
						disabled={importing}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
						on:click={handleImport}
						type="button"
						disabled={importing}
					>
						{#if importing}
							{$i18n.t('Importing...')}
						{:else}
							{$i18n.t('Import')}
						{/if}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

<!-- Create Entry Modal (reuses the same form as the Entries tab) -->
<TransactionFormModal
	bind:show={showCreateEntryModal}
	transaction={null}
	{accounts}
	{companyId}
	bankStatementLineId={createEntryBslId}
	on:save={handleCreateEntrySave}
/>

<div class="space-y-3">
	<!-- Header: Bank selector + actions -->
	<div class="flex items-center justify-between flex-wrap gap-2">
		<div class="flex items-center gap-2">
			{#if bankAccounts.length > 0}
				<select
					bind:value={selectedBankId}
					on:change={() => { displayCurrency = ''; loadStatements(); }}
					class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
				>
					{#each bankAccounts as ba}
						<option value={ba.id}>{ba.name}{ba.currency ? ` (${ba.currency})` : ''}</option>
					{/each}
				</select>
				{#if selectedBankCurrency}
					<span class="text-xs font-medium text-gray-600 dark:text-gray-400 px-2 py-1 rounded bg-gray-100 dark:bg-gray-800">
						{selectedBankCurrency}
					</span>
				{/if}
				<!-- View in currency selector -->
				<div class="flex items-center gap-1 ml-2">
					<span class="text-[10px] uppercase text-gray-400 dark:text-gray-500 font-medium">{$i18n.t('View in')}</span>
					<select
						bind:value={displayCurrency}
						class="text-xs rounded-lg px-2 py-1 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
					>
						<option value={selectedBankCurrency}>{selectedBankCurrency || '—'} ({$i18n.t('native')})</option>
						{#each CURRENCIES.filter(c => c !== selectedBankCurrency) as c}
							<option value={c}>{c}</option>
						{/each}
					</select>
				</div>
			{/if}
			<button class="text-xs text-blue-600 hover:text-blue-700" on:click={() => (showNewBank = !showNewBank)}>
				{showNewBank ? $i18n.t('Cancel') : $i18n.t('+ New Bank')}
			</button>
		</div>
		{#if selectedBankId}
			<div class="flex items-center gap-2">
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
					on:click={openImportModal}
				>
					{$i18n.t('Import Statement')}
				</button>
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition disabled:opacity-50 flex items-center gap-1.5"
					on:click={handleAutoReconcile}
					disabled={autoReconciling}
				>
					{#if autoReconciling}
						<Spinner className="size-3" />
					{/if}
					{$i18n.t('Auto-Reconcile')}
				</button>
			</div>
		{/if}
	</div>

	{#if showNewBank}
		<div class="p-3 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-800 flex gap-2 items-end">
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Name')}</label>
				<input type="text" bind:value={newBankName} placeholder="Main Checking" class="text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
			</div>
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('GL Account')}</label>
				<select bind:value={newBankAccountId} class="text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
					<option value={null}>—</option>
					{#each accounts.filter(a => a.account_type === 'asset') as acct}
						<option value={acct.id}>{acct.code} — {acct.name}</option>
					{/each}
				</select>
			</div>
			<button class="px-3 py-1.5 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition" on:click={handleCreateBank}>{$i18n.t('Create')}</button>
		</div>
	{/if}

	{#if selectedBankId}
		<!-- Auto-Reconcile AI Loading Banner -->
		{#if autoReconciling}
			<div class="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-950/30 dark:to-indigo-950/30 border border-blue-100 dark:border-blue-900/30">
				<Spinner className="size-4 text-blue-600 dark:text-blue-400" />
				<span class="text-sm font-medium text-blue-700 dark:text-blue-300">{$i18n.t('AI is matching bank statement lines to GL transactions...')}</span>
			</div>
		{/if}

		<!-- Reconciliation Summary Cards -->
		<div class="grid grid-cols-2 md:grid-cols-4 gap-3">
			<div class="rounded-xl p-3 bg-white dark:bg-gray-900 border border-gray-100/50 dark:border-gray-850/50 shadow-xs">
				<div class="text-[10px] uppercase font-medium text-gray-400 dark:text-gray-500 mb-1">{$i18n.t('Bank Balance')}</div>
				<div class="text-lg font-semibold text-gray-800 dark:text-gray-200">{fmtSigned(bankBalance)}</div>
				<div class="text-[10px] text-gray-400 dark:text-gray-500">{selectedBankCurrency}</div>
			</div>
			<div class="rounded-xl p-3 bg-white dark:bg-gray-900 border border-gray-100/50 dark:border-gray-850/50 shadow-xs">
				<div class="text-[10px] uppercase font-medium text-gray-400 dark:text-gray-500 mb-1">{$i18n.t('GL Balance')}</div>
				<div class="text-lg font-semibold text-gray-800 dark:text-gray-200">{fmtSigned(glBalance)}</div>
				<div class="text-[10px] text-gray-400 dark:text-gray-500">{$i18n.t('Matched lines')}</div>
			</div>
			<div class="rounded-xl p-3 bg-white dark:bg-gray-900 border border-gray-100/50 dark:border-gray-850/50 shadow-xs">
				<div class="text-[10px] uppercase font-medium text-gray-400 dark:text-gray-500 mb-1">{$i18n.t('Difference')}</div>
				<div class="text-lg font-semibold {Math.abs(reconDifference) < 0.01 ? 'text-green-600 dark:text-green-400' : 'text-amber-600 dark:text-amber-400'}">{fmtSigned(reconDifference)}</div>
				<div class="text-[10px] text-gray-400 dark:text-gray-500">{$i18n.t('Bank - GL')}</div>
			</div>
			<div class="rounded-xl p-3 border shadow-xs {isReconciled ? 'bg-green-50 dark:bg-green-950/20 border-green-200 dark:border-green-900/30' : 'bg-amber-50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-900/30'}">
				<div class="text-[10px] uppercase font-medium {isReconciled ? 'text-green-500 dark:text-green-400' : 'text-amber-500 dark:text-amber-400'} mb-1">{$i18n.t('Status')}</div>
				<div class="text-lg font-semibold {isReconciled ? 'text-green-700 dark:text-green-300' : 'text-amber-700 dark:text-amber-300'}">
					{isReconciled ? $i18n.t('Reconciled') : $i18n.t('Unreconciled')}
				</div>
				{#if !isReconciled}
					<div class="text-[10px] text-amber-500 dark:text-amber-400">{unmatchedCount} {$i18n.t('lines remaining')}</div>
				{:else}
					<div class="text-[10px] text-green-500 dark:text-green-400">{matchedCount}/{statements.length} {$i18n.t('matched')}</div>
				{/if}
			</div>
		</div>

		<!-- Filter tabs + Search -->
		<div class="flex flex-wrap gap-2 items-center">
			<div class="flex gap-1">
				{#each [['', 'All'], ['unmatched', 'Unmatched'], ['partial_matched', 'Partial'], ['auto_matched', 'Auto'], ['manual_matched', 'Manual'], ['excluded', 'Excluded']] as [val, label]}
					<button
						class="px-2 py-1 text-xs rounded {statusFilter === val ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'} transition"
						on:click={() => { statusFilter = val; loadStatements(); }}
					>{$i18n.t(label)}</button>
				{/each}
			</div>
			<div class="flex gap-1">
				{#each [['', 'All'], ['credit', 'Credit (In)'], ['debit', 'Debit (Out)']] as [val, label]}
					<button
						class="px-2 py-1 text-xs rounded {directionFilter === val ? 'bg-emerald-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'} transition"
						on:click={() => { directionFilter = val; }}
					>{$i18n.t(label)}</button>
				{/each}
			</div>
			<input
				type="text"
				bind:value={searchQuery}
				placeholder={$i18n.t('Search description, reference, amount...')}
				class="flex-1 min-w-[180px] text-xs rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden focus:border-blue-500 transition"
			/>
			<input type="date" bind:value={dateFrom} class="text-xs rounded-lg px-2 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" placeholder={$i18n.t('From')} />
			<input type="date" bind:value={dateTo} class="text-xs rounded-lg px-2 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" placeholder={$i18n.t('To')} />
			{#if searchQuery || dateFrom || dateTo || directionFilter}
				<button
					class="px-2 py-1 text-xs rounded bg-red-50 text-red-600 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-400 dark:hover:bg-red-900/30 transition"
					on:click={() => { searchQuery = ''; dateFrom = ''; dateTo = ''; directionFilter = ''; }}
				>{$i18n.t('Clear')}</button>
			{/if}
		</div>
		{#if searchQuery || dateFrom || dateTo || directionFilter}
			<div class="text-[10px] text-gray-400">{filteredStatements.length} / {statements.length} {$i18n.t('lines')}</div>
		{/if}

		{#if statementsLoading}
			<div class="flex justify-center my-6"><Spinner className="size-5" /></div>
		{:else if statements.length === 0}
			<div class="text-sm text-gray-400 italic text-center py-6">{$i18n.t('No statement lines. Import a CSV to get started.')}</div>
		{:else}
			{#if selectedBslIds.size > 0}
				<div class="px-3 py-2 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-t-xl flex items-center gap-3 text-xs">
					<span class="font-medium text-blue-700 dark:text-blue-300">{selectedBslIds.size} {$i18n.t('line(s) selected')}</span>
					<span class="text-blue-600 dark:text-blue-400">{$i18n.t('Total')}: {fmt(selectedBslTotal)}</span>
					<button
						class="px-3 py-1 rounded-lg bg-blue-600 text-white hover:bg-blue-700 font-medium transition"
						on:click={openMultiMatchModal}
					>{$i18n.t('Match Selected to Entry(s)')}</button>
					<button
						class="px-2 py-1 rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
						on:click={() => { selectedBslIds = new Set(); }}
					>{$i18n.t('Clear')}</button>
				</div>
			{/if}
			<div class="overflow-x-auto bg-white dark:bg-gray-900 {selectedBslIds.size > 0 ? 'rounded-b-xl' : 'rounded-xl'} border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-1 py-2 w-8">
								<input
									type="checkbox"
									class="rounded border-gray-300 dark:border-gray-600"
									checked={selectedBslIds.size > 0 && selectedBslIds.size === filteredStatements.filter(s => ['unmatched', 'partial_matched'].includes(s.match_status)).length}
									on:change={(e) => {
										if (e.currentTarget.checked) {
											selectedBslIds = new Set(filteredStatements.filter(s => ['unmatched', 'partial_matched'].includes(s.match_status)).map(s => s.id));
										} else {
											selectedBslIds = new Set();
										}
									}}
								/>
							</th>
							<th class="px-2 py-2">{$i18n.t('#')}</th>
							<th class="px-2 py-2">{$i18n.t('Date')}</th>
							<th class="px-2 py-2">{$i18n.t('Description')}</th>
							<th class="px-2 py-2">{$i18n.t('Reference')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Debit')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Credit')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Currency')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Status')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Actions')}</th>
						</tr>
					</thead>
					<tbody>
						{#if filteredStatements.length === 0 && statements.length > 0}
							<tr><td colspan="10" class="px-4 py-6 text-center text-xs text-gray-400 italic">{$i18n.t('No lines match your filters.')}</td></tr>
						{/if}
						{#each filteredStatements as line (line.id)}
							<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
								<td class="px-1 py-1.5">
									{#if ['unmatched', 'partial_matched'].includes(line.match_status)}
										<input
											type="checkbox"
											class="rounded border-gray-300 dark:border-gray-600"
											checked={selectedBslIds.has(line.id)}
											on:change={() => toggleBslSelection(line.id)}
										/>
									{/if}
								</td>
								<td class="px-2 py-1.5 whitespace-nowrap text-gray-400 font-mono text-[10px]">{line.id}</td>
								<td class="px-2 py-1.5 whitespace-nowrap">{line.transaction_date}</td>
								<td class="px-2 py-1.5 max-w-[200px] truncate" title={line.description ?? ''}>{line.description ?? '—'}</td>
								<td class="px-2 py-1.5 font-mono">
									{#if editingRefLineId === line.id}
										<input
											type="text"
											bind:value={editRefValue}
											on:keydown={handleRefKeydown}
											on:blur={saveEditRef}
											class="text-xs w-24 font-mono rounded px-1.5 py-0.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-blue-300 dark:border-blue-700 outline-hidden"
											autofocus
										/>
									{:else}
										<span class="inline-flex items-center gap-1">
											{line.reference ?? '—'}
											<button
												class="text-gray-300 hover:text-blue-600 dark:text-gray-600 dark:hover:text-blue-400 transition"
												on:click={() => startEditRef(line)}
												title={$i18n.t('Edit reference')}
											>
												<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" /></svg>
											</button>
										</span>
									{/if}
								</td>
								<!-- Debit (outflow) -->
								<td class="px-2 py-1.5 text-right font-mono text-red-600 dark:text-red-400">
									{#if parseFloat(line.amount) < 0}
										{fmt(Math.abs(parseFloat(line.amount)))}
									{/if}
								</td>
								<!-- Credit (inflow) -->
								<td class="px-2 py-1.5 text-right font-mono text-green-600 dark:text-green-400">
									{#if parseFloat(line.amount) > 0}
										{fmt(parseFloat(line.amount))}
									{/if}
								</td>
								<td class="px-2 py-1.5 text-center text-gray-500 dark:text-gray-400 font-mono text-[10px]">{showConverted ? displayCurrency : (selectedBankCurrency || '—')}</td>
								<td class="px-2 py-1.5">
									<div class="flex items-center gap-1">
										<span class="text-[10px] px-1.5 py-0.5 rounded font-medium {statusColor(line.match_status)}">
											{$i18n.t(statusLabel(line.match_status))}
										</span>
										{#if line.match_status === 'auto_matched' && line.match_confidence}
											<span class="text-[10px] text-gray-400">{Math.round(parseFloat(line.match_confidence) * 100)}%</span>
										{/if}
										{#if line.match_status === 'partial_matched'}
											<span class="text-[10px] text-orange-500 font-mono">{fmt(line.allocated_total)}/{fmt(Math.abs(parseFloat(line.amount)))}</span>
										{/if}
									</div>
									{#if line.matched_transaction_id || (line.match_groups && line.match_groups.length > 0)}
										<button
											class="mt-0.5 flex items-center gap-1 text-[10px] text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200 transition"
											on:click|stopPropagation={() => { expandedLineId = expandedLineId === line.id ? null : line.id; }}
										>
											<svg class="w-3 h-3 transition-transform {expandedLineId === line.id ? 'rotate-90' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" /></svg>
											{#if line.matched_transaction_id}
												<span class="font-mono font-medium">#{line.matched_transaction_id}</span>
											{/if}
											{#if line.match_groups && line.match_groups.length > 0}
												<span class="px-1 py-0 rounded bg-blue-50 dark:bg-blue-900/30 text-[9px]">{line.match_groups.length} {$i18n.t('group(s)')}</span>
											{/if}
										</button>
									{/if}
								</td>
								<td class="px-2 py-1.5 text-right whitespace-nowrap">
									{#if line.match_status === 'unmatched' || line.match_status === 'partial_matched'}
										<!-- Match button with popover -->
										<div class="inline-block relative">
											<button
												class="match-trigger px-2 py-0.5 text-[10px] font-medium rounded bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition"
												on:click|stopPropagation={() => {
													if (matchingLineId === line.id) { matchingLineId = null; }
													else { openMatchPopover(line); }
												}}
											>
												{line.match_status === 'partial_matched' ? $i18n.t('Match More') : $i18n.t('Match')} &#9660;
											</button>
											{#if matchingLineId === line.id}
												<!-- Full-width match panel below the row -->
											{/if}
										</div>
										{#if line.match_status === 'partial_matched'}
											<button class="ml-1 text-xs text-red-500 hover:text-red-700 transition" on:click={() => handleUnmatch(line.id)}>{$i18n.t('Unmatch')}</button>
										{:else}
											<button
												class="ml-1 px-2 py-0.5 text-[10px] font-medium rounded transition {line.suggested_bank_fee_rule ? 'bg-amber-100 text-amber-700 hover:bg-amber-200 dark:bg-amber-900/30 dark:text-amber-300 dark:hover:bg-amber-900/50' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700'}"
												on:click={() => openCreateEntry(line)}
											>
												{line.suggested_bank_fee_rule ? $i18n.t('Bank Fee') : $i18n.t('Create Entry')}
											</button>
										{/if}
									{:else}
										<button class="text-xs text-red-500 hover:text-red-700 transition" on:click={() => handleUnmatch(line.id)}>{$i18n.t('Unmatch')}</button>
										{#if line.payment_id}
											<span class="ml-1 px-2 py-0.5 text-[10px] font-medium rounded bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300">
												{$i18n.t('Posted')}
											</span>
										{:else}
											<button
												class="ml-1 px-2 py-0.5 text-[10px] font-medium rounded bg-amber-50 text-amber-700 hover:bg-amber-100 dark:bg-amber-900/30 dark:text-amber-300 dark:hover:bg-amber-900/50 transition"
												on:click|stopPropagation={() => handlePay(line)}
											>
												{$i18n.t('Post')}
											</button>
										{/if}
									{/if}
								</td>
							</tr>
							<!-- Expandable matched details row -->
							{#if expandedLineId === line.id && (line.matched_transaction_id || (line.match_groups && line.match_groups.length > 0))}
								<tr class="bg-gray-50/80 dark:bg-gray-850/50">
									<td colspan="10" class="px-4 py-2">
										<div class="text-[10px] uppercase font-medium text-gray-500 dark:text-gray-400 mb-1.5">{$i18n.t('Matched Entries')}</div>
										<div class="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
											<table class="w-full text-xs">
												<thead class="text-[10px] bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
													<tr>
														<th class="px-2 py-1 text-left">#</th>
														<th class="px-2 py-1 text-left">{$i18n.t('Date')}</th>
														<th class="px-2 py-1 text-left">{$i18n.t('Type')}</th>
														<th class="px-2 py-1 text-left">{$i18n.t('Reference')}</th>
														<th class="px-2 py-1 text-left">{$i18n.t('Description')}</th>
														<th class="px-2 py-1 text-right">{$i18n.t('Amount')}</th>
														<th class="px-2 py-1 text-left">{$i18n.t('Status')}</th>
													</tr>
												</thead>
												<tbody>
													{#if line.matched_transaction_id && line.matched_txn_type}
														<tr class="border-t border-gray-100 dark:border-gray-800">
															<td class="px-2 py-1 font-mono text-blue-600 dark:text-blue-400">
																<a href="/accounting/company/{companyId}/entries" class="hover:underline">#{line.matched_transaction_id}</a>
															</td>
															<td class="px-2 py-1 font-mono">{line.matched_txn_date ?? '—'}</td>
															<td class="px-2 py-1">
																<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{line.matched_txn_type}</span>
															</td>
															<td class="px-2 py-1 font-mono">{line.matched_txn_reference ?? '—'}</td>
															<td class="px-2 py-1 max-w-[250px] truncate" title={line.matched_txn_description ?? ''}>{line.matched_txn_description ?? '—'}</td>
															<td class="px-2 py-1 text-right font-mono font-medium">{fmt(Math.abs(parseFloat(line.amount)))}</td>
															<td class="px-2 py-1">
																<span class="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 font-medium">{$i18n.t('Matched')}</span>
															</td>
														</tr>
													{/if}
													{#if line.match_groups}
														{#each line.match_groups as mg}
															{#if mg.transactions && mg.transactions.length > 0}
																{#each mg.transactions as txn}
																	<tr class="border-t border-gray-100 dark:border-gray-800">
																		<td class="px-2 py-1 font-mono text-blue-600 dark:text-blue-400">
																			<a href="/accounting/company/{companyId}/entries" class="hover:underline">#{txn.transaction_id}</a>
																		</td>
																		<td class="px-2 py-1 font-mono">{txn.date ?? '—'}</td>
																		<td class="px-2 py-1">
																			<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{txn.type ?? '—'}</span>
																		</td>
																		<td class="px-2 py-1 font-mono">{txn.reference ?? '—'}</td>
																		<td class="px-2 py-1 max-w-[250px] truncate" title={txn.description ?? ''}>{txn.description ?? '—'}</td>
																		<td class="px-2 py-1 text-right font-mono font-medium">{fmt(txn.allocated_amount)}</td>
																		<td class="px-2 py-1">
																			<span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400 font-medium">{$i18n.t('Group')} #{mg.id}</span>
																		</td>
																	</tr>
																{/each}
															{:else}
																<tr class="border-t border-gray-100 dark:border-gray-800">
																	<td class="px-2 py-1 text-gray-400" colspan="5">
																		<span class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-50 dark:bg-indigo-900/20 text-indigo-600 dark:text-indigo-400">{$i18n.t('Group')} #{mg.id}</span>
																		<span class="ml-2">{$i18n.t('Allocated')}: {fmt(mg.allocated_amount)}</span>
																	</td>
																	<td class="px-2 py-1 text-right font-mono font-medium">{fmt(mg.allocated_amount)}</td>
																	<td class="px-2 py-1"></td>
																</tr>
															{/if}
														{/each}
													{/if}
													{#if !line.matched_transaction_id && (!line.match_groups || line.match_groups.length === 0)}
														<tr><td colspan="7" class="px-2 py-3 text-center text-gray-400 italic text-[10px]">{$i18n.t('No matched entries found.')}</td></tr>
													{/if}
												</tbody>
											</table>
										</div>
									</td>
								</tr>
							{/if}
							<!-- Match panel (shown below the row when matching) -->
							{#if matchingLineId === line.id}
								<tr class="match-popover bg-blue-50/50 dark:bg-blue-950/20">
									<td colspan="10" class="px-3 py-3">
										<div class="flex items-center justify-between mb-2">
											<div class="flex items-center gap-2">
												<span class="text-xs font-medium text-blue-700 dark:text-blue-300">{$i18n.t('Match to entry')}:</span>
												<span class="text-[10px] text-gray-500">{line.transaction_date} — {line.description} — {fmt(line.amount)}</span>
												{#if parseFloat(line.allocated_total || 0) > 0}
													<span class="text-[10px] text-orange-500">({$i18n.t('Unallocated')}: {fmt(line.unallocated_amount ?? (Math.abs(parseFloat(line.amount)) - parseFloat(line.allocated_total || 0)))})</span>
												{/if}
											</div>
											<div class="flex items-center gap-2">
												<!-- Multi-select toggle -->
												<label class="flex items-center gap-1 text-[10px] text-gray-500 cursor-pointer">
													<input type="checkbox" class="rounded border-gray-300 dark:border-gray-600" bind:checked={matchMultiSelect} on:change={() => { matchSelectedIds = new Set(); }} />
													{$i18n.t('Multi-select')}
												</label>
												<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300" on:click={() => { matchingLineId = null; }}>✕ {$i18n.t('Close')}</button>
											</div>
										</div>

										<!-- Multi-select summary bar -->
										{#if matchMultiSelect && matchSelectedIds.size > 0}
											<div class="mb-2 px-3 py-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg flex items-center justify-between text-xs">
												<div class="flex gap-4">
													<span>{$i18n.t('Selected')}: <strong>{matchSelectedIds.size}</strong> {$i18n.t('entries')}</span>
													<span>{$i18n.t('Total')}: <strong class="font-mono">{fmt(matchSelectedTotal)}</strong></span>
												</div>
												<button
													class="px-3 py-1 rounded-lg bg-blue-600 text-white hover:bg-blue-700 font-medium transition disabled:opacity-50"
													disabled={matchSelectedIds.size === 0 || matchCreating}
													on:click={handleMatchMultiple}
												>
													{#if matchCreating}
														{$i18n.t('Matching...')}
													{:else}
														{$i18n.t('Match Selected')}
													{/if}
												</button>
											</div>
										{/if}

										<div class="mb-2">
											<input
												type="text"
												bind:value={matchSearchQuery}
												placeholder={$i18n.t('Search by reference, description, amount, or #ID...')}
												class="w-full text-xs rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden focus:border-blue-500"
											/>
										</div>

										{#if matchLoading}
											<div class="flex justify-center py-4"><Spinner className="size-4" /></div>
										{:else if filteredMatchCandidates.length === 0}
											<div class="text-xs text-gray-400 italic text-center py-4">
												{$i18n.t('No posted entries found. Create a journal entry first, then match.')}
											</div>
										{:else}
											<div class="overflow-x-auto max-h-64 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700">
												<table class="w-full text-xs">
													<thead class="text-[10px] uppercase bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 sticky top-0">
														<tr>
															{#if matchMultiSelect}<th class="px-2 py-1.5 w-8"></th>{/if}
															<th class="px-2 py-1.5 text-left"></th>
															<th class="px-2 py-1.5 text-left">#</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Date')}</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Type')}</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Reference')}</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Description')}</th>
															<th class="px-2 py-1.5 text-right">{$i18n.t('Amount')}</th>
															{#if !matchMultiSelect}<th class="px-2 py-1.5"></th>{/if}
														</tr>
													</thead>
													<tbody>
														{#each filteredMatchCandidates as txn}
															{@const isSelected = matchSelectedIds.has(txn.id)}
															<tr class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-850 transition {txn._score >= 0.5 ? 'bg-green-50/30 dark:bg-green-900/10' : ''} {isSelected ? 'bg-blue-50/50 dark:bg-blue-900/20' : ''}">
																{#if matchMultiSelect}
																	<td class="px-2 py-1.5">
																		<input
																			type="checkbox"
																			class="rounded border-gray-300 dark:border-gray-600"
																			checked={isSelected}
																			on:change={() => toggleMatchEntry(txn.id)}
																		/>
																	</td>
																{/if}
																<td class="px-2 py-1.5">
																	<div class="flex gap-0.5 flex-wrap">
																		{#if txn._amountMatch}
																			<span class="text-[9px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 font-semibold">{$i18n.t('Amount')}</span>
																		{/if}
																		{#if txn._nameScore > 0.3}
																			<span class="text-[9px] px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400 font-semibold">{$i18n.t('Name')}</span>
																		{/if}
																		{#if txn.invoice_id}
																			<span class="text-[9px] px-1.5 py-0.5 rounded bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400 font-semibold">{$i18n.t('Invoice')}</span>
																		{/if}
																	</div>
																</td>
																<td class="px-2 py-1.5 font-mono text-gray-400">{txn.id}</td>
																<td class="px-2 py-1.5 font-mono">{txn.transaction_date ?? '—'}</td>
																<td class="px-2 py-1.5">
																	<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{txn.transaction_type ?? '—'}</span>
																</td>
																<td class="px-2 py-1.5 font-mono">{txn.reference ?? '—'}</td>
																<td class="px-2 py-1.5 max-w-[200px] truncate" title={txn.description ?? ''}>{txn.description ?? '—'}</td>
																<td class="px-2 py-1.5 text-right font-mono font-medium">{fmt(txn._totalAmount ?? txn.total ?? 0)}</td>
																{#if !matchMultiSelect}
																	<td class="px-2 py-1.5">
																		<button
																			class="px-2 py-0.5 text-[10px] font-medium rounded bg-blue-600 text-white hover:bg-blue-700 transition"
																			on:click={() => handleMatchSelect(txn.id)}
																		>{$i18n.t('Select')}</button>
																	</td>
																{/if}
															</tr>
														{/each}
													</tbody>
												</table>
											</div>
										{/if}
									</td>
								</tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		{/if}
	{:else if !loading}
		<div class="text-sm text-gray-400 italic text-center py-6">{$i18n.t('No bank accounts. Create one to start reconciling.')}</div>
	{/if}
</div>

<!-- Multi-BSL Match Modal -->
{#if showMultiMatchModal}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-[50000] overflow-hidden overscroll-contain"
		in:fade={{ duration: 10 }}
		on:mousedown={() => { showMultiMatchModal = false; }}
	>
		<div
			class="m-auto max-w-full w-[48rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[90dvh] shadow-3xl border border-white dark:border-gray-900 overflow-y-auto"
			in:flyAndScale
			on:mousedown={(e) => { e.stopPropagation(); }}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-2">
					{$i18n.t('Match Multiple Bank Lines to Entry(s)')}
				</div>
				<div class="text-xs text-gray-500 mb-4">
					{selectedBslIds.size} {$i18n.t('bank statement line(s)')} — {$i18n.t('Total')}: <strong class="font-mono">{fmt(selectedBslTotal)}</strong>
				</div>

				<!-- Allocation summary -->
				{#if multiMatchAllocations.size > 0}
					<div class="mb-3 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 rounded-lg flex items-center justify-between text-xs">
						<div class="flex gap-4">
							<span>{$i18n.t('BSL Total')}: <strong class="font-mono">{fmt(selectedBslTotal)}</strong></span>
							<span>{$i18n.t('Allocated')}: <strong class="font-mono text-green-600">{fmt(multiMatchAllocTotal)}</strong></span>
							<span>{$i18n.t('Remaining')}: <strong class="font-mono {multiMatchRemaining < 0.01 ? 'text-green-600' : 'text-orange-600'}">{fmt(multiMatchRemaining)}</strong></span>
						</div>
					</div>
				{/if}

				{#if multiMatchLoading}
					<div class="flex justify-center py-8"><Spinner className="size-5" /></div>
				{:else if multiMatchInvoices.length === 0}
					<div class="text-xs text-gray-400 italic text-center py-8">{$i18n.t('No posted entries found.')}</div>
				{:else}
					<div class="overflow-x-auto max-h-80 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700 mb-4">
						<table class="w-full text-xs">
							<thead class="text-[10px] uppercase bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 sticky top-0">
								<tr>
									<th class="px-2 py-1.5 w-8"></th>
									<th class="px-2 py-1.5 text-left">#</th>
									<th class="px-2 py-1.5 text-left">{$i18n.t('Date')}</th>
									<th class="px-2 py-1.5 text-left">{$i18n.t('Type')}</th>
									<th class="px-2 py-1.5 text-left">{$i18n.t('Reference')}</th>
									<th class="px-2 py-1.5 text-left">{$i18n.t('Description')}</th>
									<th class="px-2 py-1.5 text-right">{$i18n.t('Amount')}</th>
								</tr>
							</thead>
							<tbody>
								{#each multiMatchInvoices as txn}
									{@const isSelected = multiMatchAllocations.has(txn.id)}
									<tr class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-850 transition {isSelected ? 'bg-blue-50/50 dark:bg-blue-900/20' : ''}">
										<td class="px-2 py-1.5">
											<input
												type="checkbox"
												class="rounded border-gray-300 dark:border-gray-600"
												checked={isSelected}
												on:change={() => toggleMultiMatchEntry(txn)}
											/>
										</td>
										<td class="px-2 py-1.5 font-mono text-gray-400">{txn.id}</td>
										<td class="px-2 py-1.5 font-mono">{txn.transaction_date ?? '—'}</td>
										<td class="px-2 py-1.5">
											<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{txn.transaction_type ?? '—'}</span>
										</td>
										<td class="px-2 py-1.5 font-mono">{txn.reference ?? '—'}</td>
										<td class="px-2 py-1.5 max-w-[200px] truncate" title={txn.description ?? ''}>{txn.description ?? '—'}</td>
										<td class="px-2 py-1.5 text-right font-mono font-medium">{fmt(txn._totalAmount ?? 0)}</td>
									</tr>
								{/each}
							</tbody>
						</table>
					</div>
				{/if}

				<div class="mt-2 flex justify-between gap-1.5">
					<button
						class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
						on:click={() => { showMultiMatchModal = false; }}
						type="button"
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
						on:click={handleMultiMatchCreate}
						type="button"
						disabled={multiMatchAllocations.size === 0 || multiMatchCreating}
					>
						{#if multiMatchCreating}
							{$i18n.t('Creating...')}
						{:else}
							{$i18n.t('Create Match Group')}
						{/if}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

<PaymentFormModal
	bind:show={showPaymentModal}
	{accounts}
	{companyId}
	prefill={paymentPrefill}
	on:save={handlePaymentSaved}
/>
