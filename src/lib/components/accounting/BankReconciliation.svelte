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
		createTransaction,
		postTransaction,
		downloadBankStatementTemplate
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

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

	// Match popover
	let matchingLineId: number | null = null;
	let matchSearchQuery = '';
	let matchCandidates: any[] = [];
	let matchLoading = false;

	// Create entry modal
	let showCreateEntryModal = false;
	let createEntryLine: any = null;
	let createEntryDebitAccountId: number | null = null;
	let createEntryCreditAccountId: number | null = null;
	let createEntryDescription = '';
	let createEntryDate = '';
	let createEntryAmount: number = 0;
	let creatingEntry = false;

	const CURRENCIES = ['EUR', 'USD', 'GBP', 'CNY', 'JPY', 'CHF', 'CAD', 'AUD', 'HKD', 'SGD', 'SEK', 'NOK', 'DKK', 'NZD', 'KRW', 'INR', 'BRL', 'ZAR', 'MXN', 'PLN', 'CZK', 'TRY', 'THB', 'TWD', 'MAD', 'XOF'];

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
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	const loadStatements = async () => {
		if (!selectedBankId) return;
		statementsLoading = true;
		try {
			const params: Record<string, any> = {};
			if (statusFilter) params.status = statusFilter;
			statements = await getBankStatements(selectedBankId, params) ?? [];
		} catch (err) { toast.error(`${err}`); }
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
		} catch (err) { toast.error(`${err}`); }
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
		} catch (err) { toast.error(`${err}`); }
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
		} catch (err) { toast.error(`${err}`); }
		autoReconciling = false;
	};

	const handleUnmatch = async (lineId: number) => {
		try {
			await unmatchBankStatement(lineId);
			await loadStatements();
		} catch (err) { toast.error(`${err}`); }
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
			} catch (err) { toast.error(`${err}`); }
		}
		editingRefLineId = null;
	};

	const handleRefKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') editingRefLineId = null;
		if (e.key === 'Enter') saveEditRef();
	};

	// Manual match popover
	const openMatchPopover = async (line: any) => {
		matchingLineId = line.id;
		matchSearchQuery = '';
		matchLoading = true;
		try {
			const res = await getTransactions({
				company_id: companyId,
				status: 'posted',
				limit: 100
			});
			const all = res?.transactions ?? res ?? [];
			const txns = Array.isArray(all) ? all : [];

			// Find the bank line we're matching
			const bankLine = statements.find(s => s.id === matchingLineId);
			const bankAmount = bankLine ? Math.abs(parseFloat(bankLine.amount)) : 0;

			// Sort candidates: exact amount matches first, then by date proximity
			matchCandidates = txns
				.filter((t: any) => {
					// Only show transactions that touch account 512 (bank)
					const has512 = (t.lines ?? []).some((l: any) => l.account_code === '512');
					return has512;
				})
				.map((t: any) => {
					const line512 = (t.lines ?? []).find((l: any) => l.account_code === '512');
					const glAmount = line512 ? Math.abs(parseFloat(line512.debit || 0) - parseFloat(line512.credit || 0)) : 0;
					const amountMatch = bankAmount > 0 && Math.abs(glAmount - bankAmount) < 0.02;
					return { ...t, _glAmount: glAmount, _amountMatch: amountMatch };
				})
				.sort((a: any, b: any) => {
					// Amount matches first
					if (a._amountMatch && !b._amountMatch) return -1;
					if (!a._amountMatch && b._amountMatch) return 1;
					// Then by date (most recent first)
					return (b.transaction_date ?? '').localeCompare(a.transaction_date ?? '');
				});
		} catch (err) {
			toast.error(`${err}`);
			matchCandidates = [];
		}
		matchLoading = false;
	};

	const handleMatchSelect = async (transactionId: number) => {
		if (matchingLineId === null) return;
		try {
			await matchBankStatement(matchingLineId, transactionId);
			toast.success($i18n.t('Statement line matched'));
			matchingLineId = null;
			await loadStatements();
		} catch (err) { toast.error(`${err}`); }
	};

	$: filteredMatchCandidates = matchCandidates.filter((t: any) => {
		if (!matchSearchQuery) return true;
		const q = matchSearchQuery.toLowerCase();
		return (t.reference ?? '').toLowerCase().includes(q)
			|| (t.description ?? '').toLowerCase().includes(q)
			|| String(t.total ?? '').includes(q)
			|| String(t.id).includes(q);
	}).slice(0, 15);

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

	// Create entry from unmatched line
	const openCreateEntry = (line: any) => {
		createEntryLine = line;
		createEntryDate = line.transaction_date ?? '';
		createEntryDescription = line.description ?? '';
		createEntryAmount = Math.abs(parseFloat(line.amount) || 0);
		const isExpense = parseFloat(line.amount) < 0;

		const bankAcct = bankAccounts.find(ba => ba.id === selectedBankId);
		const bankGlAccountId = bankAcct?.account_id ?? null;

		if (isExpense) {
			// DR expense / CR bank
			createEntryDebitAccountId = null; // user picks expense account
			createEntryCreditAccountId = bankGlAccountId;
		} else {
			// DR bank / CR revenue
			createEntryDebitAccountId = bankGlAccountId;
			createEntryCreditAccountId = null; // user picks revenue account
		}
		showCreateEntryModal = true;
	};

	const handleCreateEntry = async () => {
		if (!createEntryDebitAccountId || !createEntryCreditAccountId || !createEntryAmount || !createEntryDate) {
			toast.error($i18n.t('Please fill all required fields'));
			return;
		}
		creatingEntry = true;
		try {
			const data: Record<string, any> = {
				transaction_date: createEntryDate,
				description: createEntryDescription,
				reference: createEntryLine?.reference ?? '',
				type: 'journal',
				lines: [
					{ account_id: createEntryDebitAccountId, debit: createEntryAmount, credit: 0 },
					{ account_id: createEntryCreditAccountId, debit: 0, credit: createEntryAmount },
				]
			};
			const txn = await createTransaction(data, companyId);
			const txnId = txn?.id ?? txn?.transaction?.id;

			// Post the transaction
			if (txnId) {
				try {
					await postTransaction(txnId);
				} catch { /* posting might not be required */ }
			}

			// Auto-match the bank line to the new transaction
			if (txnId && createEntryLine?.id) {
				await matchBankStatement(createEntryLine.id, txnId);
			}

			toast.success($i18n.t('Entry created and matched'));
			showCreateEntryModal = false;
			createEntryLine = null;
			await loadStatements();
		} catch (err) { toast.error(`${err}`); }
		creatingEntry = false;
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
			case 'excluded': return 'bg-gray-100 dark:bg-gray-800 text-gray-500';
			default: return 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400';
		}
	};

	const statusLabel = (s: string) => {
		switch (s) {
			case 'auto_matched': return 'Auto';
			case 'manual_matched': return 'Matched';
			case 'excluded': return 'Excluded';
			default: return 'Unmatched';
		}
	};

	$: matchedCount = statements.filter(s => s.match_status !== 'unmatched').length;
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

	// Create entry modal portal
	let createEntryModalEl: HTMLDivElement | null = null;

	$: if (importMounted) {
		if (showCreateEntryModal && createEntryModalEl) {
			document.body.appendChild(createEntryModalEl);
			document.body.style.overflow = 'hidden';
		} else if (createEntryModalEl) {
			try { document.body.removeChild(createEntryModalEl); } catch {}
			document.body.style.overflow = 'unset';
		}
	}

	const handleGlobalKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') {
			if (showImportModal) showImportModal = false;
			else if (showCreateEntryModal) showCreateEntryModal = false;
		}
	};

	onMount(() => {
		window.addEventListener('keydown', handleGlobalKeydown);
	});
	onDestroy(() => {
		window.removeEventListener('keydown', handleGlobalKeydown);
		if (importModalEl) try { document.body.removeChild(importModalEl); } catch {}
		if (createEntryModalEl) try { document.body.removeChild(createEntryModalEl); } catch {}
	});

	// Account helpers for create entry modal
	$: expenseAccounts = accounts.filter(a => a.account_type === 'expense' || (a.code && a.code.startsWith('6')));
	$: revenueAccounts = accounts.filter(a => a.account_type === 'revenue' || (a.code && a.code.startsWith('7')));
	$: allAccountsSorted = [...accounts].sort((a, b) => (a.code ?? '').localeCompare(b.code ?? ''));
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
							{$i18n.t('CSV File')} *
						</label>
						<input
							bind:this={importFileInput}
							type="file"
							accept=".csv"
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
						{$i18n.t('Download Sample CSV Template')}
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

<!-- Create Entry Modal -->
{#if showCreateEntryModal && createEntryLine}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		bind:this={createEntryModalEl}
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-[50000] overflow-hidden overscroll-contain"
		in:fade={{ duration: 10 }}
		on:mousedown={() => { showCreateEntryModal = false; }}
	>
		<div
			class="m-auto max-w-full w-[32rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[90dvh] shadow-3xl border border-white dark:border-gray-900 overflow-y-auto"
			in:flyAndScale
			on:mousedown={(e) => { e.stopPropagation(); }}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-1">
					{$i18n.t('Create Journal Entry')}
				</div>
				<p class="text-xs text-gray-400 dark:text-gray-500 mb-4">
					{$i18n.t('Create a journal entry from this bank statement line and auto-match it.')}
				</p>

				<div class="space-y-3">
					<!-- Date + Amount -->
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Date')} *</label>
							<input
								type="date"
								bind:value={createEntryDate}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Amount')} *</label>
							<input
								type="number"
								step="0.01"
								bind:value={createEntryAmount}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
					</div>

					<!-- Description -->
					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Description')}</label>
						<input
							type="text"
							bind:value={createEntryDescription}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						/>
					</div>

					<!-- Account selection -->
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Debit Account')} *</label>
							<select
								bind:value={createEntryDebitAccountId}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							>
								<option value={null}>{$i18n.t('Select account...')}</option>
								{#each allAccountsSorted as acct}
									<option value={acct.id}>{acct.code} — {acct.name}</option>
								{/each}
							</select>
						</div>
						<div>
							<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Credit Account')} *</label>
							<select
								bind:value={createEntryCreditAccountId}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							>
								<option value={null}>{$i18n.t('Select account...')}</option>
								{#each allAccountsSorted as acct}
									<option value={acct.id}>{acct.code} — {acct.name}</option>
								{/each}
							</select>
						</div>
					</div>

					{#if parseFloat(createEntryLine.amount) < 0}
						<p class="text-[10px] text-gray-400 dark:text-gray-500 italic">
							{$i18n.t('Expense: debit an expense account (e.g. 627 Bank Fees), credit the bank account (512).')}
						</p>
					{:else}
						<p class="text-[10px] text-gray-400 dark:text-gray-500 italic">
							{$i18n.t('Income: debit the bank account (512), credit a revenue account (e.g. 7xx).')}
						</p>
					{/if}
				</div>

				<div class="mt-6 flex justify-between gap-1.5">
					<button
						class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
						on:click={() => { showCreateEntryModal = false; createEntryLine = null; }}
						type="button"
						disabled={creatingEntry}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
						on:click={handleCreateEntry}
						type="button"
						disabled={creatingEntry}
					>
						{#if creatingEntry}
							{$i18n.t('Creating...')}
						{:else}
							{$i18n.t('Create & Match')}
						{/if}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}

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

		<!-- Filter tabs -->
		<div class="flex gap-1">
			{#each [['', 'All'], ['unmatched', 'Unmatched'], ['auto_matched', 'Auto'], ['manual_matched', 'Manual']] as [val, label]}
				<button
					class="px-2 py-1 text-xs rounded {statusFilter === val ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'} transition"
					on:click={() => { statusFilter = val; loadStatements(); }}
				>{$i18n.t(label)}</button>
			{/each}
		</div>

		{#if statementsLoading}
			<div class="flex justify-center my-6"><Spinner className="size-5" /></div>
		{:else if statements.length === 0}
			<div class="text-sm text-gray-400 italic text-center py-6">{$i18n.t('No statement lines. Import a CSV to get started.')}</div>
		{:else}
			<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-2 py-2">{$i18n.t('Date')}</th>
							<th class="px-2 py-2">{$i18n.t('Description')}</th>
							<th class="px-2 py-2">{$i18n.t('Reference')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Amount')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Currency')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Status')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Actions')}</th>
						</tr>
					</thead>
					<tbody>
						{#each statements as line (line.id)}
							<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
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
								<td class="px-2 py-1.5 text-right font-mono {parseFloat(line.amount) < 0 ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'}">
									{#if showConverted}
										{@const conv = convertAmount(parseFloat(line.amount), selectedBankCurrency, line.transaction_date)}
										{#if conv.converted !== null}
											<div class="font-semibold">{fmt(conv.converted)} {displayCurrency}</div>
											<div class="text-[10px] text-gray-400 dark:text-gray-500 font-normal">({fmt(line.amount)} {selectedBankCurrency})</div>
										{:else}
											<div>{fmt(line.amount)}</div>
											<div class="text-[10px] text-yellow-500 dark:text-yellow-400 font-normal italic">{$i18n.t('no rate')}</div>
										{/if}
									{:else}
										{fmt(line.amount)}
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
									</div>
									{#if line.matched_transaction_id}
										<a
											href="/accounting/company/{companyId}/entries"
											class="mt-0.5 block text-[10px] text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-200 leading-tight hover:underline transition"
											title="{$i18n.t('View entry')} #{line.matched_transaction_id}"
										>
											<span class="font-mono font-medium">#{line.matched_transaction_id}</span>
											{#if line.matched_txn_type}
												<span class="ml-1 px-1 py-0 rounded bg-blue-50 dark:bg-blue-900/30 text-[9px]">{line.matched_txn_type}</span>
											{/if}
											{#if line.matched_txn_reference}
												<span class="ml-1 font-mono">{line.matched_txn_reference}</span>
											{/if}
											{#if line.matched_txn_description}
												<div class="text-[10px] text-gray-400 dark:text-gray-500 truncate max-w-[200px]" title={line.matched_txn_description}>
													{line.matched_txn_description}
												</div>
											{/if}
										</a>
									{/if}
								</td>
								<td class="px-2 py-1.5 text-right whitespace-nowrap">
									{#if line.match_status === 'unmatched'}
										<!-- Match button with popover -->
										<div class="inline-block relative">
											<button
												class="match-trigger px-2 py-0.5 text-[10px] font-medium rounded bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition"
												on:click|stopPropagation={() => {
													if (matchingLineId === line.id) { matchingLineId = null; }
													else { openMatchPopover(line); }
												}}
											>
												{$i18n.t('Match')} &#9660;
											</button>
											{#if matchingLineId === line.id}
												<!-- Full-width match panel below the row -->
											{/if}
										</div>
										<button
											class="ml-1 px-2 py-0.5 text-[10px] font-medium rounded bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700 transition"
											on:click={() => openCreateEntry(line)}
										>
											{$i18n.t('Create Entry')}
										</button>
									{:else}
										<button class="text-xs text-red-500 hover:text-red-700 transition" on:click={() => handleUnmatch(line.id)}>{$i18n.t('Unmatch')}</button>
									{/if}
								</td>
							</tr>
							<!-- Match panel (shown below the row when matching) -->
							{#if matchingLineId === line.id}
								<tr class="bg-blue-50/50 dark:bg-blue-950/20">
									<td colspan="7" class="px-3 py-3">
										<div class="flex items-center justify-between mb-2">
											<div class="flex items-center gap-2">
												<span class="text-xs font-medium text-blue-700 dark:text-blue-300">{$i18n.t('Select a transaction to match with')}:</span>
												<span class="text-[10px] text-gray-500">{line.transaction_date} — {line.description} — {fmt(line.amount)}</span>
											</div>
											<button class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300" on:click={() => { matchingLineId = null; }}>✕ {$i18n.t('Close')}</button>
										</div>
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
											<div class="text-xs text-gray-400 italic text-center py-4">{$i18n.t('No transactions found. Create a journal entry first, then match.')}</div>
										{:else}
											<div class="overflow-x-auto max-h-64 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700">
												<table class="w-full text-xs">
													<thead class="text-[10px] uppercase bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 sticky top-0">
														<tr>
															<th class="px-2 py-1.5 text-left"></th>
															<th class="px-2 py-1.5 text-left">#</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Date')}</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Type')}</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Reference')}</th>
															<th class="px-2 py-1.5 text-left">{$i18n.t('Description')}</th>
															<th class="px-2 py-1.5 text-right">{$i18n.t('Amount')}</th>
															<th class="px-2 py-1.5"></th>
														</tr>
													</thead>
													<tbody>
														{#each filteredMatchCandidates as txn}
															<tr class="border-b border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-850 transition {txn._amountMatch ? 'bg-green-50/30 dark:bg-green-900/10' : ''}">
																<td class="px-2 py-1.5">
																	{#if txn._amountMatch}
																		<span class="text-[9px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 font-semibold">{$i18n.t('Match')}</span>
																	{/if}
																</td>
																<td class="px-2 py-1.5 font-mono text-gray-400">{txn.id}</td>
																<td class="px-2 py-1.5 font-mono">{txn.transaction_date ?? '—'}</td>
																<td class="px-2 py-1.5">
																	<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{txn.transaction_type ?? '—'}</span>
																</td>
																<td class="px-2 py-1.5 font-mono">{txn.reference ?? '—'}</td>
																<td class="px-2 py-1.5 max-w-[200px] truncate" title={txn.description ?? ''}>{txn.description ?? '—'}</td>
																<td class="px-2 py-1.5 text-right font-mono font-medium">{fmt(txn.total ?? 0)}</td>
																<td class="px-2 py-1.5">
																	<button
																		class="px-2 py-0.5 text-[10px] font-medium rounded bg-blue-600 text-white hover:bg-blue-700 transition"
																		on:click={() => handleMatchSelect(txn.id)}
																	>{$i18n.t('Select')}</button>
																</td>
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
