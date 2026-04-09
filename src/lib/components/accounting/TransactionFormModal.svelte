<script lang="ts">
	import { getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import { createTransaction, updateTransaction, getJournalTemplates, createTemplateFromTransaction, uploadAttachment, getAccountingAiStatus, aiValidateTransaction, postTransaction, matchBankStatement } from '$lib/apis/accounting';
	import { INVOICE_API_BASE_URL } from '$lib/constants';
	import { K4MI_BASE_URL } from '$lib/constants';
	import InvoiceSelector from '$lib/components/accounting/InvoiceSelector.svelte';
	import AccountFormModal from '$lib/components/accounting/AccountFormModal.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let transaction: any = null;
	export let accounts: any[] = [];
	export let companyId: number;
	export let bankStatementLineId: number | null = null;  // when set, shows "Direct Match" option

	// Determine read-only mode for posted/voided transactions
	$: readOnly = transaction && (transaction.status === 'posted' || transaction.status === 'voided');

	// Direct match: when creating from the Bank tab, auto-post + match to BSL
	let directMatch = false;
	$: if (bankStatementLineId) directMatch = true;

	// Form state
	let formData: {
		transaction_date: string;
		transaction_type: string;
		currency: string;
		reference: string;
		description: string;
		invoice_id: number | null;
	} = {
		transaction_date: '',
		transaction_type: 'others',
		currency: 'USD',
		reference: '',
		description: '',
		invoice_id: null
	};

	let invoiceLabel = '';
	let invoiceK4miUrl = '';
	let showInvoiceSelector = false;

	// Journal templates
	let journalTemplates: any[] = [];
	let selectedTemplateId: number | null = null;
	let savingTemplate = false;

	async function loadTemplates() {
		try {
			const res = await getJournalTemplates(companyId);
			journalTemplates = Array.isArray(res) ? res : [];
		} catch { journalTemplates = []; }
	}

	function applyTemplate(templateId: number) {
		const tpl = journalTemplates.find((t: any) => t.id === templateId);
		if (!tpl) return;
		formData.transaction_type = tpl.transaction_type || 'journal';
		formData.currency = tpl.currency || 'USD';
		formData.reference = tpl.reference_prefix || '';
		formData.description = tpl.description || '';
		lines = (tpl.lines_template || []).map((l: any) => {
			const acct = accounts.find((a: any) => a.code === l.account_code);
			return {
				account_id: acct?.id ?? null,
				debit: parseFloat(l.debit) || null,
				credit: parseFloat(l.credit) || null,
				description: l.description || ''
			};
		});
		if (lines.length < 2) {
			while (lines.length < 2) lines.push({ account_id: null, debit: null, credit: null, description: '' });
		}
		selectedTemplateId = null;
	}

	async function saveAsTemplate() {
		if (!transaction?.id) return;
		const name = prompt('Template name:');
		if (!name) return;
		savingTemplate = true;
		try {
			await createTemplateFromTransaction(transaction.id, name);
			toast.success($i18n.t('Template saved'));
			await loadTemplates();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		savingTemplate = false;
	}

	// Attachments
	let attachmentUrls: string[] = [];
	let uploading = false;
	let attachmentInput: HTMLInputElement;

	async function handleAttachmentUpload() {
		if (!attachmentInput?.files?.length) return;
		uploading = true;
		try {
			for (const file of attachmentInput.files) {
				const result = await uploadAttachment(file);
				attachmentUrls = [...attachmentUrls, result.url];
			}
			attachmentInput.value = '';
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		uploading = false;
	}

	function removeAttachment(index: number) {
		attachmentUrls = attachmentUrls.filter((_, i) => i !== index);
	}

	let lines: Array<{
		account_id: number | null;
		debit: number | null;
		credit: number | null;
		description: string;
	}> = [];

	let saving = false;

	// Account search per line
	let accountSearchIdx: number | null = null; // which line's dropdown is open
	let accountSearchQuery = '';

	const openAccountSearch = (idx: number) => {
		accountSearchIdx = idx;
		const acct = accounts.find((a: any) => a.id === lines[idx]?.account_id);
		accountSearchQuery = acct ? `${acct.code} - ${acct.name}` : '';
	};

	const selectAccount = (idx: number, accountId: number) => {
		lines[idx].account_id = accountId;
		lines = lines;
		accountSearchIdx = null;
		accountSearchQuery = '';
	};

	const clearAccount = (idx: number) => {
		lines[idx].account_id = null;
		lines = lines;
		accountSearchQuery = '';
	};

	const getAccountLabel = (accountId: number | null): string => {
		if (!accountId) return '';
		const acct = accounts.find((a: any) => a.id === accountId);
		return acct ? `${acct.code} - ${acct.name}` : `#${accountId}`;
	};

	$: filteredAccounts = accounts.filter((a: any) => {
		if (!accountSearchQuery) return true;
		const q = accountSearchQuery.toLowerCase();
		return (a.code ?? '').toLowerCase().includes(q) || (a.name ?? '').toLowerCase().includes(q);
	});

	// Inline account creation
	let showCreateAccount = false;
	let createAccountForLineIdx: number | null = null;

	const openCreateAccount = (idx: number) => {
		createAccountForLineIdx = idx;
		accountSearchIdx = null;
		showCreateAccount = true;
	};

	const handleAccountCreated = (e: CustomEvent) => {
		const newAccount = e.detail;
		if (newAccount?.id) {
			// Add to local accounts list and select it
			accounts = [...accounts, newAccount];
			if (createAccountForLineIdx !== null) {
				lines[createAccountForLineIdx].account_id = newAccount.id;
				lines = lines;
			}
		}
		showCreateAccount = false;
		createAccountForLineIdx = null;
	};

	// Reset form when modal opens or transaction changes
	$: if (show) {
		resetForm();
		loadTemplates();
	}

	function resetForm() {
		if (transaction) {
			formData = {
				transaction_date: transaction.transaction_date?.slice(0, 10) ?? '',
				transaction_type: transaction.transaction_type ?? 'journal',
				currency: transaction.currency ?? 'USD',
				reference: transaction.reference ?? '',
				description: transaction.description ?? '',
				invoice_id: transaction.invoice_id ?? null
			};
			invoiceLabel = transaction.invoice_id ? `#${transaction.invoice_id}` : '';
			attachmentUrls = transaction.attachment_urls ?? [];
			lines =
				transaction.lines?.map((l: any) => ({
					account_id: l.account_id ?? null,
					debit: parseFloat(String(l.debit ?? l.debit_amount ?? 0)) || null,
					credit: parseFloat(String(l.credit ?? l.credit_amount ?? 0)) || null,
					description: l.description ?? ''
				})) ?? [];
		} else {
			formData = {
				transaction_date: new Date().toISOString().slice(0, 10),
				transaction_type: 'others',
				currency: 'USD',
				reference: '',
				description: '',
				invoice_id: null
			};
			invoiceLabel = '';
			attachmentUrls = [];
			lines = [
				{ account_id: null, debit: null, credit: null, description: '' },
				{ account_id: null, debit: null, credit: null, description: '' }
			];
		}
	}

	function handleInvoiceSelect(event: CustomEvent) {
		const invoice = event.detail;
		formData.invoice_id = invoice.id;
		invoiceLabel = invoice.invoice_number
			? `${invoice.invoice_number} — ${invoice.vendor_name ?? ''}`
			: `#${invoice.id} — ${invoice.vendor_name ?? ''}`;
		invoiceK4miUrl = invoice.k4mi_document_id ? `${K4MI_BASE_URL}/documents/${invoice.k4mi_document_id}/details` : '';
		// Auto-fill from invoice if creating new
		if (!transaction?.id) {
			if (invoice.invoice_date) formData.transaction_date = invoice.invoice_date.slice(0, 10);
			if (invoice.currency) formData.currency = invoice.currency;
			if (invoice.invoice_number) formData.reference = invoice.invoice_number;
			if (invoice.vendor_name)
				formData.description = `Invoice from ${invoice.vendor_name}`;
			if (invoice.total_amount) {
				const amount = parseFloat(String(invoice.total_amount));
				if (amount > 0 && lines.length >= 2) {
					lines[0].debit = amount;
					lines[0].credit = null;
					lines[1].debit = null;
					lines[1].credit = amount;
					lines = lines;
				}
			}
			// Pre-select AI-suggested or confirmed expense account
			const suggestedCode = invoice.final_account_code || invoice.suggested_account_code;
			if (suggestedCode && accounts.length > 0 && lines.length >= 1) {
				const acct = accounts.find((a: any) => a.code === suggestedCode);
				if (acct) {
					lines[0].account_id = acct.id;
					lines[0].description = `${invoice.vendor_name ?? 'Expense'} — AI: ${acct.code} ${acct.name}`;
					lines = lines;
				}
			}
		}
	}

	function clearInvoice() {
		formData.invoice_id = null;
		invoiceLabel = '';
	}

	// Running balance
	$: totalDebit = lines.reduce((sum, l) => sum + (l.debit ?? 0), 0);
	$: totalCredit = lines.reduce((sum, l) => sum + (l.credit ?? 0), 0);
	$: balanceDiff = Math.round((totalDebit - totalCredit) * 100) / 100;
	$: isBalanced = balanceDiff === 0;
	$: canSubmit = !readOnly && !saving && isBalanced && lines.length >= 2;

	function addLine() {
		lines = [...lines, { account_id: null, debit: null, credit: null, description: '' }];
	}

	function removeLine(index: number) {
		lines = lines.filter((_, i) => i !== index);
	}

	// AI validation
	let aiAvailable = false;
	let validating = false;
	let validationResult: { is_valid: boolean; issues: string[]; warnings: string[]; suggestions: string[] } | null = null;

	// Check AI status on mount
	import { onMount } from 'svelte';
	onMount(() => {
		getAccountingAiStatus()
			.then((s) => (aiAvailable = s.available))
			.catch(() => {});
	});

	async function handleValidate() {
		if (!transaction?.id) {
			toast.info('Save the entry first to validate');
			return;
		}
		validating = true;
		try {
			validationResult = await aiValidateTransaction(transaction.id);
			if (validationResult.is_valid) {
				toast.success('Entry looks good!');
			} else {
				toast.warning(`Found ${validationResult.issues.length} issue(s)`);
			}
		} catch {
			toast.error('AI validation failed');
		} finally {
			validating = false;
		}
	}

	async function handleSubmit() {
		if (!canSubmit) return;
		saving = true;

		const payload: Record<string, any> = {
			...formData,
			attachment_urls: attachmentUrls.length > 0 ? attachmentUrls : undefined,
			lines: lines.map((l) => ({
				account_id: l.account_id,
				debit: l.debit ?? 0,
				credit: l.credit ?? 0,
				description: l.description
			}))
		};

		// Only include invoice_id if set
		if (!formData.invoice_id) {
			delete payload.invoice_id;
		}

		try {
			let result;
			if (transaction?.id) {
				result = await updateTransaction(transaction.id, payload);
				toast.success($i18n.t('Transaction updated'));
			} else {
				result = await createTransaction(payload, companyId);
				const txnId = result?.id ?? result?.transaction?.id;

				// Direct match: post the entry and match it to the BSL
				if (directMatch && bankStatementLineId && txnId) {
					await postTransaction(txnId);
					await matchBankStatement(bankStatementLineId, txnId);
					toast.success($i18n.t('Entry created, posted, and matched'));
				} else {
					toast.success($i18n.t('Transaction created'));
				}
			}
			dispatch('save', result);
			show = false;
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Error') + ': ' + msg);
		} finally {
			saving = false;
		}
	}
</script>

<InvoiceSelector bind:show={showInvoiceSelector} on:select={handleInvoiceSelect} />
<AccountFormModal bind:show={showCreateAccount} {accounts} {companyId} on:save={handleAccountCreated} />

<Modal bind:show size="lg">
	<div class="px-6 py-5">
		<!-- Header -->
		<div class="flex items-center justify-between mb-5">
			<h2 class="text-lg font-medium dark:text-gray-200">
				{#if readOnly}
					{$i18n.t('View Transaction')}
				{:else if transaction?.id}
					{$i18n.t('Edit Transaction')}
				{:else}
					{$i18n.t('New Transaction')}
				{/if}
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

		<!-- Invoice Link -->
		<div class="mb-4 p-3 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-700">
			<div class="flex items-center justify-between">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
						{$i18n.t('Linked Invoice')}
						<span class="text-gray-400 font-normal">({$i18n.t('optional')})</span>
					</label>
					{#if invoiceLabel}
						<div class="flex items-center gap-2">
							<span class="text-sm dark:text-gray-200">{invoiceLabel}</span>
							{#if invoiceK4miUrl}
								<a href={invoiceK4miUrl} target="_blank" rel="noopener" class="text-blue-500 hover:text-blue-700" title={$i18n.t('Open in K4mi')}>
									<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" /></svg>
								</a>
							{/if}
							{#if !readOnly}
								<button
									class="text-xs text-red-500 hover:text-red-700 transition"
									on:click={clearInvoice}
								>
									{$i18n.t('Remove')}
								</button>
							{/if}
						</div>
					{:else}
						<span class="text-sm text-gray-400">{$i18n.t('No invoice linked')}</span>
					{/if}
				</div>
				{#if !readOnly}
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition"
						on:click={() => (showInvoiceSelector = true)}
					>
						{$i18n.t('Browse Invoices')}
					</button>
				{/if}
			</div>
		</div>

		<!-- Template Selector (only when creating new) -->
		{#if !transaction?.id && journalTemplates.length > 0}
			<div class="flex items-center gap-2 mb-3">
				<label class="text-xs font-medium text-gray-500 dark:text-gray-400 whitespace-nowrap">
					{$i18n.t('From Template')}
				</label>
				<select
					class="flex-1 text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-3 py-1.5 outline-hidden focus:border-blue-500 dark:text-gray-200"
					bind:value={selectedTemplateId}
					on:change={() => { if (selectedTemplateId) applyTemplate(selectedTemplateId); }}
				>
					<option value={null}>{$i18n.t('Select a template...')}</option>
					{#each journalTemplates as tpl}
						<option value={tpl.id}>{tpl.name}</option>
					{/each}
				</select>
			</div>
		{/if}

		<!-- Save as Template (only for existing transactions) -->
		{#if transaction?.id && !savingTemplate}
			<div class="flex justify-end mb-2">
				<button
					class="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition"
					on:click={saveAsTemplate}
				>
					{$i18n.t('Save as Template')}
				</button>
			</div>
		{/if}

		<!-- Form Fields -->
		<div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
			<!-- Date -->
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Date')}
				</label>
				<input
					type="date"
					class="w-full text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-3 py-2 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
					bind:value={formData.transaction_date}
					disabled={readOnly}
				/>
			</div>

			<!-- Type -->
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Type')}
				</label>
				<select
					class="w-full text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-3 py-2 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
					bind:value={formData.transaction_type}
					disabled={readOnly}
				>
					<option value="invoice">{$i18n.t('Invoice')}</option>
					<option value="bill">{$i18n.t('Bill')}</option>
					<option value="payment_in">{$i18n.t('Payment In')}</option>
					<option value="payment_out">{$i18n.t('Payment Out')}</option>
					<option value="others">{$i18n.t('Others')}</option>
					<option value="adjustment">{$i18n.t('Adjustment')}</option>
				</select>
			</div>

			<!-- Currency -->
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Currency')}
				</label>
				<input
					type="text"
					class="w-full text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-3 py-2 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
					bind:value={formData.currency}
					maxlength="3"
					placeholder="USD"
					disabled={readOnly}
				/>
			</div>

			<!-- Reference -->
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Reference')}
				</label>
				<input
					type="text"
					class="w-full text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-3 py-2 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
					bind:value={formData.reference}
					placeholder={$i18n.t('Reference number')}
					disabled={readOnly}
				/>
			</div>

			<!-- Description -->
			<div class="md:col-span-2">
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
					{$i18n.t('Description')}
				</label>
				<textarea
					class="w-full text-sm rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-3 py-2 outline-hidden focus:border-blue-500 dark:text-gray-200 resize-none disabled:opacity-60"
					rows="2"
					bind:value={formData.description}
					placeholder={$i18n.t('Transaction description')}
					disabled={readOnly}
				></textarea>
			</div>
		</div>

		<!-- Lines Section -->
		<div class="mb-4">
			<div class="flex items-center justify-between mb-2">
				<h3 class="text-sm font-medium dark:text-gray-300">{$i18n.t('Journal Lines')}</h3>
				{#if !readOnly}
					<button
						class="text-xs px-3 py-1 rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition font-medium"
						on:click={addLine}
					>
						+ {$i18n.t('Add Line')}
					</button>
				{/if}
			</div>

			<div class="overflow-x-auto">
				<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
					<thead
						class="text-xs text-gray-500 dark:text-gray-400 uppercase bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700"
					>
						<tr>
							<th class="px-3 py-2 w-2/5">{$i18n.t('Account')}</th>
							<th class="px-3 py-2 text-right w-1/6">{$i18n.t('Debit')}</th>
							<th class="px-3 py-2 text-right w-1/6">{$i18n.t('Credit')}</th>
							<th class="px-3 py-2 w-1/4">{$i18n.t('Description')}</th>
							{#if !readOnly}
								<th class="px-3 py-2 w-10"></th>
							{/if}
						</tr>
					</thead>
					<tbody>
						{#each lines as line, idx}
							<tr class="border-b border-gray-100 dark:border-gray-800">
								<!-- Account searchable select -->
								<td class="px-2 py-1.5 relative">
									{#if accountSearchIdx === idx && !readOnly}
										<!-- Search input + dropdown -->
										<input
											type="text"
											class="w-full text-xs rounded-lg border border-blue-400 dark:border-blue-600 bg-white dark:bg-gray-850 px-2 py-1.5 outline-hidden dark:text-gray-200"
											placeholder={$i18n.t('Search accounts...')}
											bind:value={accountSearchQuery}
											on:blur={() => { setTimeout(() => { accountSearchIdx = null; }, 200); }}
											autofocus
										/>
										<div class="absolute z-50 left-2 right-2 mt-0.5 max-h-40 overflow-y-auto bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg">
											{#each filteredAccounts.slice(0, 30) as account}
												<button
													type="button"
													class="w-full text-left px-2.5 py-1.5 text-xs hover:bg-blue-50 dark:hover:bg-blue-900/20 transition border-b border-gray-50 dark:border-gray-800 last:border-b-0"
													on:mousedown|preventDefault={() => selectAccount(idx, account.id)}
												>
													<span class="font-mono font-medium">{account.code}</span>
													<span class="text-gray-500 dark:text-gray-400 ml-1">{account.name}</span>
												</button>
											{/each}
											{#if filteredAccounts.length === 0}
												<div class="px-2.5 py-2 text-xs text-gray-400 italic">{$i18n.t('No accounts found')}</div>
											{/if}
											<button
												type="button"
												class="w-full text-left px-2.5 py-1.5 text-xs font-medium text-blue-600 dark:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 transition border-t border-gray-200 dark:border-gray-700 flex items-center gap-1"
												on:mousedown|preventDefault={() => openCreateAccount(idx)}
											>
												<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" /></svg>
												{$i18n.t('Create new account')}
											</button>
										</div>
									{:else}
										<!-- Display selected account as clickable button -->
										<button
											type="button"
											class="w-full text-left text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-2 py-1.5 outline-hidden dark:text-gray-200 disabled:opacity-60 hover:border-blue-400 dark:hover:border-blue-600 transition truncate"
											disabled={readOnly}
											on:click={() => openAccountSearch(idx)}
										>
											{#if line.account_id}
												{getAccountLabel(line.account_id)}
											{:else}
												<span class="text-gray-400">-- {$i18n.t('Select account')} --</span>
											{/if}
										</button>
									{/if}
								</td>

								<!-- Debit -->
								<td class="px-2 py-1.5">
									<input
										type="number"
										step="0.01"
										min="0"
										class="w-full text-xs text-right rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-2 py-1.5 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
										bind:value={line.debit}
										placeholder="0.00"
										disabled={readOnly}
									/>
								</td>

								<!-- Credit -->
								<td class="px-2 py-1.5">
									<input
										type="number"
										step="0.01"
										min="0"
										class="w-full text-xs text-right rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-2 py-1.5 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
										bind:value={line.credit}
										placeholder="0.00"
										disabled={readOnly}
									/>
								</td>

								<!-- Line Description -->
								<td class="px-2 py-1.5">
									<input
										type="text"
										class="w-full text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-2 py-1.5 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
										bind:value={line.description}
										placeholder={$i18n.t('Line memo')}
										disabled={readOnly}
									/>
								</td>

								<!-- Remove -->
								{#if !readOnly}
									<td class="px-2 py-1.5 text-center">
										<button
											class="p-1 rounded-lg hover:bg-red-50 dark:hover:bg-red-900/20 text-red-500 transition disabled:opacity-30"
											disabled={lines.length <= 2}
											on:click={() => removeLine(idx)}
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
													d="M6 18 18 6M6 6l12 12"
												/>
											</svg>
										</button>
									</td>
								{/if}
							</tr>
						{/each}
					</tbody>

					<!-- Totals row -->
					<tfoot>
						<tr class="bg-gray-50 dark:bg-gray-800 font-medium text-xs">
							<td class="px-3 py-2 text-right">{$i18n.t('Totals')}</td>
							<td class="px-3 py-2 text-right">
								{totalDebit.toLocaleString(undefined, {
									minimumFractionDigits: 2,
									maximumFractionDigits: 2
								})}
							</td>
							<td class="px-3 py-2 text-right">
								{totalCredit.toLocaleString(undefined, {
									minimumFractionDigits: 2,
									maximumFractionDigits: 2
								})}
							</td>
							<td class="px-3 py-2" colspan={readOnly ? 1 : 2}></td>
						</tr>
					</tfoot>
				</table>
			</div>

			<!-- Balance indicator -->
			<div class="mt-3 flex items-center justify-end gap-2 text-sm">
				{#if isBalanced}
					<span class="text-green-600 dark:text-green-400 font-medium">
						{$i18n.t('Balanced')}
					</span>
				{:else}
					<span class="text-red-600 dark:text-red-400 font-medium">
						{$i18n.t('Unbalanced')} ({balanceDiff > 0 ? '+' : ''}{balanceDiff.toLocaleString(
							undefined,
							{ minimumFractionDigits: 2, maximumFractionDigits: 2 }
						)})
					</span>
				{/if}
			</div>
		</div>

		<!-- Attachments -->
		<div class="mt-3">
			<div class="flex items-center justify-between mb-1.5">
				<label class="text-xs font-medium text-gray-500 dark:text-gray-400">{$i18n.t('Attachments')}</label>
				{#if !readOnly}
					<input
						bind:this={attachmentInput}
						type="file"
						multiple
						class="hidden"
						on:change={handleAttachmentUpload}
					/>
					<button
						class="text-xs text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 transition flex items-center gap-1"
						on:click={() => attachmentInput?.click()}
						disabled={uploading}
					>
						{#if uploading}
							{$i18n.t('Uploading...')}
						{:else}
							<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M18.375 12.739l-7.693 7.693a4.5 4.5 0 01-6.364-6.364l10.94-10.94A3 3 0 1119.5 7.372L8.552 18.32m.009-.01l-.01.01m5.699-9.941l-7.81 7.81a1.5 1.5 0 002.112 2.13" /></svg>
							{$i18n.t('Attach File')}
						{/if}
					</button>
				{/if}
			</div>
			{#if attachmentUrls.length > 0}
				<div class="space-y-1">
					{#each attachmentUrls as url, i}
						<div class="flex items-center gap-2 text-xs bg-gray-50 dark:bg-gray-850 rounded-lg px-3 py-1.5">
							<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5 text-gray-400"><path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" /></svg>
							<a
								href="{INVOICE_API_BASE_URL}{url}"
								target="_blank"
								rel="noopener"
								class="text-blue-600 dark:text-blue-400 hover:underline truncate flex-1"
							>
								{url.split('/').pop()}
							</a>
							{#if !readOnly}
								<button class="text-red-500 hover:text-red-700 transition" on:click={() => removeAttachment(i)} title={$i18n.t('Remove')}>
									<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" /></svg>
								</button>
							{/if}
						</div>
					{/each}
				</div>
			{:else}
				<div class="text-[10px] text-gray-400 dark:text-gray-500 italic">{$i18n.t('No attachments')}</div>
			{/if}
		</div>

		<!-- Direct Match option (only when creating from Bank tab) -->
		{#if bankStatementLineId && !transaction?.id}
			<label class="flex items-center gap-2 pt-3 text-xs text-gray-600 dark:text-gray-400 cursor-pointer border-t border-gray-100 dark:border-gray-800">
				<input type="checkbox" bind:checked={directMatch} class="rounded" />
				{$i18n.t('Direct match — post entry and match to bank statement line')}
			</label>
		{/if}

		<!-- Footer -->
		<div class="flex items-center justify-end gap-2 pt-3" class:border-t={!bankStatementLineId || transaction?.id} class:border-gray-100={!bankStatementLineId} class:dark:border-gray-800={!bankStatementLineId}>
			<button
				class="text-sm px-4 py-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium transition"
				on:click={() => (show = false)}
			>
				{readOnly ? $i18n.t('Close') : $i18n.t('Cancel')}
			</button>

			{#if !readOnly}
				{#if aiAvailable && transaction?.id}
					<button
						class="text-sm px-3 py-2 rounded-xl border border-purple-300 dark:border-purple-700 text-purple-700 dark:text-purple-400 hover:bg-purple-50 dark:hover:bg-purple-900/20 transition disabled:opacity-50"
						disabled={validating}
						on:click={handleValidate}
					>
						{validating ? $i18n.t('Validating...') : $i18n.t('AI Validate')}
					</button>
				{/if}
				<button
					class="text-sm px-4 py-2 rounded-xl bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium transition disabled:opacity-50 disabled:cursor-not-allowed"
					disabled={!canSubmit}
					on:click={handleSubmit}
				>
					{#if saving}
						{$i18n.t('Saving...')}
					{:else if transaction?.id}
						{$i18n.t('Update')}
					{:else}
						{$i18n.t('Create')}
					{/if}
				</button>
			{/if}
		</div>

		<!-- AI Validation Results -->
		{#if validationResult}
			<div class="px-5 pb-4 -mt-2">
				<div class="text-xs rounded-lg p-3 space-y-1.5 {validationResult.is_valid ? 'bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800' : 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800'}">
					<div class="font-medium {validationResult.is_valid ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}">
						{validationResult.is_valid ? $i18n.t('Entry is valid') : $i18n.t('Issues found')}
					</div>
					{#each validationResult.issues as issue}
						<div class="text-red-600 dark:text-red-400">- {issue}</div>
					{/each}
					{#each validationResult.warnings as warning}
						<div class="text-yellow-600 dark:text-yellow-400">- {warning}</div>
					{/each}
					{#each validationResult.suggestions as suggestion}
						<div class="text-blue-600 dark:text-blue-400">- {suggestion}</div>
					{/each}
				</div>
			</div>
		{/if}
	</div>
</Modal>
