<script lang="ts">
	import { getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import { createTransaction, updateTransaction } from '$lib/apis/accounting';
	import { K4MI_BASE_URL } from '$lib/constants';
	import InvoiceSelector from '$lib/components/accounting/InvoiceSelector.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let transaction: any = null;
	export let accounts: any[] = [];
	export let companyId: number;

	// Determine read-only mode for posted/voided transactions
	$: readOnly = transaction && (transaction.status === 'posted' || transaction.status === 'voided');

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
		transaction_type: 'journal',
		currency: 'USD',
		reference: '',
		description: '',
		invoice_id: null
	};

	let invoiceLabel = '';
	let invoiceK4miUrl = '';
	let showInvoiceSelector = false;

	let lines: Array<{
		account_id: number | null;
		debit: number | null;
		credit: number | null;
		description: string;
	}> = [];

	let saving = false;

	// Reset form when modal opens or transaction changes
	$: if (show) {
		resetForm();
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
				transaction_type: 'journal',
				currency: 'USD',
				reference: '',
				description: '',
				invoice_id: null
			};
			invoiceLabel = '';
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

	async function handleSubmit() {
		if (!canSubmit) return;
		saving = true;

		const payload: Record<string, any> = {
			...formData,
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
				toast.success($i18n.t('Transaction created'));
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
					<option value="journal">{$i18n.t('Journal')}</option>
					<option value="invoice">{$i18n.t('Invoice')}</option>
					<option value="bill">{$i18n.t('Bill')}</option>
					<option value="payment">{$i18n.t('Payment')}</option>
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
								<!-- Account select -->
								<td class="px-2 py-1.5">
									<select
										class="w-full text-xs rounded-lg border border-gray-200 dark:border-gray-700 bg-transparent dark:bg-gray-850 px-2 py-1.5 outline-hidden focus:border-blue-500 dark:text-gray-200 disabled:opacity-60"
										bind:value={line.account_id}
										disabled={readOnly}
									>
										<option value={null}>-- {$i18n.t('Select account')} --</option>
										{#each accounts as account}
											<option value={account.id}>
												{account.code} - {account.name}
											</option>
										{/each}
									</select>
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

		<!-- Footer -->
		<div class="flex items-center justify-end gap-2 pt-3 border-t border-gray-100 dark:border-gray-800">
			<button
				class="text-sm px-4 py-2 rounded-xl bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium transition"
				on:click={() => (show = false)}
			>
				{readOnly ? $i18n.t('Close') : $i18n.t('Cancel')}
			</button>

			{#if !readOnly}
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
	</div>
</Modal>
