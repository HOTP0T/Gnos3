<script lang="ts">
	import { getContext, createEventDispatcher, onMount, onDestroy, tick } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';

	import { createPayment } from '$lib/apis/accounting';
	import { K4MI_BASE_URL } from '$lib/constants';
	import InvoiceSelector from '$lib/components/accounting/InvoiceSelector.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let accounts: any[] = [];
	export let companyId: number;
	export let prefill: any = null;

	// Form state
	let payment_date = '';
	let amount: number | null = null;
	let currency = 'USD';
	let direction = 'outbound';
	let method = 'bank_transfer';
	let payer = '';
	let payee = '';
	let reference = '';
	let notes = '';
	let invoice_id: number | null = null;
	let debit_account_id: number | null = null;
	let credit_account_id: number | null = null;

	let showAdvanced = false;
	let saving = false;
	let showInvoiceSelector = false;
	let invoiceLabel = '';
	let invoiceK4miUrl = '';

	// Custom payment methods (persisted in localStorage)
	const CUSTOM_METHODS_KEY = 'accounting-custom-payment-methods';
	let customMethods: string[] = [];
	let customMethodInput = '';

	const builtinMethods = ['cash', 'bank_transfer', 'check', 'credit_card'];

	function loadCustomMethods() {
		try {
			const saved = localStorage.getItem(CUSTOM_METHODS_KEY);
			if (saved) customMethods = JSON.parse(saved);
		} catch {}
	}

	function saveCustomMethod(name: string) {
		if (!name || builtinMethods.includes(name) || customMethods.includes(name)) return;
		customMethods = [...customMethods, name];
		try {
			localStorage.setItem(CUSTOM_METHODS_KEY, JSON.stringify(customMethods));
		} catch {}
	}

	loadCustomMethods();

	let modalElement: HTMLDivElement | null = null;
	let mounted = false;

	const resetForm = () => {
		const today = new Date();
		payment_date = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
		amount = null;
		currency = 'USD';
		direction = 'outbound';
		method = 'bank_transfer';
		payer = '';
		payee = '';
		reference = '';
		notes = '';
		invoice_id = null;
		debit_account_id = null;
		credit_account_id = null;
		showAdvanced = false;
		invoiceLabel = '';
	};

	$: if (show) {
		resetForm();
		if (prefill) {
			payment_date = prefill.payment_date || payment_date;
			amount = prefill.amount || null;
			currency = prefill.currency || 'USD';
			direction = prefill.direction || 'outbound';
			method = prefill.method || 'bank_transfer';
			payer = prefill.payer || '';
			payee = prefill.payee || '';
			reference = prefill.reference || '';
			if (prefill.invoice_id) {
				invoice_id = prefill.invoice_id;
				invoiceLabel = prefill.reference || `#${prefill.invoice_id}`;
			}
		}
	}

	const handleKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			show = false;
		}
	};

	const handleSave = async () => {
		if (!payment_date || !amount || amount <= 0) {
			toast.error($i18n.t('Please fill in required fields (date and amount)'));
			return;
		}

		saving = true;
		try {
			// Resolve custom method
			let finalMethod = method;
			if (method === '__other__') {
				if (!customMethodInput.trim()) {
					toast.error($i18n.t('Please enter a custom payment method'));
					saving = false;
					return;
				}
				finalMethod = customMethodInput.trim().toLowerCase().replace(/\s+/g, '_');
				saveCustomMethod(finalMethod);
				customMethodInput = '';
			}

			const data: Record<string, any> = {
				payment_date,
				amount,
				currency,
				direction,
				method: finalMethod,
				payer: payer || null,
				payee: payee || null,
				reference: reference || null,
				notes: notes || null
			};

			if (invoice_id) {
				data.invoice_id = invoice_id;
			}
			if (debit_account_id) {
				data.debit_account_id = debit_account_id;
			}
			if (credit_account_id) {
				data.credit_account_id = credit_account_id;
			}

			await createPayment(data, companyId, prefill?._bank_statement_line_id);
			toast.success($i18n.t('Payment recorded'));
			show = false;
			await tick();
			dispatch('save');
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to record payment')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	onMount(() => {
		mounted = true;
	});

	$: if (mounted) {
		if (show && modalElement) {
			document.body.appendChild(modalElement);
			window.addEventListener('keydown', handleKeyDown);
			document.body.style.overflow = 'hidden';
		} else if (modalElement) {
			window.removeEventListener('keydown', handleKeyDown);
			try {
				document.body.removeChild(modalElement);
			} catch {
				// already removed
			}
			document.body.style.overflow = 'unset';
		}
	}

	onDestroy(() => {
		show = false;
		window.removeEventListener('keydown', handleKeyDown);
		if (modalElement) {
			try {
				document.body.removeChild(modalElement);
			} catch {
				// already removed
			}
		}
	});
</script>

<InvoiceSelector
	bind:show={showInvoiceSelector}
	on:select={(e) => {
		const inv = e.detail;
		invoice_id = inv.id;
		invoiceLabel = inv.invoice_number
			? `${inv.invoice_number} — ${inv.vendor_name ?? ''}`
			: `#${inv.id} — ${inv.vendor_name ?? ''}`;
		invoiceK4miUrl = inv.k4mi_document_id ? `${K4MI_BASE_URL}/documents/${inv.k4mi_document_id}/details` : '';
		if (inv.vendor_name && direction === 'outbound' && !payee) payee = inv.vendor_name;
		if (inv.vendor_name && direction === 'inbound' && !payer) payer = inv.vendor_name;
		if (inv.total_amount && !amount) amount = parseFloat(String(inv.total_amount));
		if (inv.currency) currency = inv.currency;
	}}
/>

{#if show}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		bind:this={modalElement}
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-[50000] overflow-hidden overscroll-contain"
		in:fade={{ duration: 10 }}
		on:mousedown={() => {
			show = false;
		}}
	>
		<div
			class="m-auto max-w-full w-[36rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[90dvh] shadow-3xl border border-white dark:border-gray-900 overflow-y-auto"
			in:flyAndScale
			on:mousedown={(e) => {
				e.stopPropagation();
			}}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{$i18n.t('Record Payment')}
				</div>

				<div class="space-y-3">
					<!-- Row: Date + Amount + Currency -->
					<div class="grid grid-cols-3 gap-3">
						<div>
							<label
								for="payment-date"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Date')} *
							</label>
							<input
								id="payment-date"
								type="date"
								bind:value={payment_date}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="payment-amount"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Amount')} *
							</label>
							<input
								id="payment-amount"
								type="number"
								step="0.01"
								min="0"
								bind:value={amount}
								placeholder="0.00"
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="payment-currency"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Currency')}
							</label>
							<input
								id="payment-currency"
								type="text"
								maxlength="3"
								bind:value={currency}
								placeholder="USD"
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition uppercase"
							/>
						</div>
					</div>

					<!-- Row: Direction + Method -->
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label
								for="payment-direction"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Direction')}
							</label>
							<select
								id="payment-direction"
								bind:value={direction}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							>
								<option value="inbound">{$i18n.t('Inbound')}</option>
								<option value="outbound">{$i18n.t('Outbound')}</option>
							</select>
						</div>
						<div>
							<label
								for="payment-method"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Method')}
							</label>
							<select
								id="payment-method"
								bind:value={method}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							>
								<option value="cash">{$i18n.t('Cash')}</option>
								<option value="bank_transfer">{$i18n.t('Bank Transfer')}</option>
								<option value="check">{$i18n.t('Check')}</option>
								<option value="credit_card">{$i18n.t('Credit Card')}</option>
								{#each customMethods as cm}
									<option value={cm}>{cm.replace(/_/g, ' ')}</option>
								{/each}
								<option value="__other__">{$i18n.t('Other (custom)...')}</option>
							</select>
							{#if method === '__other__'}
								<input
									type="text"
									bind:value={customMethodInput}
									placeholder={$i18n.t('e.g. Mobile Payment, Wire Transfer...')}
									class="w-full mt-1.5 text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-blue-300 dark:border-blue-700 outline-hidden focus:border-blue-500 transition"
								/>
							{/if}
						</div>
					</div>

					<!-- Row: Payer + Payee -->
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label
								for="payment-payer"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Payer')}
							</label>
							<input
								id="payment-payer"
								type="text"
								bind:value={payer}
								placeholder={$i18n.t('Payer name')}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="payment-payee"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Payee')}
							</label>
							<input
								id="payment-payee"
								type="text"
								bind:value={payee}
								placeholder={$i18n.t('Payee name')}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
					</div>

					<!-- Reference -->
					<div>
						<label
							for="payment-reference"
							class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>
							{$i18n.t('Reference')}
						</label>
						<input
							id="payment-reference"
							type="text"
							bind:value={reference}
							placeholder={$i18n.t('Check #, transaction ID, etc.')}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						/>
					</div>

					<!-- Invoice Link -->
					<div>
						<label
							class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>
							{$i18n.t('Invoice')}
							<span class="text-gray-400 font-normal">({$i18n.t('optional')})</span>
						</label>
						<div class="flex items-center gap-2">
							{#if invoiceLabel}
								<span class="text-sm dark:text-gray-200 flex-1 truncate">{invoiceLabel}</span>
								{#if invoiceK4miUrl}
									<a href={invoiceK4miUrl} target="_blank" rel="noopener" class="text-blue-500 hover:text-blue-700 flex-shrink-0" title={$i18n.t('Open in K4mi')}>
										<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" /></svg>
									</a>
								{/if}
								<button
									class="text-xs text-red-500 hover:text-red-700 transition whitespace-nowrap"
									on:click={() => { invoice_id = null; invoiceLabel = ''; invoiceK4miUrl = ''; }}
									type="button"
								>
									{$i18n.t('Remove')}
								</button>
							{:else}
								<span class="text-sm text-gray-400 flex-1">{$i18n.t('No invoice linked')}</span>
							{/if}
							<button
								class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/30 dark:text-blue-300 dark:hover:bg-blue-900/50 transition whitespace-nowrap"
								on:click={() => (showInvoiceSelector = true)}
								type="button"
							>
								{$i18n.t('Browse')}
							</button>
						</div>
					</div>

					<!-- Notes -->
					<div>
						<label
							for="payment-notes"
							class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>
							{$i18n.t('Notes')}
						</label>
						<textarea
							id="payment-notes"
							bind:value={notes}
							placeholder={$i18n.t('Additional notes...')}
							rows="2"
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition resize-none"
						/>
					</div>

					<!-- Advanced (collapsible) -->
					{#if accounts.length > 0}
						<div>
							<button
								type="button"
								class="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition"
								on:click={() => {
									showAdvanced = !showAdvanced;
								}}
							>
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									stroke-width="2"
									stroke="currentColor"
									class="size-3.5 transition-transform {showAdvanced ? 'rotate-180' : ''}"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										d="m19.5 8.25-7.5 7.5-7.5-7.5"
									/>
								</svg>
								{$i18n.t('Advanced')}
							</button>

							{#if showAdvanced}
								<div class="mt-2 grid grid-cols-2 gap-3">
									<div>
										<label
											for="payment-debit-account"
											class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
										>
											{$i18n.t('Debit Account')}
										</label>
										<select
											id="payment-debit-account"
											bind:value={debit_account_id}
											class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
										>
											<option value={null}>{$i18n.t('Default')}</option>
											{#each accounts as account}
												<option value={account.id}>
													{account.code ? `${account.code} - ` : ''}{account.name}
												</option>
											{/each}
										</select>
									</div>
									<div>
										<label
											for="payment-credit-account"
											class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
										>
											{$i18n.t('Credit Account')}
										</label>
										<select
											id="payment-credit-account"
											bind:value={credit_account_id}
											class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
										>
											<option value={null}>{$i18n.t('Default')}</option>
											{#each accounts as account}
												<option value={account.id}>
													{account.code ? `${account.code} - ` : ''}{account.name}
												</option>
											{/each}
										</select>
									</div>
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<!-- Actions -->
				<div class="mt-6 flex justify-between gap-1.5">
					<button
						class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
						on:click={() => {
							show = false;
						}}
						type="button"
						disabled={saving}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
						on:click={handleSave}
						type="button"
						disabled={saving}
					>
						{#if saving}
							{$i18n.t('Saving...')}
						{:else}
							{$i18n.t('Record Payment')}
						{/if}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}
