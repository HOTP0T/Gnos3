<script lang="ts">
	import { getContext, createEventDispatcher, onMount, onDestroy, tick } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';

	import { createPayment, getPaymentPreview } from '$lib/apis/accounting';
	import K4miDocLink from '$lib/components/common/K4miDocLink.svelte';
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

	// What the payment will book — resolved by the backend from the linked
	// invoice, the bank account's defaults, bank-line rules or the company
	// default; never guessed. The human fills whatever is still open.
	let plan: any = null;
	let planLoading = false;
	let planTimer: any = null;
	let editingSide: string | null = null;
	let sideSearch = '';

	$: parentIds = new Set(accounts.map((a: any) => a.parent_id).filter(Boolean));
	$: leafAccounts = accounts.filter((a: any) => !parentIds.has(a.id));
	$: counterpartId = direction === 'outbound' ? debit_account_id : credit_account_id;
	$: bankId = direction === 'outbound' ? credit_account_id : debit_account_id;

	const refreshPlan = async () => {
		if (!show || !companyId) return;
		planLoading = true;
		try {
			plan = await getPaymentPreview({
				company_id: companyId,
				direction,
				amount: amount ?? 0,
				invoice_id: invoice_id ?? undefined,
				bank_statement_line_id: prefill?._bank_statement_line_id ?? undefined,
				payee: payee || undefined,
				payer: payer || undefined,
				reference: reference || undefined,
				debit_account_id: debit_account_id ?? undefined,
				credit_account_id: credit_account_id ?? undefined,
				currency: currency || undefined,
				payment_date: payment_date || undefined
			});
		} catch {
			plan = null;
		}
		planLoading = false;
	};
	const schedulePlan = () => {
		clearTimeout(planTimer);
		planTimer = setTimeout(refreshPlan, 250);
	};
	$: if (show && mounted) {
		// re-resolve whenever an input that can change the accounts changes
		void [direction, invoice_id, payee, payer, reference, debit_account_id, credit_account_id, amount, currency, payment_date];
		schedulePlan();
	}
	const chooseSide = (role: string, id: number) => {
		const isCounterpart = role === 'counterpart';
		if ((direction === 'outbound') === isCounterpart) debit_account_id = id;
		else credit_account_id = id;
		editingSide = null;
		sideSearch = '';
	};
	const sideCandidates = (line: any) => {
		const q = sideSearch.toLowerCase();
		const cands = (line.candidates ?? []).map((c: any) => c.id);
		const typeOk = (a: any) => (line.role === 'bank' ? a.account_type === 'asset' : ['liability', 'asset'].includes(a.account_type));
		return leafAccounts
			.filter((a: any) => typeOk(a) && (!q || a.code.includes(q) || a.name.toLowerCase().includes(q)))
			.sort((a: any, b: any) => (cands.includes(b.id) ? 1 : 0) - (cands.includes(a.id) ? 1 : 0) || a.code.localeCompare(b.code))
			.slice(0, 60);
	};
	const SOURCE_LABEL: Record<string, string> = {
		human: 'chosen', invoice: "from the invoice's entry", matched: 'from the matched entry', bank_default: 'bank account default', rule: 'from rule', company: 'company default', missing: 'account required'
	};
	$: fxRate = plan?.exchange_rate ?? null;
	$: fxForeign = !!fxRate && fxRate.currency !== fxRate.base;
	$: planReady = !!plan && plan.complete;
	let showInvoiceSelector = false;
	let invoiceLabel = '';
	let invoiceK4miDocId: number | null = null;

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
		invoiceK4miDocId = inv.k4mi_document_id ?? null;
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
								{#if invoiceK4miDocId}
									<K4miDocLink docId={invoiceK4miDocId} extraClass="text-blue-500 hover:text-blue-700 flex-shrink-0" title={$i18n.t('Open in K4mi')}>
										<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3.5 h-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" /></svg>
									</K4miDocLink>
								{/if}
								<button
									class="text-xs text-red-500 hover:text-red-700 transition whitespace-nowrap"
									on:click={() => { invoice_id = null; invoiceLabel = ''; invoiceK4miDocId = null; }}
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

					<!-- What this payment will book -->
					<div class="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden {planLoading ? 'opacity-60' : ''}">
						<div class="px-3 py-1.5 bg-gray-100 dark:bg-gray-800 text-[10px] uppercase text-gray-500 dark:text-gray-400 flex justify-between">
							<span>{$i18n.t('Journal entry')}</span>
							{#if plan && !plan.complete}<span class="text-red-600 dark:text-red-300 normal-case">{$i18n.t('account required')}</span>{/if}
						</div>
						{#if plan}
							<table class="w-full text-xs dark:text-gray-200">
								<tbody>
									{#each plan.lines as line}
										<tr class="border-t border-gray-100 dark:border-gray-800 align-top">
											<td class="px-3 py-1.5 w-40">
												<div class="font-medium">{$i18n.t(line.label)}</div>
												<div class="text-[10px] text-gray-400">{line.side === 'debit' ? $i18n.t('Debit') : $i18n.t('Credit')} {plan.amount?.toFixed ? plan.amount.toFixed(2) : plan.amount}</div>
											</td>
											<td class="px-3 py-1.5">
												{#if editingSide === line.role}
													<input
														type="text"
														placeholder={$i18n.t('Search accounts...')}
														class="w-full text-xs rounded-lg px-2 py-1 border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 outline-hidden mb-1"
														on:input={(e) => { sideSearch = e.currentTarget.value; }}
													/>
													<div class="max-h-28 overflow-y-auto border border-gray-200 dark:border-gray-700 rounded-lg">
														{#each sideCandidates(line) as a}
															<button type="button" class="w-full text-left px-2 py-1 hover:bg-blue-50 dark:hover:bg-blue-900/20 border-b border-gray-100 dark:border-gray-800 last:border-b-0" on:click={() => chooseSide(line.role, a.id)}>
																<span class="font-mono font-medium">{a.code}</span> <span class="text-gray-500">{a.name}</span>
																{#if (line.candidates ?? []).some((c: any) => c.id === a.id)}<span class="ml-1 text-[9px] text-amber-600">{$i18n.t('suggested')}</span>{/if}
															</button>
														{/each}
													</div>
												{:else}
													<div class="flex items-center gap-2 flex-wrap">
														{#if line.account_id}
															<span class="font-mono">{line.account_code} <span class="font-sans text-gray-500">{line.account_name}</span></span>
														{/if}
														<span class="px-1.5 py-0.5 rounded-full text-[9px] font-medium {line.source === 'missing' ? 'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300'}">{$i18n.t(SOURCE_LABEL[line.source] ?? line.source)}</span>
														<button type="button" class="text-[10px] text-blue-600 dark:text-blue-400 hover:underline" on:click={() => { editingSide = line.role; sideSearch = ''; }}>
															{line.account_id ? $i18n.t('Change') : $i18n.t('Choose')}
														</button>
													</div>
													{#if line.note}<div class="text-[10px] text-amber-700 dark:text-amber-300 mt-0.5">{line.note}</div>{/if}
												{/if}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
							{#if fxForeign}
								<div class="px-3 py-1.5 border-t border-gray-100 dark:border-gray-800 text-[10px] {fxRate.rate ? 'text-gray-500 dark:text-gray-400' : 'text-red-700 dark:text-red-300'}">
									{fxRate.currency} → {fxRate.base}:
									{fxRate.rate
										? `${fxRate.rate} (${$i18n.t('rate on file')}, ${fxRate.date})`
										: $i18n.t('no rate on file for {{date}} — the entry stays a draft until one is added in Exchange Rates', { date: fxRate.date })}
								</div>
							{/if}
							{#if plan.fx}
								<div class="px-3 py-1.5 border-t border-gray-100 dark:border-gray-800 text-[10px] {plan.fx.account_id ? 'text-gray-600 dark:text-gray-300' : 'text-amber-700 dark:text-amber-300'}">
									{$i18n.t(plan.fx.label)}: {plan.fx.amount_base?.toFixed ? plan.fx.amount_base.toFixed(2) : plan.fx.amount_base} {fxRate?.base ?? ''}
									({$i18n.t('booked at')} {plan.fx.settled_rate}, {$i18n.t('paid at')} {fxRate?.rate})
									→ {plan.fx.account_id ? `${plan.fx.account_code} ${plan.fx.account_name}` : $i18n.t('exchange gain / loss account not set — booked as a draft with a gap')}
								</div>
							{/if}
						{:else}
							<div class="px-3 py-2 text-[11px] text-gray-400">{$i18n.t('Fill in the payment to see the entry.')}</div>
						{/if}
					</div>
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
						disabled={saving || !planReady}
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
