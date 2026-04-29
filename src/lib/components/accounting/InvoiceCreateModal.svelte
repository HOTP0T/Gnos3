<script lang="ts">
	import { getContext, createEventDispatcher, onMount, onDestroy, tick } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';
	import type { Writable } from 'svelte/store';

	import { createManualInvoice, getNextInvoiceNumber } from '$lib/apis/accounting';

	const i18n = getContext('i18n');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let companyId: number;
	export let companyName: string = '';

	// Form state
	let invoice_number = '';
	let nextNumberHint = '';
	let client_name = '';
	let vendor_name = '';
	let invoice_date = '';
	let due_date = '';
	let currency = 'EUR';
	let subtotal: number | null = null;
	let tax_amount: number | null = null;
	let tax_inclusive = false;
	let total_amount: number | null = null;
	let payment_terms = '';
	let po_number = '';
	let description = '';
	let business_unit = '';

	// Line items
	let lineItems: { description: string; quantity: number; unit_price: number; amount: number }[] = [];
	let showLineItems = false;
	let showAdvanced = false;
	let saving = false;

	let modalElement: HTMLDivElement | null = null;
	let mounted = false;

	const resetForm = () => {
		const today = new Date();
		invoice_date = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
		invoice_number = '';
		nextNumberHint = '';
		client_name = '';
		vendor_name = companyName;
		due_date = '';
		currency = $companyCurrencyCtx || 'EUR';
		subtotal = null;
		tax_amount = null;
		tax_inclusive = false;
		total_amount = null;
		payment_terms = '';
		po_number = '';
		description = '';
		business_unit = '';
		lineItems = [];
		showLineItems = false;
		showAdvanced = false;
	};

	async function fetchNextNumber() {
		try {
			const res = await getNextInvoiceNumber(companyId);
			nextNumberHint = res.next_number;
		} catch {
			nextNumberHint = '';
		}
	}

	$: if (show) {
		resetForm();
		fetchNextNumber();
	}

	// Auto-compute amounts from line items
	function recalcFromLines() {
		if (lineItems.length > 0) {
			for (const li of lineItems) {
				li.amount = Math.round((li.quantity || 0) * (li.unit_price || 0) * 100) / 100;
			}
			lineItems = [...lineItems]; // trigger reactivity
			subtotal = Math.round(lineItems.reduce((sum, li) => sum + (li.amount || 0), 0) * 100) / 100;
			total_amount = Math.round(((subtotal || 0) + (tax_amount || 0)) * 100) / 100;
		}
	}

	// Auto-compute total from subtotal + tax
	$: if (subtotal !== null && tax_amount !== null) {
		total_amount = Math.round((subtotal + tax_amount) * 100) / 100;
	}

	function addLineItem() {
		lineItems = [...lineItems, { description: '', quantity: 1, unit_price: 0, amount: 0 }];
		showLineItems = true;
	}

	function removeLineItem(index: number) {
		lineItems = lineItems.filter((_, i) => i !== index);
		recalcFromLines();
	}

	const handleKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			show = false;
		}
	};

	const handleSave = async () => {
		if (!client_name.trim()) {
			toast.error($i18n.t('Client name is required'));
			return;
		}
		if (!total_amount || total_amount <= 0) {
			toast.error($i18n.t('Total amount must be greater than 0'));
			return;
		}
		if (!invoice_date) {
			toast.error($i18n.t('Invoice date is required'));
			return;
		}

		saving = true;
		try {
			const data: Record<string, any> = {
				client_name: client_name.trim(),
				invoice_date,
				total_amount,
				currency
			};

			if (invoice_number.trim()) data.invoice_number = invoice_number.trim();
			if (vendor_name.trim()) data.vendor_name = vendor_name.trim();
			if (due_date) data.due_date = due_date;
			if (subtotal !== null) data.subtotal = subtotal;
			if (tax_amount !== null) data.tax_amount = tax_amount;
			if (tax_inclusive) data.tax_inclusive = true;
			if (payment_terms.trim()) data.payment_terms = payment_terms.trim();
			if (po_number.trim()) data.po_number = po_number.trim();
			if (description.trim()) data.description = description.trim();
			if (business_unit.trim()) data.business_unit = business_unit.trim();
			if (lineItems.length > 0) {
				data.line_items = lineItems.map((li) => ({
					description: li.description,
					quantity: li.quantity,
					unit_price: li.unit_price,
					amount: li.amount
				}));
			}

			await createManualInvoice(companyId, data);
			toast.success($i18n.t('Invoice created'));
			show = false;
			await tick();
			dispatch('save');
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to create invoice')}: ${err?.detail ?? err}`);
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
			class="m-auto max-w-full w-[42rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[90dvh] shadow-3xl border border-white dark:border-gray-900 overflow-y-auto"
			in:flyAndScale
			on:mousedown={(e) => {
				e.stopPropagation();
			}}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{$i18n.t('Create Invoice')}
				</div>

				<div class="space-y-3">
					<!-- Row: Invoice # + Date + Due Date -->
					<div class="grid grid-cols-3 gap-3">
						<div>
							<label
								for="inv-number"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Invoice #')}
							</label>
							<input
								id="inv-number"
								type="text"
								bind:value={invoice_number}
								placeholder={nextNumberHint || $i18n.t('Auto-generated')}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="inv-date"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Date')} *
							</label>
							<input
								id="inv-date"
								type="date"
								bind:value={invoice_date}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="inv-due-date"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Due Date')}
							</label>
							<input
								id="inv-due-date"
								type="date"
								bind:value={due_date}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
					</div>

					<!-- Row: Vendor (From) + Client (To) -->
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label
								for="inv-vendor"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('From (Vendor)')}
							</label>
							<input
								id="inv-vendor"
								type="text"
								bind:value={vendor_name}
								placeholder={companyName || $i18n.t('Your company')}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="inv-client"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('To (Client)')} *
							</label>
							<input
								id="inv-client"
								type="text"
								bind:value={client_name}
								placeholder={$i18n.t('Client name')}
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
					</div>

					<!-- Line Items Section -->
					<div>
						<div class="flex items-center justify-between mb-2">
							<button
								type="button"
								class="flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
								on:click={() => {
									if (lineItems.length === 0) addLineItem();
									showLineItems = !showLineItems;
								}}
							>
								<svg xmlns="http://www.w3.org/2000/svg" class="size-3.5 transition-transform" class:rotate-180={showLineItems} fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor">
									<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
								</svg>
								{$i18n.t('Line Items')} ({lineItems.length})
							</button>
							{#if showLineItems}
								<button
									type="button"
									class="text-xs text-green-600 dark:text-green-400 hover:underline"
									on:click={addLineItem}
								>
									+ {$i18n.t('Add Line')}
								</button>
							{/if}
						</div>

						{#if showLineItems && lineItems.length > 0}
							<div class="border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden">
								<table class="w-full text-xs">
									<thead>
										<tr class="bg-gray-100 dark:bg-gray-900 text-gray-500 dark:text-gray-400">
											<th class="px-2 py-1.5 text-left font-medium">{$i18n.t('Description')}</th>
											<th class="px-2 py-1.5 text-right font-medium w-16">{$i18n.t('Qty')}</th>
											<th class="px-2 py-1.5 text-right font-medium w-24">{$i18n.t('Unit Price')}</th>
											<th class="px-2 py-1.5 text-right font-medium w-24">{$i18n.t('Amount')}</th>
											<th class="w-8"></th>
										</tr>
									</thead>
									<tbody>
										{#each lineItems as item, idx}
											<tr class="border-t border-gray-100 dark:border-gray-800">
												<td class="px-1 py-1">
													<input
														type="text"
														bind:value={item.description}
														placeholder={$i18n.t('Description')}
														class="w-full text-xs rounded px-2 py-1 bg-transparent dark:text-gray-200 border-0 outline-hidden focus:ring-1 focus:ring-blue-400"
													/>
												</td>
												<td class="px-1 py-1">
													<input
														type="number"
														step="0.01"
														min="0"
														bind:value={item.quantity}
														on:change={recalcFromLines}
														class="w-full text-xs text-right rounded px-2 py-1 bg-transparent dark:text-gray-200 border-0 outline-hidden focus:ring-1 focus:ring-blue-400"
													/>
												</td>
												<td class="px-1 py-1">
													<input
														type="number"
														step="0.01"
														min="0"
														bind:value={item.unit_price}
														on:change={recalcFromLines}
														class="w-full text-xs text-right rounded px-2 py-1 bg-transparent dark:text-gray-200 border-0 outline-hidden focus:ring-1 focus:ring-blue-400"
													/>
												</td>
												<td class="px-1 py-1 text-right text-gray-600 dark:text-gray-400 font-mono">
													{(item.amount || 0).toFixed(2)}
												</td>
												<td class="px-1 py-1 text-center">
													<button
														type="button"
														class="text-red-400 hover:text-red-600 text-xs"
														on:click={() => removeLineItem(idx)}
														title={$i18n.t('Remove')}
													>
														&times;
													</button>
												</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{/if}
					</div>

					<!-- Row: Subtotal + Tax + Currency + Total -->
					<div class="grid grid-cols-4 gap-3">
						<div>
							<label
								for="inv-subtotal"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Subtotal')}
							</label>
							<input
								id="inv-subtotal"
								type="number"
								step="0.01"
								min="0"
								bind:value={subtotal}
								placeholder="0.00"
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="inv-tax"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Tax')}
							</label>
							<input
								id="inv-tax"
								type="number"
								step="0.01"
								min="0"
								bind:value={tax_amount}
								placeholder="0.00"
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
						<div>
							<label
								for="inv-currency"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Currency')}
							</label>
							<input
								id="inv-currency"
								type="text"
								maxlength="3"
								bind:value={currency}
								placeholder="EUR"
								class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition uppercase"
							/>
						</div>
						<div>
							<label
								for="inv-total"
								class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>
								{$i18n.t('Total')} *
							</label>
							<input
								id="inv-total"
								type="number"
								step="0.01"
								min="0"
								bind:value={total_amount}
								placeholder="0.00"
								class="w-full text-sm font-medium rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
						</div>
					</div>

					<!-- Advanced section -->
					<div>
						<button
							type="button"
							class="flex items-center gap-1 text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 transition"
							on:click={() => {
								showAdvanced = !showAdvanced;
							}}
						>
							<svg
								xmlns="http://www.w3.org/2000/svg"
								class="size-3.5 transition-transform"
								class:rotate-180={showAdvanced}
								fill="none"
								viewBox="0 0 24 24"
								stroke-width="1.5"
								stroke="currentColor"
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
							<div class="mt-2 space-y-3">
								<div class="grid grid-cols-2 gap-3">
									<div>
										<label
											for="inv-terms"
											class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
										>
											{$i18n.t('Payment Terms')}
										</label>
										<input
											id="inv-terms"
											type="text"
											bind:value={payment_terms}
											placeholder={$i18n.t('e.g. Net 30')}
											class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
										/>
									</div>
									<div>
										<label
											for="inv-po"
											class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
										>
											{$i18n.t('PO Number')}
										</label>
										<input
											id="inv-po"
											type="text"
											bind:value={po_number}
											placeholder={$i18n.t('Purchase order ref')}
											class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
										/>
									</div>
								</div>
								<div class="grid grid-cols-2 gap-3">
									<div>
										<label
											for="inv-bu"
											class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
										>
											{$i18n.t('Business Unit')}
										</label>
										<input
											id="inv-bu"
											type="text"
											bind:value={business_unit}
											class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
										/>
									</div>
									<div class="flex items-end pb-1">
										<label class="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 cursor-pointer">
											<input type="checkbox" bind:checked={tax_inclusive} class="rounded" />
											{$i18n.t('Tax Inclusive')}
										</label>
									</div>
								</div>
								<div>
									<label
										for="inv-desc"
										class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
									>
										{$i18n.t('Description')}
									</label>
									<textarea
										id="inv-desc"
										bind:value={description}
										rows="2"
										placeholder={$i18n.t('Invoice notes or description')}
										class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition resize-none"
									></textarea>
								</div>
							</div>
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
						disabled={saving}
					>
						{#if saving}
							{$i18n.t('Creating...')}
						{:else}
							{$i18n.t('Create Invoice')}
						{/if}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}
