<script lang="ts">
	import { getContext } from 'svelte';
	import dayjs from 'dayjs';
	import { toast } from 'svelte-sonner';

	import Modal from '$lib/components/common/Modal.svelte';
	import { updateInvoice } from '$lib/apis/invoices';
	import { INVOICE_API_BASE_URL, K4MI_BASE_URL } from '$lib/constants';

	const i18n = getContext('i18n');

	export let show = false;
	export let invoice: any = null;
	export let onUpdate: (updated: any) => void = () => {};

	const INVOICE_API_BASE = INVOICE_API_BASE_URL;

	$: previewUrl = invoice ? `${INVOICE_API_BASE}/api/invoices/${invoice.id}/preview` : '';

	// Editable field definitions
	const EDITABLE_FIELDS = [
		{ key: 'vendor_name', label: 'Vendor', type: 'text', section: 'info' },
		{ key: 'invoice_number', label: 'Invoice #', type: 'text', section: 'info' },
		{ key: 'invoice_date', label: 'Date', type: 'date', section: 'info' },
		{ key: 'due_date', label: 'Due Date', type: 'date', section: 'info' },
		{ key: 'client_name', label: 'Client', type: 'text', section: 'info' },
		{ key: 'subtotal', label: 'Subtotal', type: 'number', section: 'money' },
		{ key: 'tax_amount', label: 'Tax', type: 'number', section: 'money' },
		{ key: 'total_amount', label: 'Total', type: 'number', section: 'money', bold: true },
		{ key: 'amount_paid', label: 'Paid', type: 'number', section: 'money' },
		{ key: 'balance_due', label: 'Balance Due', type: 'number', section: 'money' },
		{ key: 'payment_terms', label: 'Payment Terms', type: 'text', section: 'details' },
		{ key: 'po_number', label: 'PO #', type: 'text', section: 'details' },
		{ key: 'description', label: 'Description', type: 'text', section: 'details' }
	];

	// Edit state
	let editingField: string | null = null;
	let editValue = '';

	function startEdit(field: (typeof EDITABLE_FIELDS)[0]) {
		if (!invoice) return;
		editingField = field.key;
		const val = invoice[field.key];
		if (field.type === 'date' && val) {
			editValue = dayjs(val).format('YYYY-MM-DD');
		} else if (field.type === 'number' && val !== null && val !== undefined) {
			editValue = String(parseFloat(val));
		} else {
			editValue = val ?? '';
		}
	}

	function cancelEdit() {
		editingField = null;
	}

	async function saveEdit() {
		if (!editingField || !invoice) return;

		const field = EDITABLE_FIELDS.find((f) => f.key === editingField);
		if (!field) return;

		let updateVal: any = editValue;
		if (field.type === 'number') {
			updateVal = editValue === '' ? null : parseFloat(editValue);
		} else if (field.type === 'date') {
			updateVal = editValue === '' ? null : editValue;
		} else {
			updateVal = editValue === '' ? null : editValue;
		}

		// Skip if unchanged
		const oldVal = invoice[field.key];
		const oldStr = oldVal === null || oldVal === undefined ? '' : String(oldVal);
		const newStr = updateVal === null ? '' : String(updateVal);
		if (
			oldStr === newStr ||
			(field.type === 'date' && dayjs(oldStr).format('YYYY-MM-DD') === newStr)
		) {
			editingField = null;
			return;
		}

		try {
			const res = await updateInvoice(localStorage.token, invoice.id, {
				[field.key]: updateVal
			});
			if (res) {
				invoice = res;
				onUpdate(res);
				toast.success($i18n.t('Updated'));
			}
		} catch (err) {
			toast.error(`${err}`);
		}

		editingField = null;
	}

	function handleKeydown(e: KeyboardEvent) {
		if (e.key === 'Enter') {
			e.preventDefault();
			saveEdit();
		} else if (e.key === 'Escape') {
			cancelEdit();
		}
	}

	function formatMoney(val: any, currency: string): string {
		if (val === null || val === undefined) return '-';
		return `${currency}${parseFloat(val).toFixed(2)}`;
	}

	function formatDate(val: any): string {
		if (!val) return '-';
		return dayjs(val).format('YYYY-MM-DD');
	}

	function displayValue(field: (typeof EDITABLE_FIELDS)[0]): string {
		const val = invoice?.[field.key];
		if (field.type === 'date') return formatDate(val);
		if (field.type === 'number') return formatMoney(val, invoice?.currency ?? '');
		return val ?? '-';
	}
</script>

<Modal size="2xl" bind:show>
	<div class="p-5 max-h-[90vh] overflow-y-auto">
		<div class="flex justify-between items-center mb-4">
			<div class="text-lg font-medium dark:text-gray-200">
				{$i18n.t('Document Preview')}
			</div>
			<div class="flex items-center gap-2">
				{#if invoice?.k4mi_document_id}
					<a
						href="{K4MI_BASE_URL}/documents/{invoice.k4mi_document_id}/details"
						target="_blank"
						rel="noopener noreferrer"
						class="text-xs px-3 py-1.5 bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 transition rounded-lg font-medium dark:text-gray-200"
					>
						{$i18n.t('Open in K4mi')}
					</a>
				{/if}
				<button
					class="self-center p-1 hover:bg-gray-100 dark:hover:bg-gray-850 rounded-lg transition"
					on:click={() => {
						show = false;
					}}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						stroke-width="2"
						stroke="currentColor"
						class="size-5"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							d="M6 18 18 6M6 6l12 12"
						/>
					</svg>
				</button>
			</div>
		</div>

		{#if invoice}
			<div class="flex flex-col lg:flex-row gap-4">
				<!-- Left: Document Preview -->
				<div class="flex-1 min-w-0">
					<div
						class="bg-gray-50 dark:bg-gray-850 rounded-lg overflow-hidden"
						style="min-height: 70vh;"
					>
						<iframe
							src={previewUrl}
							title="Invoice Document"
							class="w-full border-0"
							style="min-height: 70vh;"
						/>
					</div>
				</div>

				<!-- Right: Extracted Data (editable) -->
				<div class="w-full lg:w-96 space-y-3 lg:max-h-[75vh] lg:overflow-y-auto">
					<div class="text-sm font-medium dark:text-gray-200">
						{$i18n.t('Extracted Data')}
					</div>

					<div class="space-y-2 text-xs">
						<!-- Info fields -->
						{#each EDITABLE_FIELDS.filter((f) => f.section === 'info') as field}
							<div class="flex justify-between items-center {field.bold ? 'font-medium' : ''}">
								<span class="text-gray-500">{$i18n.t(field.label)}</span>
								{#if editingField === field.key}
									<input
										type={field.type === 'date' ? 'date' : field.type === 'number' ? 'number' : 'text'}
										step={field.type === 'number' ? '0.01' : undefined}
										class="w-40 text-xs text-right bg-transparent dark:text-gray-200 outline-hidden border-b border-blue-500 px-1 py-0.5"
										bind:value={editValue}
										on:keydown={handleKeydown}
										on:blur={saveEdit}
										autofocus
									/>
								{:else}
									<button
										class="dark:text-gray-200 text-right hover:bg-gray-100 dark:hover:bg-gray-800 px-1 py-0.5 rounded cursor-pointer transition max-w-[60%] truncate"
										on:click={() => startEdit(field)}
										title={$i18n.t('Click to edit')}
									>
										{displayValue(field)}
									</button>
								{/if}
							</div>
						{/each}

						<hr class="border-gray-100/30 dark:border-gray-850/30" />

						<!-- Money fields -->
						{#each EDITABLE_FIELDS.filter((f) => f.section === 'money') as field}
							<div class="flex justify-between items-center {field.bold ? 'font-medium' : ''}">
								<span class="text-gray-500">{$i18n.t(field.label)}</span>
								{#if editingField === field.key}
									<input
										type="number"
										step="0.01"
										class="w-40 text-xs text-right bg-transparent dark:text-gray-200 outline-hidden border-b border-blue-500 px-1 py-0.5"
										bind:value={editValue}
										on:keydown={handleKeydown}
										on:blur={saveEdit}
										autofocus
									/>
								{:else}
									<button
										class="dark:text-gray-200 text-right hover:bg-gray-100 dark:hover:bg-gray-800 px-1 py-0.5 rounded cursor-pointer transition"
										on:click={() => startEdit(field)}
										title={$i18n.t('Click to edit')}
									>
										{displayValue(field)}
									</button>
								{/if}
							</div>
						{/each}

						<hr class="border-gray-100/30 dark:border-gray-850/30" />

						<!-- Detail fields -->
						{#each EDITABLE_FIELDS.filter((f) => f.section === 'details') as field}
							<div class="flex justify-between items-center">
								<span class="text-gray-500">{$i18n.t(field.label)}</span>
								{#if editingField === field.key}
									<input
										type="text"
										class="w-40 text-xs text-right bg-transparent dark:text-gray-200 outline-hidden border-b border-blue-500 px-1 py-0.5"
										bind:value={editValue}
										on:keydown={handleKeydown}
										on:blur={saveEdit}
										autofocus
									/>
								{:else}
									<button
										class="dark:text-gray-200 text-right hover:bg-gray-100 dark:hover:bg-gray-800 px-1 py-0.5 rounded cursor-pointer transition max-w-[60%] truncate"
										on:click={() => startEdit(field)}
										title={$i18n.t('Click to edit')}
									>
										{displayValue(field)}
									</button>
								{/if}
							</div>
						{/each}

						<hr class="border-gray-100/30 dark:border-gray-850/30" />

						<!-- Read-only metadata -->
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Model')}</span>
							<span class="dark:text-gray-200"
								>{invoice.extraction_model ?? '-'}</span
							>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Confidence')}</span>
							<span class="dark:text-gray-200">
								{invoice.confidence_score !== null
									? `${(parseFloat(invoice.confidence_score) * 100).toFixed(0)}%`
									: '-'}
							</span>
						</div>
					</div>

					<!-- Tags -->
					{#if invoice.k4mi_tags?.length}
						<div>
							<div class="text-xs text-gray-500 mb-1">{$i18n.t('Tags')}</div>
							<div class="flex flex-wrap gap-1">
								{#each invoice.k4mi_tags as tag}
									<span
										class="px-1.5 py-0.5 rounded-xl bg-gray-100 dark:bg-gray-850 text-xs dark:text-gray-200"
									>
										{tag}
									</span>
								{/each}
							</div>
						</div>
					{/if}

					<!-- Notes -->
					{#if invoice.k4mi_notes?.length}
						<div>
							<div class="text-xs text-gray-500 mb-1">{$i18n.t('Notes')}</div>
							<div class="space-y-1">
								{#each invoice.k4mi_notes as note}
									<div
										class="text-xs bg-gray-50 dark:bg-gray-850 rounded-lg p-2 dark:text-gray-300"
									>
										{note.text || note.note || ''}
									</div>
								{/each}
							</div>
						</div>
					{/if}
				</div>
			</div>
		{/if}
	</div>
</Modal>
