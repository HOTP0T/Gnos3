<script lang="ts">
	import { getContext } from 'svelte';
	import dayjs from 'dayjs';

	import Modal from '$lib/components/common/Modal.svelte';

	const i18n = getContext('i18n');

	export let show = false;
	export let invoice: any = null;

	// K4mi is accessible at localhost:8000 from the browser
	const K4MI_BASE_URL = 'http://localhost:8000';

	$: previewUrl = invoice
		? `${K4MI_BASE_URL}/api/documents/${invoice.k4mi_document_id}/preview/`
		: '';
	$: downloadUrl = invoice
		? `${K4MI_BASE_URL}/api/documents/${invoice.k4mi_document_id}/download/`
		: '';
</script>

<Modal size="xl" bind:show>
	<div class="p-5">
		<div class="flex justify-between items-center mb-4">
			<div class="text-lg font-medium dark:text-gray-200">
				{$i18n.t('Document Preview')}
			</div>
			<div class="flex items-center gap-2">
				{#if invoice?.k4mi_document_url}
					<a
						href={invoice.k4mi_document_url}
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
						style="min-height: 500px;"
					>
						<iframe
							src={downloadUrl}
							title="Invoice Document"
							class="w-full border-0"
							style="min-height: 500px;"
						/>
					</div>
				</div>

				<!-- Right: Extracted Data -->
				<div class="w-full lg:w-80 space-y-3">
					<div class="text-sm font-medium dark:text-gray-200">
						{$i18n.t('Extracted Data')}
					</div>

					<div class="space-y-2 text-xs">
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Vendor')}</span>
							<span class="dark:text-gray-200 text-right"
								>{invoice.vendor_name}</span
							>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Invoice #')}</span>
							<span class="dark:text-gray-200"
								>{invoice.invoice_number ?? '-'}</span
							>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Date')}</span>
							<span class="dark:text-gray-200">
								{invoice.invoice_date
									? dayjs(invoice.invoice_date).format('YYYY-MM-DD')
									: '-'}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Due Date')}</span>
							<span class="dark:text-gray-200">
								{invoice.due_date
									? dayjs(invoice.due_date).format('YYYY-MM-DD')
									: '-'}
							</span>
						</div>

						<hr class="border-gray-100/30 dark:border-gray-850/30" />

						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Subtotal')}</span>
							<span class="dark:text-gray-200">
								{invoice.subtotal !== null
									? `${invoice.currency}${parseFloat(invoice.subtotal).toFixed(2)}`
									: '-'}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Tax')}</span>
							<span class="dark:text-gray-200">
								{invoice.tax_amount !== null
									? `${invoice.currency}${parseFloat(invoice.tax_amount).toFixed(2)}`
									: '-'}
							</span>
						</div>
						<div class="flex justify-between font-medium">
							<span class="text-gray-500">{$i18n.t('Total')}</span>
							<span class="dark:text-gray-200">
								{invoice.currency}{parseFloat(invoice.total_amount).toFixed(2)}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Paid')}</span>
							<span class="dark:text-gray-200">
								{invoice.amount_paid !== null
									? `${invoice.currency}${parseFloat(invoice.amount_paid).toFixed(2)}`
									: '-'}
							</span>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Balance Due')}</span>
							<span class="dark:text-gray-200">
								{invoice.balance_due !== null
									? `${invoice.currency}${parseFloat(invoice.balance_due).toFixed(2)}`
									: '-'}
							</span>
						</div>

						<hr class="border-gray-100/30 dark:border-gray-850/30" />

						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Payment Terms')}</span>
							<span class="dark:text-gray-200"
								>{invoice.payment_terms ?? '-'}</span
							>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('PO #')}</span>
							<span class="dark:text-gray-200"
								>{invoice.po_number ?? '-'}</span
							>
						</div>
						<div class="flex justify-between">
							<span class="text-gray-500">{$i18n.t('Description')}</span>
							<span class="dark:text-gray-200 text-right max-w-[60%]"
								>{invoice.description ?? '-'}</span
							>
						</div>

						<hr class="border-gray-100/30 dark:border-gray-850/30" />

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
