<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import type { Writable } from 'svelte/store';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import { getPayments, deletePayment, getAccounts } from '$lib/apis/accounting';
	import { convertAmount } from '$lib/utils/currency';

	import Pagination from '$lib/components/common/Pagination.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import PaymentFormModal from '$lib/components/accounting/PaymentFormModal.svelte';

	const i18n = getContext('i18n');
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrencyCtx = getContext<Writable<string>>('companyCurrency');

	export let companyId: number;

	// Data
	let payments: any[] = [];
	let total = 0;
	let loading = true;
	let accounts: any[] = [];

	// Pagination
	let page = 1;
	let perPage = 20;

	// Filters
	let filterDirection = '';
	let filterMethod = '';
	let filterDateFrom = '';
	let filterDateTo = '';

	// Delete confirmation
	let showDeleteConfirm = false;
	let deleteTarget: any = null;

	// Form modal
	let showFormModal = false;

	// ─── Method badge colors ────────────────────────────────────────────────────

	const METHOD_BADGE: Record<string, string> = {
		cash: 'bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-200',
		bank_transfer: 'bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-200',
		check: 'bg-gray-100 text-gray-800 dark:bg-gray-500/20 dark:text-gray-200',
		credit_card: 'bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-200',
		other: 'bg-gray-100 text-gray-800 dark:bg-gray-500/20 dark:text-gray-200'
	};

	const METHOD_LABELS: Record<string, string> = {
		cash: 'Cash',
		bank_transfer: 'Bank Transfer',
		check: 'Check',
		credit_card: 'Credit Card',
		other: 'Other'
	};

	// ─── Helpers ────────────────────────────────────────────────────────────────

	const formatCurrency = (val: any) => {
		if (val === null || val === undefined) return '-';
		return parseFloat(val).toLocaleString(undefined, {
			minimumFractionDigits: 2,
			maximumFractionDigits: 2
		});
	};

	const formatDate = (val: any) => {
		if (!val) return '-';
		return dayjs(val).format('YYYY-MM-DD');
	};

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

	// ─── Data loading ───────────────────────────────────────────────────────────

	const loadPayments = async () => {
		loading = true;
		try {
			const params: Record<string, any> = {
				limit: perPage,
				offset: (page - 1) * perPage,
				company_id: companyId
			};
			if (filterDirection) params.direction = filterDirection;
			if (filterMethod) params.method = filterMethod;
			if (filterDateFrom) params.date_from = filterDateFrom;
			if (filterDateTo) params.date_to = filterDateTo;

			const res = await getPayments(params);
			if (res) {
				if (Array.isArray(res)) {
					payments = res;
					total = res.length;
				} else {
					payments = res.payments ?? [];
					total = res.total ?? payments.length;
				}
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const loadAccounts = async () => {
		try {
			const res = await getAccounts({ company_id: companyId });
			if (Array.isArray(res)) {
				accounts = res;
			} else {
				accounts = res?.items ?? res?.accounts ?? [];
			}
		} catch {
			accounts = [];
		}
	};

	// ─── Filter handlers ────────────────────────────────────────────────────────

	const applyFilters = () => {
		page = 1;
		loadPayments();
	};

	const resetFilters = () => {
		filterDirection = '';
		filterMethod = '';
		filterDateFrom = '';
		filterDateTo = '';
		page = 1;
		loadPayments();
	};

	// ─── Delete ─────────────────────────────────────────────────────────────────

	const confirmDelete = (payment: any) => {
		deleteTarget = payment;
		showDeleteConfirm = true;
	};

	const handleDelete = async () => {
		if (!deleteTarget) return;
		try {
			await deletePayment(deleteTarget.id);
			toast.success($i18n.t('Payment deleted'));
			if (payments.length === 1 && page > 1) {
				page -= 1;
			}
			loadPayments();
		} catch (err) {
			toast.error(`${err}`);
		}
		deleteTarget = null;
	};

	// ─── Form modal ─────────────────────────────────────────────────────────────

	const handleFormSave = () => {
		loadPayments();
	};

	// ─── Lifecycle ──────────────────────────────────────────────────────────────

	let mounted = false;

	$: if (mounted && page) {
		loadPayments();
	}

	onMount(() => {
		loadPayments().then(() => {
			mounted = true;
		});
		loadAccounts();
	});
</script>

<ConfirmDialog
	bind:show={showDeleteConfirm}
	on:confirm={handleDelete}
	title={$i18n.t('Delete Payment')}
	message={$i18n.t('Are you sure you want to delete this payment? This action cannot be undone.')}
/>

<PaymentFormModal bind:show={showFormModal} {accounts} {companyId} on:save={handleFormSave} />

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Payments')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{total}
			</div>
		</div>

		<div class="flex gap-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition"
				on:click={() => {
					showFormModal = true;
				}}
			>
				{$i18n.t('Record Payment')}
			</button>
		</div>
	</div>

	<!-- Filter Bar -->
	<div
		class="flex flex-wrap items-end gap-2 py-2 px-0.5"
	>
		<div>
			<label
				for="filter-direction"
				class="block text-xs text-gray-500 dark:text-gray-400 mb-0.5"
			>
				{$i18n.t('Direction')}
			</label>
			<select
				id="filter-direction"
				bind:value={filterDirection}
				on:change={applyFilters}
				class="text-xs bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-200 dark:border-gray-800 rounded-lg px-2 py-1.5"
			>
				<option value="">{$i18n.t('All')}</option>
				<option value="inbound">{$i18n.t('Inbound')}</option>
				<option value="outbound">{$i18n.t('Outbound')}</option>
			</select>
		</div>

		<div>
			<label
				for="filter-method"
				class="block text-xs text-gray-500 dark:text-gray-400 mb-0.5"
			>
				{$i18n.t('Method')}
			</label>
			<select
				id="filter-method"
				bind:value={filterMethod}
				on:change={applyFilters}
				class="text-xs bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-200 dark:border-gray-800 rounded-lg px-2 py-1.5"
			>
				<option value="">{$i18n.t('All')}</option>
				<option value="cash">{$i18n.t('Cash')}</option>
				<option value="bank_transfer">{$i18n.t('Bank Transfer')}</option>
				<option value="check">{$i18n.t('Check')}</option>
				<option value="credit_card">{$i18n.t('Credit Card')}</option>
				<option value="other">{$i18n.t('Other')}</option>
			</select>
		</div>

		<div>
			<label
				for="filter-date-from"
				class="block text-xs text-gray-500 dark:text-gray-400 mb-0.5"
			>
				{$i18n.t('From')}
			</label>
			<input
				id="filter-date-from"
				type="date"
				bind:value={filterDateFrom}
				on:change={applyFilters}
				class="text-xs bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-200 dark:border-gray-800 rounded-lg px-2 py-1.5"
			/>
		</div>

		<div>
			<label
				for="filter-date-to"
				class="block text-xs text-gray-500 dark:text-gray-400 mb-0.5"
			>
				{$i18n.t('To')}
			</label>
			<input
				id="filter-date-to"
				type="date"
				bind:value={filterDateTo}
				on:change={applyFilters}
				class="text-xs bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-200 dark:border-gray-800 rounded-lg px-2 py-1.5"
			/>
		</div>

		{#if filterDirection || filterMethod || filterDateFrom || filterDateTo}
			<button
				class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition self-end pb-1.5"
				on:click={resetFilters}
			>
				{$i18n.t('Clear')}
			</button>
		{/if}
	</div>

	<!-- Table -->
	{#if loading && payments.length === 0}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if payments.length === 0}
		<div class="flex justify-center my-10 text-sm text-gray-500">
			{$i18n.t('No payments found')}
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table
				class="w-full text-sm text-left text-gray-900 dark:text-gray-100 table-auto"
			>
				<thead
					class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
				>
					<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('#')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Date')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Direction')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap text-right">{$i18n.t('Amount')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Currency')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Method')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Payer / Payee')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Invoice')}</th>
						<th scope="col" class="px-2.5 py-2 whitespace-nowrap">{$i18n.t('Reference')}</th>
						<th scope="col" class="px-2.5 py-2 text-right whitespace-nowrap">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each payments as payment (payment.id)}
						<tr
							class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
						>
							<!-- # -->
							<td class="px-3 py-1.5 whitespace-nowrap text-gray-400 font-mono text-[10px]">
								{payment.id}
							</td>
							<!-- Date -->
							<td class="px-3 py-1.5 whitespace-nowrap">
								{formatDate(payment.payment_date)}
							</td>

							<!-- Direction Badge -->
							<td class="px-3 py-1.5">
								{#if payment.direction === 'inbound'}
									<span
										class="text-xs font-medium bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-200 w-fit px-[5px] rounded-lg uppercase line-clamp-1"
									>
										{$i18n.t('In')}
									</span>
								{:else}
									<span
										class="text-xs font-medium bg-red-100 text-red-800 dark:bg-red-500/20 dark:text-red-200 w-fit px-[5px] rounded-lg uppercase line-clamp-1"
									>
										{$i18n.t('Out')}
									</span>
								{/if}
							</td>

							<!-- Amount -->
							<td class="px-3 py-1.5 text-right whitespace-nowrap font-medium">
								{#key $displayCurrency}
								{#if isConverting}
									{@const c = cvt(payment.amount, payment.payment_date)}
									{#if c.hasRate}
										<span class="font-medium">{c.display} <span class="text-[9px] text-gray-400">{$displayCurrency}</span></span>
										<div class="text-[9px] text-gray-400">{c.original} {nativeCurrency}</div>
									{:else}
										<span>{c.original} {nativeCurrency}</span>
										<span class="text-[9px] text-amber-500 italic" title="No exchange rate available">&#9888;</span>
									{/if}
								{:else}
									{formatCurrency(payment.amount)}
								{/if}
								{/key}
							</td>

							<!-- Currency -->
							<td class="px-3 py-1.5 uppercase text-gray-500">
								{payment.currency ?? '-'}
							</td>

							<!-- Method Badge -->
							<td class="px-3 py-1.5">
								<span
									class="text-xs font-medium {METHOD_BADGE[payment.method] ?? METHOD_BADGE.other} w-fit px-[5px] rounded-lg uppercase line-clamp-1"
								>
									{$i18n.t(METHOD_LABELS[payment.method] ?? payment.method ?? '-')}
								</span>
							</td>

							<!-- Payer / Payee -->
							<td class="px-3 py-1.5 max-w-[160px] truncate">
								{#if payment.direction === 'inbound'}
									{payment.payer || '-'}
								{:else}
									{payment.payee || '-'}
								{/if}
							</td>

							<!-- Invoice Link -->
							<td class="px-3 py-1.5">
								{#if payment.invoice_id}
									<a
										href="/accounting/company/{companyId}/entries?invoice_id={payment.invoice_id}"
										class="text-blue-600 dark:text-blue-400 hover:underline"
									>
										#{payment.invoice_id}
									</a>
								{:else}
									<span class="text-gray-400">-</span>
								{/if}
							</td>

							<!-- Reference -->
							<td class="px-3 py-1.5 max-w-[140px] truncate">
								{#if payment.reference}
									<Tooltip content={payment.reference}>
										<span class="block overflow-hidden text-ellipsis whitespace-nowrap">
											{payment.reference}
										</span>
									</Tooltip>
								{:else}
									<span class="text-gray-400">-</span>
								{/if}
							</td>

							<!-- Actions -->
							<td class="px-3 py-1.5 text-right whitespace-nowrap">
								<div class="flex items-center justify-end gap-1">
									<Tooltip content={$i18n.t('Delete')}>
										<button
											class="p-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition text-red-500"
											on:click={() => confirmDelete(payment)}
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
													d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
												/>
											</svg>
										</button>
									</Tooltip>
								</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>

		<!-- Pagination -->
		<div class="flex items-center justify-between mt-2">
			<div class="flex items-center gap-2">
				<span class="text-xs text-gray-500">{$i18n.t('Per page')}:</span>
				<select
					class="text-xs bg-transparent dark:bg-gray-900 dark:text-gray-300 outline-hidden border border-gray-100/30 dark:border-gray-850/30 rounded-lg px-2 py-1"
					bind:value={perPage}
					on:change={() => {
						page = 1;
						loadPayments();
					}}
				>
					<option value={20}>20</option>
					<option value={50}>50</option>
					<option value={100}>100</option>
				</select>
			</div>

			<Pagination bind:page count={total} {perPage} />
		</div>
	{/if}
</div>
