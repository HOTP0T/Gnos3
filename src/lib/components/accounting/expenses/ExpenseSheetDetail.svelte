<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import {
		getExpenseSheet,
		updateExpenseSheet,
		deleteExpenseSheet,
		transitionExpenseSheet,
		getExpenseSheetCandidates,
		expenseSheetPdfUrl,
		expenseSheetExcelUrl
	} from '$lib/apis/accounting';
	import K4miDocLink from '$lib/components/common/K4miDocLink.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import ExpenseSheetStatusBadge from './ExpenseSheetStatusBadge.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;
	export let sheetId: number;

	let sheet: any = null;
	let loading = true;
	let showPicker = false;
	let candidates: any[] = [];
	let selectedIds = new Set<number>();
	let note = '';
	let userInfo = '';
	let rejectionReason = '';
	let transitionTarget = '';

	const load = async () => {
		loading = true;
		try {
			sheet = await getExpenseSheet(sheetId);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		loading = false;
	};

	onMount(load);

	const openPicker = async () => {
		if (!sheet) return;
		try {
			const res = await getExpenseSheetCandidates(companyId, {
				employee_id: sheet.employee_id,
				period_start: sheet.period_start,
				period_end: sheet.period_end
			});
			const existing = new Set((sheet.lines ?? []).map((l: any) => l.invoice_id));
			candidates = (res?.invoices ?? []).filter((inv: any) => !existing.has(inv.id));
			selectedIds = new Set();
			showPicker = true;
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const addSelected = async () => {
		if (!selectedIds.size) {
			toast.error($i18n.t('Pick at least one invoice'));
			return;
		}
		try {
			sheet = await updateExpenseSheet(sheetId, {
				add_invoice_ids: Array.from(selectedIds)
			});
			toast.success($i18n.t('Receipts added'));
			showPicker = false;
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const removeLine = async (invoiceId: number) => {
		try {
			sheet = await updateExpenseSheet(sheetId, {
				remove_invoice_ids: [invoiceId]
			});
			toast.success($i18n.t('Receipt removed'));
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleDelete = async () => {
		if (!confirm($i18n.t('Delete this draft sheet?'))) return;
		try {
			await deleteExpenseSheet(sheetId);
			toast.success($i18n.t('Sheet deleted'));
			goto(`/accounting/company/${companyId}/expenses`);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const openTransition = (target: string) => {
		transitionTarget = target;
		note = '';
		rejectionReason = '';
	};

	const confirmTransition = async () => {
		try {
			sheet = await transitionExpenseSheet(sheetId, {
				target: transitionTarget as any,
				note: note || undefined,
				user_info: userInfo || undefined,
				rejection_reason:
					transitionTarget === 'reject' ? rejectionReason || undefined : undefined
			});
			toast.success($i18n.t(`Sheet ${transitionTarget}`));
			transitionTarget = '';
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const toggleCandidate = (id: number) => {
		if (selectedIds.has(id)) selectedIds.delete(id);
		else selectedIds.add(id);
		selectedIds = new Set(selectedIds);
	};

	const money = (amount: any, currency?: string) => {
		if (amount == null) return '';
		const n = typeof amount === 'number' ? amount : parseFloat(amount);
		if (!isFinite(n)) return '';
		try {
			return new Intl.NumberFormat(undefined, {
				style: 'currency',
				currency: currency || sheet?.currency || 'USD'
			}).format(n);
		} catch {
			return `${currency} ${n.toFixed(2)}`;
		}
	};

	// SSO bridge handles the URL; we just need to know whether a doc id exists.
	const hasK4miDoc = (id: number | null | undefined) => !!id;
</script>

{#if loading}
	<div class="flex justify-center py-10"><Spinner className="size-6" /></div>
{:else if !sheet}
	<div class="text-sm text-red-500 py-6">{$i18n.t('Sheet not found')}</div>
{:else}
	<div class="py-2 space-y-4">
		<!-- Header -->
		<div class="flex items-start justify-between flex-wrap gap-3">
			<div>
				<div class="flex items-center gap-2 flex-wrap">
					<button
						class="text-xs text-gray-500 hover:text-gray-700 dark:hover:text-gray-300"
						on:click={() => goto(`/accounting/company/${companyId}/expenses`)}
					>
						← {$i18n.t('All sheets')}
					</button>
				</div>
				<div class="flex items-center gap-2 mt-1 flex-wrap">
					<h2 class="text-lg font-semibold dark:text-gray-200 font-mono">
						{sheet.reference}
					</h2>
					<ExpenseSheetStatusBadge status={sheet.status} />
				</div>
				{#if sheet.title}
					<div class="text-sm text-gray-600 dark:text-gray-400 mt-0.5">{sheet.title}</div>
				{/if}
			</div>
			<div class="flex items-center gap-2 flex-wrap">
				<a
					href={expenseSheetPdfUrl(sheet.id)}
					target="_blank"
					rel="noopener"
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
				>
					{$i18n.t('PDF')}
				</a>
				<a
					href={expenseSheetExcelUrl(sheet.id)}
					target="_blank"
					rel="noopener"
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
				>
					{$i18n.t('Excel')}
				</a>

				{#if sheet.status === 'draft'}
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
						on:click={openPicker}
					>
						+ {$i18n.t('Add Receipts')}
					</button>
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition"
						on:click={() => openTransition('submit')}
					>
						{$i18n.t('Submit')}
					</button>
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500 text-white hover:bg-red-600 transition"
						on:click={handleDelete}
					>
						{$i18n.t('Delete')}
					</button>
				{:else if sheet.status === 'submitted'}
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition"
						on:click={() => openTransition('approve')}
					>
						{$i18n.t('Approve')}
					</button>
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500 text-white hover:bg-red-600 transition"
						on:click={() => openTransition('reject')}
					>
						{$i18n.t('Reject')}
					</button>
				{:else if sheet.status === 'approved'}
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition"
						on:click={() => openTransition('mark_paid')}
					>
						{$i18n.t('Mark Paid')}
					</button>
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-red-500 text-white hover:bg-red-600 transition"
						on:click={() => openTransition('reject')}
					>
						{$i18n.t('Reject')}
					</button>
				{/if}
			</div>
		</div>

		<!-- Info blocks -->
		<div class="grid grid-cols-1 md:grid-cols-3 gap-3">
			<div class="p-3 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{$i18n.t('Employee')}</div>
				<div class="text-sm font-medium dark:text-gray-200 mt-0.5">
					{sheet.employee_name ?? `#${sheet.employee_id}`}
					{#if sheet.employee_code}
						<span class="text-xs text-gray-400">({sheet.employee_code})</span>
					{/if}
				</div>
			</div>
			<div class="p-3 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{$i18n.t('Period')}</div>
				<div class="text-sm font-medium dark:text-gray-200 mt-0.5">
					{sheet.period_start} → {sheet.period_end}
				</div>
			</div>
			<div class="p-3 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{$i18n.t('Totals')}</div>
				<div class="text-sm font-medium dark:text-gray-200 mt-0.5">
					{money(sheet.total_amount, sheet.currency)}
					<span class="text-xs text-gray-400 font-normal ml-1">
						({money(sheet.subtotal, sheet.currency)} + {money(sheet.tax_amount, sheet.currency)})
					</span>
				</div>
			</div>
		</div>

		<!-- Timeline / status trail -->
		<div
			class="flex items-center gap-4 flex-wrap text-xs text-gray-500 dark:text-gray-400 px-1"
		>
			{#if sheet.submitted_at}
				<span
					>{$i18n.t('Submitted')}: {new Date(sheet.submitted_at).toLocaleString()}
					{#if sheet.submitted_by}<span class="text-gray-400">({sheet.submitted_by})</span>{/if}</span
				>
			{/if}
			{#if sheet.approved_at}
				<span
					>{$i18n.t('Approved')}: {new Date(sheet.approved_at).toLocaleString()}
					{#if sheet.approved_by}<span class="text-gray-400">({sheet.approved_by})</span>{/if}</span
				>
			{/if}
			{#if sheet.paid_at}
				<span
					>{$i18n.t('Paid')}: {new Date(sheet.paid_at).toLocaleString()}
					{#if sheet.paid_by}<span class="text-gray-400">({sheet.paid_by})</span>{/if}</span
				>
			{/if}
			{#if sheet.rejected_at}
				<span class="text-red-500"
					>{$i18n.t('Rejected')}: {new Date(sheet.rejected_at).toLocaleString()}
					{#if sheet.rejection_reason}<span class="text-gray-400 ml-1"
							>— {sheet.rejection_reason}</span
						>{/if}</span
				>
			{/if}
			{#if sheet.transaction_entry_number}
				<span
					>{$i18n.t('Journal Entry')}:
					<span class="font-mono text-gray-700 dark:text-gray-300">{sheet.transaction_entry_number}</span></span
				>
			{:else if sheet.transaction_id}
				<span>{$i18n.t('Journal Entry: draft')}</span>
			{/if}
		</div>

		<!-- Category totals -->
		{#if sheet.category_totals?.length}
			<div class="rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
				<table class="w-full text-sm">
					<thead class="text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-850">
						<tr>
							<th class="text-left py-2 px-3">{$i18n.t('Category')}</th>
							<th class="text-center py-2 px-3">{$i18n.t('Count')}</th>
							<th class="text-right py-2 px-3">{$i18n.t('Subtotal')}</th>
							<th class="text-right py-2 px-3">{$i18n.t('Tax')}</th>
							<th class="text-right py-2 px-3">{$i18n.t('Total')}</th>
						</tr>
					</thead>
					<tbody>
						{#each sheet.category_totals as g}
							<tr class="border-t border-gray-100 dark:border-gray-850">
								<td class="py-2 px-3 dark:text-gray-200">{g.category_label}</td>
								<td class="py-2 px-3 text-center text-gray-500">{g.count}</td>
								<td class="py-2 px-3 text-right">{money(g.subtotal, sheet.currency)}</td>
								<td class="py-2 px-3 text-right">{money(g.tax_amount, sheet.currency)}</td>
								<td class="py-2 px-3 text-right font-medium"
									>{money(g.total_amount, sheet.currency)}</td
								>
							</tr>
						{/each}
					</tbody>
					<tfoot class="bg-gray-50 dark:bg-gray-850 font-semibold">
						<tr class="border-t border-gray-200 dark:border-gray-800">
							<td class="py-2 px-3 dark:text-gray-200">{$i18n.t('Grand Total')}</td>
							<td></td>
							<td class="py-2 px-3 text-right">{money(sheet.subtotal, sheet.currency)}</td>
							<td class="py-2 px-3 text-right">{money(sheet.tax_amount, sheet.currency)}</td>
							<td class="py-2 px-3 text-right">{money(sheet.total_amount, sheet.currency)}</td>
						</tr>
					</tfoot>
				</table>
			</div>
		{/if}

		<!-- Lines -->
		<div class="rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
			<div
				class="px-3 py-2 text-sm font-semibold bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border-b border-gray-200 dark:border-gray-800"
			>
				{$i18n.t('Receipts')} ({sheet.lines?.length ?? 0})
			</div>
			{#if !sheet.lines?.length}
				<div class="px-3 py-6 text-sm text-gray-400 italic text-center">
					{$i18n.t('No receipts on this sheet yet.')}
				</div>
			{:else}
				<table class="w-full text-sm">
					<thead class="text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-850">
						<tr>
							<th class="text-left py-2 px-3">{$i18n.t('Date')}</th>
							<th class="text-left py-2 px-3">{$i18n.t('Vendor')}</th>
							<th class="text-left py-2 px-3">{$i18n.t('Category')}</th>
							<th class="text-left py-2 px-3">{$i18n.t('Description')}</th>
							<th class="text-right py-2 px-3">{$i18n.t('Subtotal')}</th>
							<th class="text-right py-2 px-3">{$i18n.t('Tax')}</th>
							<th class="text-right py-2 px-3">{$i18n.t('Total')}</th>
							<th class="text-right py-2 px-3">K4mi</th>
							<th></th>
						</tr>
					</thead>
					<tbody>
						{#each sheet.lines as l}
							<tr class="border-t border-gray-100 dark:border-gray-850 hover:bg-gray-50 dark:hover:bg-gray-900/50">
								<td class="py-2 px-3 text-xs text-gray-600 dark:text-gray-400">
									{l.invoice_date ?? '—'}
								</td>
								<td class="py-2 px-3 dark:text-gray-200">{l.vendor_name ?? ''}</td>
								<td class="py-2 px-3 text-gray-500">{l.category_label}</td>
								<td class="py-2 px-3 text-gray-500 max-w-xs truncate"
									>{l.description ?? ''}</td
								>
								<td class="py-2 px-3 text-right">{money(l.subtotal, l.currency)}</td>
								<td class="py-2 px-3 text-right">{money(l.tax_amount, l.currency)}</td>
								<td class="py-2 px-3 text-right font-medium"
									>{money(l.total_amount, l.currency)}</td
								>
								<td class="py-2 px-3 text-right">
									{#if hasK4miDoc(l.k4mi_document_id)}
										<K4miDocLink docId={l.k4mi_document_id} extraClass="text-xs text-blue-600 hover:text-blue-700">#{l.k4mi_document_id}</K4miDocLink>
									{/if}
								</td>
								<td class="py-2 px-3 text-right">
									{#if sheet.status === 'draft'}
										<button
											class="text-xs text-red-500 hover:text-red-700"
											on:click={() => removeLine(l.invoice_id)}>×</button
										>
									{/if}
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		</div>
	</div>

	<!-- Receipt picker modal -->
	{#if showPicker}
		<div
			class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
			on:click={() => (showPicker = false)}
		>
			<!-- svelte-ignore a11y-no-static-element-interactions -->
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<div
				class="w-full max-w-3xl max-h-[80vh] rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 flex flex-col"
				on:click|stopPropagation
			>
				<div
					class="px-4 py-3 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between"
				>
					<h3 class="text-sm font-semibold dark:text-gray-200">
						{$i18n.t('Add Receipts')} ({candidates.length})
					</h3>
					<button
						class="text-sm text-gray-500 hover:text-gray-700"
						on:click={() => (showPicker = false)}>×</button
					>
				</div>
				<div class="flex-1 overflow-y-auto p-4">
					{#if candidates.length === 0}
						<div class="text-sm text-gray-400 italic text-center py-8">
							{$i18n.t(
								'No eligible invoices for this employee in this period. Assign the employee in K4mi first.'
							)}
						</div>
					{:else}
						<table class="w-full text-xs">
							<thead class="text-gray-500 dark:text-gray-400">
								<tr>
									<th class="w-6"></th>
									<th class="text-left py-1 px-2">{$i18n.t('Date')}</th>
									<th class="text-left py-1 px-2">{$i18n.t('Vendor')}</th>
									<th class="text-right py-1 px-2">{$i18n.t('Total')}</th>
									<th class="text-right py-1 px-2">K4mi</th>
								</tr>
							</thead>
							<tbody>
								{#each candidates as inv}
									<tr
										class="border-t border-gray-100 dark:border-gray-850 hover:bg-gray-50 dark:hover:bg-gray-900/50"
									>
										<td class="px-2">
											<input
												type="checkbox"
												checked={selectedIds.has(inv.id)}
												on:change={() => toggleCandidate(inv.id)}
												class="rounded"
											/>
										</td>
										<td class="py-1 px-2 text-gray-600 dark:text-gray-400"
											>{inv.invoice_date ?? '—'}</td
										>
										<td class="py-1 px-2 dark:text-gray-200">{inv.vendor_name ?? ''}</td>
										<td class="py-1 px-2 text-right"
											>{money(inv.total_amount, inv.currency)}</td
										>
										<td class="py-1 px-2 text-right">
											{#if hasK4miDoc(inv.k4mi_document_id)}
												<K4miDocLink docId={inv.k4mi_document_id} extraClass="text-blue-600 hover:text-blue-700">#{inv.k4mi_document_id}</K4miDocLink>
											{/if}
										</td>
									</tr>
								{/each}
							</tbody>
						</table>
					{/if}
				</div>
				<div
					class="px-4 py-3 border-t border-gray-200 dark:border-gray-800 flex items-center justify-end gap-2"
				>
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
						on:click={() => (showPicker = false)}>{$i18n.t('Cancel')}</button
					>
					<button
						class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-60"
						disabled={!selectedIds.size}
						on:click={addSelected}
					>
						{$i18n.t('Add')} ({selectedIds.size})
					</button>
				</div>
			</div>
		</div>
	{/if}

	<!-- Transition modal -->
	{#if transitionTarget}
		<div
			class="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
			on:click={() => (transitionTarget = '')}
		>
			<!-- svelte-ignore a11y-no-static-element-interactions -->
			<!-- svelte-ignore a11y-click-events-have-key-events -->
			<div
				class="w-full max-w-md rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800"
				on:click|stopPropagation
			>
				<div
					class="px-4 py-3 border-b border-gray-200 dark:border-gray-800 text-sm font-semibold dark:text-gray-200"
				>
					{$i18n.t('Confirm')}: {transitionTarget}
				</div>
				<div class="p-4 space-y-3">
					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>{$i18n.t('Your name (for audit log)')}</label
						>
						<input
							type="text"
							bind:value={userInfo}
							placeholder={$i18n.t('Optional')}
							class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
						/>
					</div>
					{#if transitionTarget === 'reject'}
						<div>
							<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
								>{$i18n.t('Rejection reason')}</label
							>
							<input
								type="text"
								bind:value={rejectionReason}
								class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
							/>
						</div>
					{/if}
					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
							>{$i18n.t('Note (optional)')}</label
						>
						<input
							type="text"
							bind:value={note}
							class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
						/>
					</div>
				</div>
				<div
					class="px-4 py-3 border-t border-gray-200 dark:border-gray-800 flex items-center justify-end gap-2"
				>
					<button
						class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
						on:click={() => (transitionTarget = '')}>{$i18n.t('Cancel')}</button
					>
					<button
						class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
						on:click={confirmTransition}
					>
						{$i18n.t('Confirm')}
					</button>
				</div>
			</div>
		</div>
	{/if}
{/if}
