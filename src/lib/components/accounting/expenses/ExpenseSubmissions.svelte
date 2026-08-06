<script lang="ts">
	import { onMount, onDestroy, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import {
		listCompanyExpenses,
		approveExpense,
		rejectExpense,
		receiptBlobUrl,
		type ExpenseItem
	} from '$lib/apis/expenses';
	import { createExpenseSheetFromItems } from '$lib/apis/accounting';

	const i18n: any = getContext('i18n');
	export let companyId: number;

	let items: ExpenseItem[] = [];
	let loading = true;
	let busyId: number | null = null;
	let statusFilter = 'submitted';

	// Bundling (only in the "To reimburse" view): select approved, un-bundled
	// items for a single employee and create a reimbursement sheet.
	let selected = new Set<number>();
	let bundling = false;

	$: selectableItems = items.filter((i) => i.status === 'approved' && !i.expense_sheet_id);
	$: selectedItems = items.filter((i) => selected.has(i.id));
	$: selectedEmployeeId = selectedItems.length ? selectedItems[0].employee_id : null;
	$: mixedEmployees = new Set(selectedItems.map((i) => i.employee_id)).size > 1;

	function toggle(id: number) {
		const next = new Set(selected);
		next.has(id) ? next.delete(id) : next.add(id);
		selected = next;
	}

	// One-click: select every selectable item in the same trip (same employee).
	function selectTrip(item: ExpenseItem) {
		if (!item.report_title) return;
		const next = new Set(selected);
		for (const i of selectableItems) {
			if (i.report_title === item.report_title && i.employee_id === item.employee_id) {
				next.add(i.id);
			}
		}
		selected = next;
	}

	async function bundle() {
		if (!selectedItems.length) return;
		if (mixedEmployees) {
			toast.error('Select expenses for a single employee to bundle');
			return;
		}
		bundling = true;
		try {
			const sheet = await createExpenseSheetFromItems(companyId, {
				employee_id: selectedEmployeeId as number,
				expense_item_ids: selectedItems.map((i) => i.id)
			});
			toast.success(`Reimbursement sheet ${sheet.reference} created`);
			goto(`/accounting/company/${companyId}/expenses/${sheet.id}`);
		} catch (e: any) {
			toast.error(e?.detail ?? e?.message ?? 'Could not bundle');
		} finally {
			bundling = false;
		}
	}

	// Receipt preview modal
	let previewUrl = '';
	let previewIsPdf = false;

	const FILTERS = [
		{ key: 'submitted', label: 'To review' },
		{ key: 'approved', label: 'To reimburse' },
		{ key: 'reimbursed', label: 'Reimbursed' },
		{ key: 'rejected', label: 'Rejected' },
		{ key: '', label: 'All' }
	];

	const badgeClass: Record<string, string> = {
		draft: 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300',
		submitted: 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300',
		approved: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300',
		rejected: 'bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300',
		reimbursed: 'bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300'
	};

	function fmtMoney(amount: string, currency: string): string {
		const n = Number(amount);
		return `${currency} ${isNaN(n) ? amount : n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
	}

	async function load() {
		loading = true;
		try {
			const res = await listCompanyExpenses(companyId, {
				status: statusFilter || undefined
			});
			items = res.items;
			// Drop selections that are no longer selectable (bundled/paid elsewhere).
			selected = new Set([...selected].filter((id) => items.some((i) => i.id === id)));
		} catch (e: any) {
			toast.error(e?.message ?? 'Failed to load submissions');
		} finally {
			loading = false;
		}
	}

	async function viewReceipt(item: ExpenseItem) {
		try {
			if (previewUrl) URL.revokeObjectURL(previewUrl);
			previewUrl = await receiptBlobUrl(item.id);
			previewIsPdf = !!item.receipt_filename?.toLowerCase().endsWith('.pdf');
		} catch (e: any) {
			toast.error(e?.message ?? 'Could not load receipt');
		}
	}

	function closePreview() {
		if (previewUrl) URL.revokeObjectURL(previewUrl);
		previewUrl = '';
	}

	function onPreviewKey(e: KeyboardEvent) {
		if (e.key === 'Escape') closePreview();
	}

	async function doApprove(item: ExpenseItem) {
		busyId = item.id;
		try {
			await approveExpense(item.id);
			toast.success('Approved');
			await load();
		} catch (e: any) {
			toast.error(e?.message ?? 'Could not approve');
		} finally {
			busyId = null;
		}
	}

	async function doReject(item: ExpenseItem) {
		const reason = prompt('Reason for rejection:');
		if (!reason) return;
		busyId = item.id;
		try {
			await rejectExpense(item.id, reason);
			toast.success('Rejected');
			await load();
		} catch (e: any) {
			toast.error(e?.message ?? 'Could not reject');
		} finally {
			busyId = null;
		}
	}

	function setFilter(key: string) {
		statusFilter = key;
		selected = new Set();
		load();
	}

	onMount(load);
	onDestroy(() => {
		if (previewUrl) URL.revokeObjectURL(previewUrl);
	});
</script>

<div class="flex flex-col gap-3">
	<!-- Filter tabs -->
	<div class="flex gap-2 overflow-x-auto pb-1">
		{#each FILTERS as f}
			<button
				class="whitespace-nowrap rounded-lg px-3 py-1.5 text-sm font-medium transition {statusFilter ===
				f.key
					? 'bg-gray-900 text-white dark:bg-white dark:text-gray-900'
					: 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700'}"
				on:click={() => setFilter(f.key)}
			>
				{f.label}
			</button>
		{/each}
	</div>

	{#if statusFilter === 'approved'}
		<p class="text-xs text-gray-500 dark:text-gray-400">
			Select approved expenses for one employee and bundle them into a reimbursement sheet — that
			sheet posts the journal entry when it's approved &amp; paid.
		</p>
	{/if}

	{#if loading}
		<div class="flex justify-center py-12"><Spinner /></div>
	{:else if items.length === 0}
		<div class="py-12 text-center text-gray-400">
			{statusFilter === 'submitted' ? 'Nothing awaiting review.' : 'No expenses here.'}
		</div>
	{:else}
		<div class="overflow-x-auto rounded-xl border border-gray-100 dark:border-gray-850">
			<table class="w-full text-sm">
				<thead class="bg-gray-50 text-left text-xs uppercase text-gray-500 dark:bg-gray-850 dark:text-gray-400">
					<tr>
						{#if statusFilter === 'approved'}<th class="px-3 py-2"></th>{/if}
						<th class="px-3 py-2 font-medium">Employee</th>
						<th class="px-3 py-2 font-medium">Date</th>
						<th class="px-3 py-2 font-medium">Merchant</th>
						<th class="px-3 py-2 font-medium">Category</th>
						<th class="px-3 py-2 text-right font-medium">Amount</th>
						<th class="px-3 py-2 font-medium">Status</th>
						<th class="px-3 py-2 text-right font-medium">Actions</th>
					</tr>
				</thead>
				<tbody class="divide-y divide-gray-100 dark:divide-gray-850">
					{#each items as item (item.id)}
						<tr class="hover:bg-gray-50 dark:hover:bg-gray-850/50">
							{#if statusFilter === 'approved'}
								<td class="px-3 py-2">
									{#if item.status === 'approved' && !item.expense_sheet_id}
										<input
											type="checkbox"
											class="rounded"
											checked={selected.has(item.id)}
											on:change={() => toggle(item.id)}
										/>
									{/if}
								</td>
							{/if}
							<td class="px-3 py-2">{item.employee_name ?? '—'}</td>
							<td class="px-3 py-2 whitespace-nowrap text-gray-500">{item.expense_date}</td>
							<td class="px-3 py-2">
								<div class="font-medium text-gray-800 dark:text-gray-100">{item.merchant}</div>
								<div class="max-w-xs truncate text-xs text-gray-400">{item.description}</div>
								{#if item.report_title}
									{#if statusFilter === 'approved' && item.status === 'approved' && !item.expense_sheet_id}
										<button
											class="mt-0.5 inline-block rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700"
											title="Select all in this trip"
											on:click={() => selectTrip(item)}>🧳 {item.report_title}</button
										>
									{:else}
										<div class="mt-0.5 inline-block rounded bg-gray-100 px-1.5 py-0.5 text-[10px] text-gray-500 dark:bg-gray-800 dark:text-gray-400">🧳 {item.report_title}</div>
									{/if}
								{/if}
							</td>
							<td class="px-3 py-2 text-gray-500">{item.category_label ?? '—'}</td>
							<td class="px-3 py-2 text-right font-semibold whitespace-nowrap">
								{fmtMoney(item.amount, item.currency)}
								{#if item.payment_method === 'company_card'}
									<div class="text-xs font-normal text-amber-600">company card</div>
								{/if}
							</td>
							<td class="px-3 py-2">
								<span class="inline-flex rounded-full px-2 py-0.5 text-xs font-medium {badgeClass[item.status]}">
									{item.status}
								</span>
							</td>
							<td class="px-3 py-2">
								<div class="flex items-center justify-end gap-1.5">
									{#if item.has_receipt}
										<button
											class="rounded-md border border-gray-200 px-2 py-1 text-xs hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800"
											on:click={() => viewReceipt(item)}>Receipt</button
										>
									{:else}
										<span class="text-xs text-amber-600">no receipt</span>
									{/if}
									{#if item.status === 'submitted'}
										<button
											disabled={busyId === item.id}
											class="rounded-md border border-red-300 px-2 py-1 text-xs text-red-600 hover:bg-red-50 disabled:opacity-50 dark:border-red-900/60 dark:text-red-400 dark:hover:bg-red-900/20"
											on:click={() => doReject(item)}>Reject</button
										>
										<button
											disabled={busyId === item.id}
											class="rounded-md bg-emerald-600 px-3 py-1 text-xs font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
											on:click={() => doApprove(item)}>Approve</button
										>
									{:else if item.status === 'approved' && item.expense_sheet_id}
										<a
											class="rounded-md border border-gray-200 px-2 py-1 text-xs text-blue-600 hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800"
											href={`/accounting/company/${companyId}/expenses/${item.expense_sheet_id}`}
											>On sheet →</a
										>
									{/if}
								</div>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<!-- Bundle action bar -->
{#if selected.size > 0}
	<div
		class="fixed inset-x-0 bottom-0 z-40 border-t border-gray-200 bg-white/95 px-4 py-3 backdrop-blur dark:border-gray-800 dark:bg-gray-900/95"
	>
		<div class="mx-auto flex max-w-4xl items-center justify-between gap-3">
			<div class="text-sm text-gray-700 dark:text-gray-200">
				<span class="font-semibold">{selected.size}</span> selected
				{#if mixedEmployees}
					<span class="ml-2 text-red-600">— select one employee only</span>
				{:else if selectedItems.length}
					· {selectedItems[0].employee_name}
				{/if}
			</div>
			<div class="flex gap-2">
				<button
					class="rounded-lg border border-gray-300 px-3 py-1.5 text-sm dark:border-gray-700 dark:text-gray-200"
					on:click={() => (selected = new Set())}>Clear</button
				>
				<button
					class="rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-gray-700 disabled:opacity-50 dark:bg-white dark:text-gray-900"
					disabled={bundling || mixedEmployees}
					on:click={bundle}>{bundling ? 'Bundling…' : 'Bundle into reimbursement sheet'}</button
				>
			</div>
		</div>
	</div>
{/if}

{#if previewUrl}
	<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
		on:click={closePreview}
		on:keydown={onPreviewKey}
		role="dialog"
		aria-modal="true"
		aria-label="Receipt preview"
		tabindex="-1"
	>
		<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
		<div class="relative max-h-[90vh] w-full max-w-3xl" on:click|stopPropagation>
			<button
				class="absolute -right-3 -top-3 z-10 flex h-8 w-8 items-center justify-center rounded-full bg-white text-gray-700 shadow-md hover:bg-gray-100 dark:bg-gray-800 dark:text-gray-200"
				on:click={closePreview}
				aria-label="Close"
			>
				✕
			</button>
			{#if previewIsPdf}
				<iframe
					src={previewUrl}
					title="Receipt"
					class="h-[85vh] w-full rounded-lg border-0 bg-white"
				></iframe>
			{:else}
				<img
					src={previewUrl}
					alt="Receipt"
					class="mx-auto max-h-[85vh] max-w-full rounded-lg object-contain"
				/>
			{/if}
		</div>
	</div>
{/if}
