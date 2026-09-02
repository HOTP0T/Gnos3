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
		updateExpense,
		getCategories,
		type ExpenseItem,
		type ExpenseCategory
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

	// ── Finance correction ──────────────────────────────────────────────
	// Reviewers fix employee mistakes (wrong category, amount, tax, currency)
	// without bouncing the claim back. Mirrors the backend rule: editable until
	// the money moves. The server additionally rejects edits once the parent
	// sheet's journal entry is built — surfaced via its 409 message.
	const EDITABLE_STATUSES = ['draft', 'submitted', 'approved'];
	let categories: ExpenseCategory[] = [];
	let editing: ExpenseItem | null = null;
	let saving = false;
	let form = {
		expense_date: '',
		merchant: '',
		description: '',
		category_id: '' as string,
		currency: '',
		amount: '',
		tax_amount: '',
		payment_method: 'personal',
		reimbursable: true
	};

	function canEdit(item: ExpenseItem): boolean {
		return EDITABLE_STATUSES.includes(item.status);
	}

	async function openEdit(item: ExpenseItem) {
		editing = item;
		form = {
			expense_date: item.expense_date ?? '',
			merchant: item.merchant ?? '',
			description: item.description ?? '',
			category_id: item.category_id != null ? String(item.category_id) : '',
			currency: item.currency ?? '',
			amount: item.amount ?? '',
			tax_amount: item.tax_amount ?? '',
			payment_method: item.payment_method ?? 'personal',
			reimbursable: item.reimbursable
		};
		if (!categories.length) {
			try {
				categories = await getCategories(companyId);
			} catch {
				/* non-fatal — the category dropdown just stays empty */
			}
		}
	}

	function closeEdit() {
		editing = null;
	}

	async function saveEdit() {
		if (!editing) return;
		saving = true;
		try {
			// Send only what actually changed so a PATCH never clobbers a field
			// the reviewer didn't touch.
			const before = editing;
			const payload: Record<string, any> = {};
			if (form.expense_date && form.expense_date !== before.expense_date)
				payload.expense_date = form.expense_date;
			if (form.merchant.trim() && form.merchant !== before.merchant)
				payload.merchant = form.merchant.trim();
			if (form.description !== before.description) payload.description = form.description;
			const catId = form.category_id === '' ? null : Number(form.category_id);
			if (catId !== (before.category_id ?? null) && catId !== null) payload.category_id = catId;
			if (form.currency && form.currency.toUpperCase() !== before.currency)
				payload.currency = form.currency.toUpperCase();
			if (form.amount !== '' && form.amount !== before.amount) payload.amount = form.amount;
			if (form.tax_amount !== (before.tax_amount ?? '')) payload.tax_amount = form.tax_amount || 0;
			if (form.payment_method !== before.payment_method)
				payload.payment_method = form.payment_method;
			if (form.reimbursable !== before.reimbursable) payload.reimbursable = form.reimbursable;

			if (!Object.keys(payload).length) {
				closeEdit();
				return;
			}
			await updateExpense(editing.id, payload as any);
			toast.success('Expense updated');
			closeEdit();
			await load();
		} catch (e: any) {
			toast.error(e?.detail ?? e?.message ?? 'Could not save changes');
		} finally {
			saving = false;
		}
	}

	function onEditKey(e: KeyboardEvent) {
		if (e.key === 'Escape') closeEdit();
	}

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
									{#if canEdit(item)}
										<button
											class="rounded-md border border-gray-200 px-2 py-1 text-xs hover:bg-gray-100 dark:border-gray-700 dark:hover:bg-gray-800"
											title="Correct this claim"
											on:click={() => openEdit(item)}>Edit</button
										>
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

{#if editing}
	<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
		on:click={closeEdit}
		on:keydown={onEditKey}
		role="dialog"
		aria-modal="true"
		aria-label="Edit expense"
		tabindex="-1"
	>
		<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
		<div
			class="w-full max-w-lg rounded-xl bg-white p-5 shadow-xl dark:bg-gray-900"
			on:click|stopPropagation
		>
			<div class="mb-4 flex items-start justify-between gap-3">
				<div>
					<h3 class="text-base font-semibold text-gray-900 dark:text-gray-100">Correct expense</h3>
					<p class="mt-0.5 text-xs text-gray-500 dark:text-gray-400">
						{editing.employee_name ?? 'Employee'} · {editing.status}
						{#if editing.expense_sheet_id}· on a reimbursement sheet{/if}
					</p>
				</div>
				<button
					class="rounded-md px-2 py-1 text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
					on:click={closeEdit}
					aria-label="Close">✕</button
				>
			</div>

			<div class="grid grid-cols-2 gap-3">
				<label class="col-span-1 text-xs text-gray-500 dark:text-gray-400">
					Date
					<input
						type="date"
						bind:value={form.expense_date}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					/>
				</label>
				<label class="col-span-1 text-xs text-gray-500 dark:text-gray-400">
					Category
					<select
						bind:value={form.category_id}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					>
						<option value="">— none —</option>
						{#each categories as c}
							<option value={String(c.id)}>{c.label}</option>
						{/each}
					</select>
				</label>
				<label class="col-span-2 text-xs text-gray-500 dark:text-gray-400">
					Merchant
					<input
						type="text"
						bind:value={form.merchant}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					/>
				</label>
				<label class="col-span-2 text-xs text-gray-500 dark:text-gray-400">
					Description
					<input
						type="text"
						bind:value={form.description}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					/>
				</label>
				<label class="text-xs text-gray-500 dark:text-gray-400">
					Currency
					<input
						type="text"
						maxlength="3"
						bind:value={form.currency}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm uppercase text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					/>
				</label>
				<label class="text-xs text-gray-500 dark:text-gray-400">
					Payment
					<select
						bind:value={form.payment_method}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					>
						<option value="personal">Personal</option>
						<option value="company_card">Company card</option>
					</select>
				</label>
				<label class="text-xs text-gray-500 dark:text-gray-400">
					Total amount
					<input
						type="number"
						step="0.01"
						min="0"
						bind:value={form.amount}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					/>
				</label>
				<label class="text-xs text-gray-500 dark:text-gray-400">
					Tax amount
					<input
						type="number"
						step="0.01"
						min="0"
						bind:value={form.tax_amount}
						class="mt-1 w-full rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-sm text-gray-900 dark:border-gray-700 dark:bg-gray-850 dark:text-gray-100"
					/>
				</label>
				<label class="col-span-2 mt-1 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300">
					<input type="checkbox" class="rounded" bind:checked={form.reimbursable} />
					Reimbursable to the employee
				</label>
			</div>

			{#if editing.expense_sheet_id}
				<p class="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:bg-amber-900/20 dark:text-amber-300">
					This expense is bundled on a reimbursement sheet — saving re-calculates that sheet's
					totals.
				</p>
			{/if}

			<div class="mt-5 flex justify-end gap-2">
				<button
					class="rounded-lg border border-gray-300 px-3 py-1.5 text-sm dark:border-gray-700 dark:text-gray-200"
					on:click={closeEdit}>Cancel</button
				>
				<button
					class="rounded-lg bg-gray-900 px-4 py-1.5 text-sm font-medium text-white hover:bg-gray-700 disabled:opacity-50 dark:bg-white dark:text-gray-900"
					disabled={saving}
					on:click={saveEdit}>{saving ? 'Saving…' : 'Save changes'}</button
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
