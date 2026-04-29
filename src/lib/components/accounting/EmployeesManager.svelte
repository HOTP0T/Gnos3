<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import {
		getEmployees,
		createEmployee,
		updateEmployee,
		deleteEmployee,
		restoreEmployee,
		syncEmployeesToK4mi,
		getAccounts,
		getExpenseCategories
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let employees: any[] = [];
	let accounts: any[] = [];
	let categories: any[] = [];
	let loading = true;
	let showForm = false;
	let editingId: number | null = null;
	let showInactive = false;
	let searchTerm = '';

	const emptyForm = () => ({
		code: '',
		full_name: '',
		email: '',
		department: '',
		reimbursement_account_id: null as number | null,
		default_expense_category_id: null as number | null,
		is_active: true
	});

	let form = emptyForm();

	const load = async () => {
		loading = true;
		try {
			const [empData, acctData, catData] = await Promise.all([
				getEmployees(companyId, {
					active: showInactive ? undefined : true,
					search: searchTerm || undefined
				}),
				getAccounts({ company_id: companyId }),
				getExpenseCategories(companyId)
			]);
			employees = empData?.employees ?? [];
			const accts = acctData?.accounts ?? acctData ?? [];
			accounts = Array.isArray(accts) ? accts : [];
			categories = catData?.categories ?? [];
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		loading = false;
	};

	onMount(load);

	const resetForm = () => {
		form = emptyForm();
		editingId = null;
		showForm = false;
	};

	const openNew = () => {
		form = emptyForm();
		editingId = null;
		showForm = true;
	};

	const openEdit = (emp: any) => {
		form = {
			code: emp.code,
			full_name: emp.full_name,
			email: emp.email ?? '',
			department: emp.department ?? '',
			reimbursement_account_id: emp.reimbursement_account_id,
			default_expense_category_id: emp.default_expense_category_id,
			is_active: emp.is_active
		};
		editingId = emp.id;
		showForm = true;
	};

	const handleSubmit = async () => {
		if (!form.code.trim() || !form.full_name.trim()) {
			toast.error($i18n.t('Code and full name are required'));
			return;
		}
		const payload: Record<string, any> = {
			code: form.code.trim(),
			full_name: form.full_name.trim(),
			email: form.email.trim() || null,
			department: form.department.trim() || null,
			reimbursement_account_id: form.reimbursement_account_id ?? null,
			default_expense_category_id: form.default_expense_category_id ?? null,
			is_active: form.is_active
		};
		try {
			if (editingId) {
				await updateEmployee(editingId, payload);
				toast.success($i18n.t('Employee updated'));
			} else {
				await createEmployee(companyId, payload);
				toast.success($i18n.t('Employee created'));
			}
			resetForm();
			await load();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleDeactivate = async (id: number) => {
		try {
			await deleteEmployee(id);
			toast.success($i18n.t('Employee deactivated'));
			await load();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleRestore = async (id: number) => {
		try {
			await restoreEmployee(id);
			toast.success($i18n.t('Employee restored'));
			await load();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleSyncK4mi = async () => {
		try {
			await syncEmployeesToK4mi(companyId);
			toast.success($i18n.t('K4mi Employee field synced'));
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const reload = async () => {
		await load();
	};

	const accountLabel = (id: number | null) => {
		if (!id) return '';
		const a = accounts.find((x) => x.id === id);
		return a ? `${a.code} — ${a.name}` : '';
	};

	const categoryLabel = (id: number | null) => {
		if (!id) return '';
		const c = categories.find((x) => x.id === id);
		return c ? c.label : '';
	};
</script>

<div class="space-y-3">
	<div class="flex items-center justify-between flex-wrap gap-2">
		<h3 class="text-sm font-semibold dark:text-gray-200">{$i18n.t('Employees')}</h3>
		<div class="flex items-center gap-2">
			<input
				type="text"
				bind:value={searchTerm}
				on:input={reload}
				placeholder={$i18n.t('Search...')}
				class="text-xs rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
			/>
			<label class="flex items-center gap-1 text-xs text-gray-600 dark:text-gray-400">
				<input type="checkbox" bind:checked={showInactive} on:change={reload} class="rounded" />
				{$i18n.t('Show inactive')}
			</label>
			<button
				class="px-3 py-1 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
				on:click={handleSyncK4mi}
				title={$i18n.t('Force resync of the K4mi Employee select field')}
			>
				{$i18n.t('Sync K4mi')}
			</button>
			<button
				class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
				on:click={showForm ? resetForm : openNew}
			>
				{showForm ? $i18n.t('Cancel') : $i18n.t('New Employee')}
			</button>
		</div>
	</div>

	{#if showForm}
		<div
			class="p-4 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-800 space-y-3"
		>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-2">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Code')}</label
					>
					<input
						type="text"
						bind:value={form.code}
						placeholder="EMP001"
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div class="md:col-span-2">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Full Name')}</label
					>
					<input
						type="text"
						bind:value={form.full_name}
						placeholder="Alice Nguyen"
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Department')}</label
					>
					<input
						type="text"
						bind:value={form.department}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div class="md:col-span-2">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Email')}</label
					>
					<input
						type="email"
						bind:value={form.email}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Reimbursement Account')}</label
					>
					<select
						bind:value={form.reimbursement_account_id}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					>
						<option value={null}>{$i18n.t('Use company default')}</option>
						{#each accounts.filter((a) => a.account_type === 'liability') as acct}
							<option value={acct.id}>{acct.code} — {acct.name}</option>
						{/each}
					</select>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Default Category')}</label
					>
					<select
						bind:value={form.default_expense_category_id}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					>
						<option value={null}>{$i18n.t('None')}</option>
						{#each categories as cat}
							<option value={cat.id}>{cat.label}</option>
						{/each}
					</select>
				</div>
			</div>
			<div class="flex items-center justify-between">
				<label class="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400">
					<input type="checkbox" bind:checked={form.is_active} class="rounded" />
					{$i18n.t('Active')}
				</label>
				<button
					class="px-4 py-1.5 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition"
					on:click={handleSubmit}
				>
					{editingId ? $i18n.t('Update') : $i18n.t('Create')}
				</button>
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="flex justify-center my-6"><Spinner className="size-5" /></div>
	{:else if employees.length === 0}
		<div class="text-sm text-gray-400 italic">
			{$i18n.t('No employees yet. Add one to start assigning expense receipts.')}
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-xs">
				<thead class="text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-800">
					<tr>
						<th class="text-left py-2 px-2">{$i18n.t('Code')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Name')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Email')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Department')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Reimbursement Account')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Default Category')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Status')}</th>
						<th class="text-right py-2 px-2">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each employees as emp}
						<tr class="border-b border-gray-100 dark:border-gray-850 hover:bg-gray-50 dark:hover:bg-gray-900/50">
							<td class="py-2 px-2 font-mono dark:text-gray-200">{emp.code}</td>
							<td class="py-2 px-2 dark:text-gray-200">{emp.full_name}</td>
							<td class="py-2 px-2 text-gray-500">{emp.email ?? ''}</td>
							<td class="py-2 px-2 text-gray-500">{emp.department ?? ''}</td>
							<td class="py-2 px-2 text-gray-500"
								>{accountLabel(emp.reimbursement_account_id)}</td
							>
							<td class="py-2 px-2 text-gray-500"
								>{categoryLabel(emp.default_expense_category_id)}</td
							>
							<td class="py-2 px-2">
								{#if emp.is_active}
									<span
										class="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400"
										>{$i18n.t('Active')}</span
									>
								{:else}
									<span
										class="text-[10px] px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400"
										>{$i18n.t('Inactive')}</span
									>
								{/if}
							</td>
							<td class="py-2 px-2 text-right">
								<button
									class="text-xs text-blue-600 hover:text-blue-700 mr-2"
									on:click={() => openEdit(emp)}>{$i18n.t('Edit')}</button
								>
								{#if emp.is_active}
									<button
										class="text-xs text-red-500 hover:text-red-700"
										on:click={() => handleDeactivate(emp.id)}>{$i18n.t('Deactivate')}</button
									>
								{:else}
									<button
										class="text-xs text-emerald-600 hover:text-emerald-700"
										on:click={() => handleRestore(emp.id)}>{$i18n.t('Restore')}</button
									>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
