<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import {
		getExpenseCategories,
		createExpenseCategory,
		updateExpenseCategory,
		deleteExpenseCategory,
		instantiateDefaultExpenseCategories,
		getAccounts
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let categories: any[] = [];
	let accounts: any[] = [];
	let loading = true;
	let showForm = false;
	let editingId: number | null = null;
	let editingIsBuiltin = false;

	const emptyForm = () => ({
		key: '',
		label: '',
		description: '',
		default_account_id: null as number | null,
		sort_order: 0,
		is_active: true
	});

	let form = emptyForm();

	const load = async () => {
		loading = true;
		try {
			const [catData, acctData] = await Promise.all([
				getExpenseCategories(companyId),
				getAccounts({ company_id: companyId })
			]);
			categories = catData?.categories ?? [];
			const accts = acctData?.accounts ?? acctData ?? [];
			accounts = Array.isArray(accts) ? accts : [];
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		loading = false;
	};

	onMount(load);

	const resetForm = () => {
		form = emptyForm();
		editingId = null;
		editingIsBuiltin = false;
		showForm = false;
	};

	const openNew = () => {
		form = emptyForm();
		editingId = null;
		editingIsBuiltin = false;
		showForm = true;
	};

	const openEdit = (cat: any) => {
		form = {
			key: cat.key,
			label: cat.label,
			description: cat.description ?? '',
			default_account_id: cat.default_account_id,
			sort_order: cat.sort_order,
			is_active: cat.is_active
		};
		editingId = cat.id;
		editingIsBuiltin = cat.is_builtin;
		showForm = true;
	};

	const handleSubmit = async () => {
		if (!form.label.trim()) {
			toast.error($i18n.t('Label is required'));
			return;
		}
		if (!editingId && !form.key.trim()) {
			toast.error($i18n.t('Key is required'));
			return;
		}
		try {
			if (editingId) {
				await updateExpenseCategory(editingId, {
					label: form.label.trim(),
					description: form.description.trim() || null,
					default_account_id: form.default_account_id ?? null,
					sort_order: form.sort_order,
					is_active: form.is_active
				});
				toast.success($i18n.t('Category updated'));
			} else {
				await createExpenseCategory(companyId, {
					key: form.key.trim().toLowerCase(),
					label: form.label.trim(),
					description: form.description.trim() || null,
					default_account_id: form.default_account_id ?? null,
					sort_order: form.sort_order,
					is_active: form.is_active
				});
				toast.success($i18n.t('Category created'));
			}
			resetForm();
			await load();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleDelete = async (cat: any) => {
		if (cat.is_builtin) {
			toast.error($i18n.t('Built-in categories cannot be deleted — deactivate instead'));
			return;
		}
		try {
			await deleteExpenseCategory(cat.id);
			toast.success($i18n.t('Category deleted'));
			await load();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleInstantiateDefaults = async () => {
		try {
			await instantiateDefaultExpenseCategories(companyId);
			toast.success($i18n.t('Default categories restored'));
			await load();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const accountLabel = (id: number | null) => {
		if (!id) return '';
		const a = accounts.find((x) => x.id === id);
		return a ? `${a.code} — ${a.name}` : '';
	};
</script>

<div class="space-y-3">
	<div class="flex items-center justify-between flex-wrap gap-2">
		<h3 class="text-sm font-semibold dark:text-gray-200">{$i18n.t('Expense Categories')}</h3>
		<div class="flex items-center gap-2">
			<button
				class="px-3 py-1 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
				on:click={handleInstantiateDefaults}
				title={$i18n.t('Re-seed any missing built-in categories')}
			>
				{$i18n.t('Restore Defaults')}
			</button>
			<button
				class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
				on:click={showForm ? resetForm : openNew}
			>
				{showForm ? $i18n.t('Cancel') : $i18n.t('New Category')}
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
						>{$i18n.t('Key')}</label
					>
					<input
						type="text"
						bind:value={form.key}
						placeholder="accommodation"
						disabled={editingId !== null}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden disabled:opacity-60"
					/>
				</div>
				<div class="md:col-span-2">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Label')}</label
					>
					<input
						type="text"
						bind:value={form.label}
						placeholder="Accommodation"
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Sort Order')}</label
					>
					<input
						type="number"
						bind:value={form.sort_order}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div class="md:col-span-3">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Description')}</label
					>
					<input
						type="text"
						bind:value={form.description}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Default Expense Account')}</label
					>
					<select
						bind:value={form.default_account_id}
						class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					>
						<option value={null}>{$i18n.t('None')}</option>
						{#each accounts.filter((a) => a.account_type === 'expense') as acct}
							<option value={acct.id}>{acct.code} — {acct.name}</option>
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
	{:else if categories.length === 0}
		<div class="text-sm text-gray-400 italic">
			{$i18n.t('No expense categories. Click Restore Defaults to seed the built-in taxonomy.')}
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-xs">
				<thead class="text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-800">
					<tr>
						<th class="text-left py-2 px-2">{$i18n.t('Key')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Label')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Default Account')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Description')}</th>
						<th class="text-left py-2 px-2">{$i18n.t('Status')}</th>
						<th class="text-right py-2 px-2">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each categories as cat}
						<tr class="border-b border-gray-100 dark:border-gray-850 hover:bg-gray-50 dark:hover:bg-gray-900/50">
							<td class="py-2 px-2 font-mono dark:text-gray-200">
								{cat.key}
								{#if cat.is_builtin}
									<span
										class="ml-1 text-[10px] px-1 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400"
										>{$i18n.t('built-in')}</span
									>
								{/if}
							</td>
							<td class="py-2 px-2 dark:text-gray-200">{cat.label}</td>
							<td class="py-2 px-2 text-gray-500">{accountLabel(cat.default_account_id)}</td>
							<td class="py-2 px-2 text-gray-500">{cat.description ?? ''}</td>
							<td class="py-2 px-2">
								{#if cat.is_active}
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
									on:click={() => openEdit(cat)}>{$i18n.t('Edit')}</button
								>
								{#if !cat.is_builtin}
									<button
										class="text-xs text-red-500 hover:text-red-700"
										on:click={() => handleDelete(cat)}>{$i18n.t('Delete')}</button
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
