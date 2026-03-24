<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getFixedAssets,
		createFixedAsset,
		deleteFixedAsset,
		generateDepreciation,
		getAccounts,
		getPeriods
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	// State
	let loading = true;
	let assets: any[] = [];
	let accounts: any[] = [];

	// Create form
	let showAddForm = false;
	let creating = false;
	let newName = '';
	let newDescription = '';
	let newAssetAccountId: number | '' = '';
	let newDepreciationAccountId: number | '' = '';
	let newExpenseAccountId: number | '' = '';
	let newAcquisitionDate = '';
	let newValue = '';
	let newSalvageValue = '0';
	let newUsefulLife = '';
	let newMethod = 'linear';

	// Depreciation generation
	let generatingDep = false;
	let depMonth = '';
	let monthOptions: Array<{ value: string; label: string; to: string }> = [];

	// Delete confirmation
	let showDeleteConfirm = false;
	let deleteTarget: any = null;

	// ─── Helpers ────────────────────────────────────────────────────────────────

	const formatDate = (val: any) => {
		if (!val) return '-';
		return dayjs(val).format('YYYY-MM-DD');
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '0.00';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	function buildMonthOptions(periods: any[]) {
		const options: typeof monthOptions = [];
		for (const p of periods) {
			const start = new Date(p.start_date);
			const end = new Date(p.end_date);
			let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
			while (cursor <= end) {
				const y = cursor.getFullYear();
				const m = cursor.getMonth();
				const lastDay = new Date(y, m + 1, 0).getDate();
				const to = `${y}-${String(m + 1).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
				const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				options.push({ value: `${y}-${String(m + 1).padStart(2, '0')}`, label, to });
				cursor = new Date(y, m + 1, 1);
			}
		}
		const seen = new Map<string, (typeof options)[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
	}

	// Account filters
	$: assetAccounts = accounts.filter((a: any) => {
		const code = (a.code ?? '').replace(/\./g, '');
		return code.startsWith('2') && !code.startsWith('28');
	});

	$: depreciationAccounts = accounts.filter((a: any) => {
		const code = (a.code ?? '').replace(/\./g, '');
		return code.startsWith('28');
	});

	$: expenseAccounts = accounts.filter((a: any) => {
		const code = (a.code ?? '').replace(/\./g, '');
		return code.startsWith('681');
	});

	// ─── Data loading ───────────────────────────────────────────────────────────

	const loadAssets = async () => {
		loading = true;
		try {
			const res = await getFixedAssets(companyId);
			assets = Array.isArray(res) ? res : res?.items ?? [];
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load fixed assets')}: ${err}`);
		}
		loading = false;
	};

	onMount(async () => {
		try {
			const [, acctRes, periodRes] = await Promise.all([
				loadAssets(),
				getAccounts({ company_id: companyId }),
				getPeriods({ company_id: companyId })
			]);
			accounts = Array.isArray(acctRes) ? acctRes : acctRes?.items ?? acctRes?.accounts ?? [];
			const periods = periodRes.periods ?? periodRes ?? [];
			monthOptions = buildMonthOptions(periods);

			const now = new Date();
			const curKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
			const match = monthOptions.find((o) => o.value === curKey);
			if (match) {
				depMonth = match.value;
			} else if (monthOptions.length > 0) {
				depMonth = monthOptions[0].value;
			}
		} catch (err) {
			console.error('Failed to load accounts/periods:', err);
		}
	});

	// ─── Create ─────────────────────────────────────────────────────────────────

	const handleCreate = async () => {
		if (!newName || !newAssetAccountId || !newAcquisitionDate || !newValue || !newUsefulLife) {
			toast.error($i18n.t('Please fill in all required fields'));
			return;
		}
		const valueNum = parseFloat(newValue);
		const salvageNum = parseFloat(newSalvageValue) || 0;
		const lifeNum = parseInt(String(newUsefulLife), 10);
		if (isNaN(valueNum) || valueNum <= 0) {
			toast.error($i18n.t('Value must be a positive number'));
			return;
		}
		if (isNaN(lifeNum) || lifeNum <= 0) {
			toast.error($i18n.t('Useful life must be a positive number'));
			return;
		}

		creating = true;
		try {
			await createFixedAsset(companyId, {
				name: newName,
				description: newDescription,
				asset_account_id: Number(newAssetAccountId),
				depreciation_account_id: newDepreciationAccountId ? Number(newDepreciationAccountId) : undefined,
				expense_account_id: newExpenseAccountId ? Number(newExpenseAccountId) : undefined,
				acquisition_date: newAcquisitionDate,
				acquisition_value: valueNum,
				salvage_value: salvageNum,
				useful_life_months: lifeNum,
				method: newMethod
			});
			toast.success($i18n.t('Fixed asset created'));
			showAddForm = false;
			newName = '';
			newDescription = '';
			newAssetAccountId = '';
			newDepreciationAccountId = '';
			newExpenseAccountId = '';
			newAcquisitionDate = '';
			newValue = '';
			newSalvageValue = '0';
			newUsefulLife = '';
			newMethod = 'linear';
			await loadAssets();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create fixed asset') + ': ' + msg);
		}
		creating = false;
	};

	// ─── Delete ─────────────────────────────────────────────────────────────────

	const confirmDelete = (asset: any) => {
		deleteTarget = asset;
		showDeleteConfirm = true;
	};

	const handleDelete = async () => {
		if (!deleteTarget) return;
		try {
			await deleteFixedAsset(deleteTarget.id);
			toast.success($i18n.t('Fixed asset deleted'));
			await loadAssets();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to delete fixed asset') + ': ' + msg);
		}
		deleteTarget = null;
	};

	// ─── Generate Depreciation ──────────────────────────────────────────────────

	const handleGenerateDepreciation = async () => {
		const opt = monthOptions.find((o) => o.value === depMonth);
		if (!opt) {
			toast.error($i18n.t('Please select a month'));
			return;
		}
		generatingDep = true;
		try {
			const result = await generateDepreciation(companyId, opt.to);
			const count = result?.entries_created ?? result?.count ?? 0;
			toast.success(`${count} ${$i18n.t('depreciation entries created as Draft')}`);
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to generate depreciation') + ': ' + msg);
		}
		generatingDep = false;
	};
</script>

<ConfirmDialog
	bind:show={showDeleteConfirm}
	on:confirm={handleDelete}
	title={$i18n.t('Delete Fixed Asset')}
	message={$i18n.t('Are you sure you want to delete this fixed asset? This action cannot be undone.')}
/>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Fixed Assets')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{assets.length}
			</div>
		</div>

		<div class="flex gap-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition"
				on:click={() => {
					showAddForm = !showAddForm;
				}}
			>
				{showAddForm ? $i18n.t('Cancel') : $i18n.t('Add Asset')}
			</button>
		</div>
	</div>

	<!-- Description -->
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t('Manage fixed assets and generate monthly depreciation entries.')}
	</div>

	<!-- Add Asset Form -->
	{#if showAddForm}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30 mb-3"
		>
			<div class="text-sm font-medium dark:text-gray-200 mb-3">
				{$i18n.t('Add Fixed Asset')}
			</div>
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
				<div>
					<label
						for="asset-name"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Name')} *
					</label>
					<input
						id="asset-name"
						type="text"
						bind:value={newName}
						placeholder={$i18n.t('e.g. Office Computer')}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="asset-description"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Description')}
					</label>
					<input
						id="asset-description"
						type="text"
						bind:value={newDescription}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
				<div>
					<label
						for="asset-account"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Asset Account')} *
					</label>
					<select
						id="asset-account"
						bind:value={newAssetAccountId}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					>
						<option value="">{$i18n.t('Select account...')}</option>
						{#each assetAccounts as acct}
							<option value={acct.id}>{acct.code} - {acct.name}</option>
						{/each}
					</select>
				</div>
				<div>
					<label
						for="depreciation-account"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Depreciation Account')}
					</label>
					<select
						id="depreciation-account"
						bind:value={newDepreciationAccountId}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					>
						<option value="">{$i18n.t('Select account...')}</option>
						{#each depreciationAccounts as acct}
							<option value={acct.id}>{acct.code} - {acct.name}</option>
						{/each}
					</select>
				</div>
				<div>
					<label
						for="expense-account"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Expense Account')}
					</label>
					<select
						id="expense-account"
						bind:value={newExpenseAccountId}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					>
						<option value="">{$i18n.t('Select account...')}</option>
						{#each expenseAccounts as acct}
							<option value={acct.id}>{acct.code} - {acct.name}</option>
						{/each}
					</select>
				</div>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-5 gap-3 items-end">
				<div>
					<label
						for="asset-acquisition-date"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Acquisition Date')} *
					</label>
					<input
						id="asset-acquisition-date"
						type="date"
						bind:value={newAcquisitionDate}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="asset-value"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Value')} *
					</label>
					<input
						id="asset-value"
						type="number"
						step="0.01"
						min="0"
						bind:value={newValue}
						placeholder="10000.00"
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="asset-salvage"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Salvage Value')}
					</label>
					<input
						id="asset-salvage"
						type="number"
						step="0.01"
						min="0"
						bind:value={newSalvageValue}
						placeholder="0"
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="asset-useful-life"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Useful Life (months)')} *
					</label>
					<input
						id="asset-useful-life"
						type="number"
						step="1"
						min="1"
						bind:value={newUsefulLife}
						placeholder="60"
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="asset-method"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Method')}
					</label>
					<select
						id="asset-method"
						bind:value={newMethod}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					>
						<option value="linear">{$i18n.t('Linear')}</option>
						<option value="declining">{$i18n.t('Declining')}</option>
					</select>
				</div>
			</div>
			<div class="mt-3 flex justify-end">
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
					on:click={handleCreate}
					disabled={creating || !newName || !newAssetAccountId || !newAcquisitionDate || !newValue || !newUsefulLife}
				>
					{creating ? $i18n.t('Saving...') : $i18n.t('Save')}
				</button>
			</div>
		</div>
	{/if}

	<!-- Generate Monthly Depreciation -->
	<div
		class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30 mb-3"
	>
		<div class="text-sm font-medium dark:text-gray-200 mb-2">
			{$i18n.t('Generate Monthly Depreciation')}
		</div>
		<div class="flex flex-wrap gap-3 items-end">
			<div>
				<label
					for="dep-month"
					class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>
					{$i18n.t('Month')}
				</label>
				{#if monthOptions.length > 0}
					<select
						id="dep-month"
						bind:value={depMonth}
						class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
					>
						{#each monthOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				{:else}
					<span class="text-xs text-gray-400 italic">
						{$i18n.t('No accounting periods defined')}
					</span>
				{/if}
			</div>
			<button
				class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
				disabled={!depMonth || generatingDep}
				on:click={handleGenerateDepreciation}
			>
				{$i18n.t('Generate')}
			</button>
		</div>

		<!-- AI Loading Spinner -->
		{#if generatingDep}
			<div
				class="relative overflow-hidden rounded-xl border border-blue-200/50 dark:border-blue-800/30 bg-blue-50 dark:bg-blue-900/20 p-4 mt-3"
			>
				<div
					class="absolute top-0 left-0 h-1 bg-blue-500 animate-pulse"
					style="width: 100%;"
				/>
				<div class="flex items-center gap-3">
					<Spinner className="size-5 text-blue-600 dark:text-blue-400" />
					<span class="text-sm font-medium text-blue-700 dark:text-blue-300">
						{$i18n.t('Generating depreciation entries...')}
					</span>
				</div>
			</div>
		{/if}
	</div>

	<!-- Assets Table -->
	{#if loading}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if assets.length === 0}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-8 border border-gray-100/30 dark:border-gray-850/30 text-center"
		>
			<div class="text-gray-400 dark:text-gray-500 text-sm mb-3">
				{$i18n.t('No fixed assets registered')}
			</div>
			<div class="text-gray-400 dark:text-gray-500 text-xs">
				{$i18n.t('Add fixed assets to track depreciation and generate monthly entries.')}
			</div>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead
					class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
				>
					<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
						<th class="px-3 py-2">{$i18n.t('Name')}</th>
						<th class="px-3 py-2">{$i18n.t('Acquisition Date')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Value')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Useful Life')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Monthly Dep.')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Accumulated')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Book Value')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each assets as asset (asset.id)}
						{@const acqValue = parseFloat(asset.acquisition_value ?? asset.value ?? 0)}
						{@const salvage = parseFloat(asset.salvage_value ?? 0)}
						{@const lifeMonths = asset.useful_life_months ?? asset.useful_life ?? 1}
						{@const monthlyDep = lifeMonths > 0 ? (acqValue - salvage) / lifeMonths : 0}
						{@const accumulated = parseFloat(asset.accumulated_depreciation ?? 0)}
						{@const bookValue = asset.book_value !== undefined ? parseFloat(asset.book_value) : acqValue - accumulated}
						<tr
							class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
						>
							<td class="px-3 py-2 font-medium dark:text-gray-200">
								{asset.name}
								{#if asset.description}
									<div class="text-[10px] text-gray-400 dark:text-gray-500">
										{asset.description}
									</div>
								{/if}
							</td>
							<td class="px-3 py-2">
								{formatDate(asset.acquisition_date)}
							</td>
							<td class="px-3 py-2 text-right font-mono">
								{fmt(acqValue)}
							</td>
							<td class="px-3 py-2 text-right">
								{lifeMonths} {$i18n.t('mo.')}
							</td>
							<td class="px-3 py-2 text-right font-mono">
								{fmt(monthlyDep)}
							</td>
							<td class="px-3 py-2 text-right font-mono text-red-600 dark:text-red-400">
								{fmt(accumulated)}
							</td>
							<td class="px-3 py-2 text-right font-mono font-medium">
								{fmt(bookValue)}
							</td>
							<td class="px-3 py-2 text-right">
								<Tooltip content={$i18n.t('Delete this fixed asset')}>
									<button
										class="px-3 py-1 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 dark:hover:bg-red-900/40 transition"
										on:click={() => confirmDelete(asset)}
									>
										{$i18n.t('Delete')}
									</button>
								</Tooltip>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
