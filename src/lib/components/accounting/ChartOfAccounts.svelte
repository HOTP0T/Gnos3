<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getAccounts, deleteAccount, getOpeningBalances, updateOpeningBalances, updateCompany } from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import Badge from '$lib/components/common/Badge.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import AccountFormModal from '$lib/components/accounting/AccountFormModal.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	// Data
	let accounts: any[] = [];
	let openingBalances: Record<number, { debit: number; credit: number }> = {};
	let editedBalances: Record<number, { debit: number; credit: number }> = {};
	let savingBalances = false;
	let loading = true;

	// Opening balance date
	let openingBalanceDate = '';
	let originalObDate = '';

	$: isDateDirty = openingBalanceDate !== originalObDate;
	$: isDirty = Object.keys(editedBalances).length > 0 || isDateDirty;

	// Modal state
	let showFormModal = false;
	let editingAccount: any = null;

	// Delete confirmation
	let showDeleteConfirm = false;
	let deleteTarget: any = null;

	// Collapsible sections
	let collapsedSections: Set<string> = new Set();

	const ACCOUNT_TYPES = [
		{ key: 'asset', label: 'Assets', badgeType: 'info' },
		{ key: 'liability', label: 'Liabilities', badgeType: 'error' },
		{ key: 'equity', label: 'Equity', badgeType: 'purple' },
		{ key: 'revenue', label: 'Revenue', badgeType: 'success' },
		{ key: 'expense', label: 'Expenses', badgeType: 'warning' }
	] as const;

	const TYPE_BADGE_CLASSES: Record<string, string> = {
		asset: 'bg-blue-500/20 text-blue-700 dark:text-blue-200',
		liability: 'bg-red-500/20 text-red-700 dark:text-red-200',
		equity: 'bg-purple-500/20 text-purple-700 dark:text-purple-200',
		revenue: 'bg-green-500/20 text-green-700 dark:text-green-200',
		expense: 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-200'
	};

	$: groupedAccounts = ACCOUNT_TYPES.map((type) => ({
		...type,
		accounts: accounts
			.filter((a) => a.account_type === type.key)
			.sort((a, b) => a.code.localeCompare(b.code))
	}));

	const loadAccounts = async () => {
		loading = true;
		try {
			const res = await getAccounts({ company_id: companyId });
			if (res) {
				accounts = Array.isArray(res) ? res : res.accounts ?? [];
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const loadOpeningBalances = async () => {
		try {
			const res = await getOpeningBalances(companyId);
			const obj: Record<number, { debit: number; credit: number }> = {};
			if (res?.balances) {
				for (const [idStr, val] of Object.entries(res.balances)) {
					const v = val as { debit: number; credit: number };
					obj[Number(idStr)] = { debit: v.debit ?? 0, credit: v.credit ?? 0 };
				}
			}
			openingBalances = obj;
			if (res?.opening_balance_date) {
				openingBalanceDate = res.opening_balance_date;
				originalObDate = res.opening_balance_date;
			}
		} catch (err) {
			// Opening balances are non-critical, silently fail
		}
	};

	const getBalanceForAccount = (accountId: number): { debit: number; credit: number } => {
		if (editedBalances[accountId]) {
			return editedBalances[accountId];
		}
		return openingBalances[accountId] ?? { debit: 0, credit: 0 };
	};

	const handleBalanceInput = (accountId: number, field: 'debit' | 'credit', value: string) => {
		const current = getBalanceForAccount(accountId);
		const numVal = parseFloat(value) || 0;
		const updated = { ...current, [field]: numVal };
		editedBalances = { ...editedBalances, [accountId]: updated };
	};

	const saveOpeningBalances = async () => {
		if (Object.keys(editedBalances).length === 0 && !isDateDirty) return;
		savingBalances = true;
		try {
			// Persist OB date change first
			if (isDateDirty && openingBalanceDate) {
				await updateCompany(companyId, { opening_balance_date: openingBalanceDate });
				originalObDate = openingBalanceDate;
			}
			// Save balance entries
			if (Object.keys(editedBalances).length > 0) {
				const entries = Object.entries(editedBalances).map(([accountId, bal]) => ({
					account_id: Number(accountId),
					debit: bal.debit,
					credit: bal.credit
				}));
				await updateOpeningBalances(companyId, entries);
				openingBalances = { ...openingBalances, ...editedBalances };
				editedBalances = {};
			}
			toast.success($i18n.t('Opening balances saved'));
		} catch (err: any) {
			toast.error(err?.detail || `${err}`);
		}
		savingBalances = false;
	};

	const toggleSection = (key: string) => {
		if (collapsedSections.has(key)) {
			collapsedSections.delete(key);
		} else {
			collapsedSections.add(key);
		}
		collapsedSections = collapsedSections;
	};

	const openAddModal = () => {
		editingAccount = null;
		showFormModal = true;
	};

	const openEditModal = (account: any) => {
		editingAccount = account;
		showFormModal = true;
	};

	const confirmDeactivate = (e: MouseEvent, account: any) => {
		e.stopPropagation();
		deleteTarget = account;
		showDeleteConfirm = true;
	};

	const handleDeactivate = async () => {
		if (!deleteTarget) return;
		try {
			await deleteAccount(deleteTarget.id);
			toast.success($i18n.t('Account deactivated'));
			await loadAccounts();
			await loadOpeningBalances();
		} catch (err: any) {
			toast.error(err?.detail || `${err}`);
		}
		deleteTarget = null;
	};

	const handleSave = async () => {
		await loadAccounts();
		await loadOpeningBalances();
	};

	onMount(async () => {
		await loadAccounts();
		await loadOpeningBalances();
	});
</script>

<ConfirmDialog
	bind:show={showDeleteConfirm}
	on:confirm={handleDeactivate}
	title={$i18n.t('Deactivate Account')}
	message={$i18n.t(
		'Are you sure you want to deactivate this account? It will no longer appear in active listings.'
	)}
/>

<AccountFormModal
	bind:show={showFormModal}
	account={editingAccount}
	{accounts}
	{companyId}
	on:save={handleSave}
/>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Chart of Accounts')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{accounts.length}
			</div>
		</div>

		<div class="flex gap-1 items-center">
			<!-- OB Date Picker -->
			<div class="flex items-center gap-1.5 mr-1">
				<label class="text-xs text-gray-500 dark:text-gray-400 whitespace-nowrap">
					{$i18n.t('OB Date')}
				</label>
				<input
					type="date"
					bind:value={openingBalanceDate}
					class="text-xs font-mono px-2 py-1.5 rounded-lg border bg-transparent dark:text-gray-200
						{isDateDirty ? 'border-emerald-400 dark:border-emerald-600' : 'border-gray-200 dark:border-gray-700'}
						focus:outline-none focus:ring-1 focus:ring-emerald-400 dark:focus:ring-emerald-600"
				/>
			</div>

			{#if isDirty}
				<button
					class="px-3.5 py-1.5 text-sm rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-medium transition flex items-center gap-1.5 disabled:opacity-50"
					on:click={saveOpeningBalances}
					disabled={savingBalances}
				>
					{#if savingBalances}
						<Spinner className="size-3.5" />
					{:else}
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							stroke-width="2"
							stroke="currentColor"
							class="size-4"
						>
							<path stroke-linecap="round" stroke-linejoin="round" d="m4.5 12.75 6 6 9-13.5" />
						</svg>
					{/if}
					{$i18n.t('Save Opening Balances')}
				</button>
			{/if}
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-gray-900 hover:bg-gray-850 text-white dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium transition flex items-center gap-1.5"
				on:click={openAddModal}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4"
				>
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
				</svg>
				{$i18n.t('Add Account')}
			</button>
		</div>
	</div>

	<!-- Content -->
	{#if loading && accounts.length === 0}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if accounts.length === 0}
		<div class="flex justify-center my-10 text-sm text-gray-500">
			{$i18n.t('No accounts found')}
		</div>
	{:else}
		<div class="space-y-3 mt-2">
			{#each groupedAccounts as group}
				{#if group.accounts.length > 0}
					<div class="rounded-xl border border-gray-100 dark:border-gray-850 overflow-hidden">
						<!-- Section Header -->
						<!-- svelte-ignore a11y-click-events-have-key-events -->
						<!-- svelte-ignore a11y-no-static-element-interactions -->
						<div
							class="flex items-center justify-between px-4 py-2.5 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
							on:click={() => toggleSection(group.key)}
						>
							<div class="flex items-center gap-2">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									stroke-width="2"
									stroke="currentColor"
									class="size-3.5 transition-transform {collapsedSections.has(group.key)
										? '-rotate-90'
										: ''}"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										d="m19.5 8.25-7.5 7.5-7.5-7.5"
									/>
								</svg>
								<span class="text-sm font-semibold dark:text-gray-200">
									{$i18n.t(group.label)}
								</span>
								<span class="text-xs text-gray-400">
									({group.accounts.length})
								</span>
							</div>
						</div>

						<!-- Section Table -->
						{#if !collapsedSections.has(group.key)}
							<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
								<thead
									class="text-xs text-gray-500 dark:text-gray-400 uppercase bg-gray-50/50 dark:bg-gray-850/30"
								>
									<tr class="border-b border-gray-100 dark:border-gray-850">
										<th scope="col" class="px-4 py-2 w-28">{$i18n.t('Code')}</th>
										<th scope="col" class="px-4 py-2">{$i18n.t('Name')}</th>
										<th scope="col" class="px-4 py-2 w-28">{$i18n.t('Type')}</th>
										<th scope="col" class="px-4 py-2 w-28 text-right"
											>{$i18n.t('Opening DR')}</th
										>
										<th scope="col" class="px-4 py-2 w-28 text-right"
											>{$i18n.t('Opening CR')}</th
										>
										<th scope="col" class="px-4 py-2 w-24">{$i18n.t('Status')}</th>
										<th scope="col" class="px-4 py-2 w-20 text-right"
											>{$i18n.t('Actions')}</th
										>
									</tr>
								</thead>
								<tbody>
									{#each group.accounts as acct (acct.id)}
										<!-- svelte-ignore a11y-click-events-have-key-events -->
										<!-- svelte-ignore a11y-no-static-element-interactions -->
										<tr
											class="bg-white dark:bg-gray-900 border-b border-gray-50 dark:border-gray-850/50 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition cursor-pointer"
											on:click={() => openEditModal(acct)}
										>
											<!-- Code -->
											<td class="px-4 py-2 font-mono" style="padding-left: {1 + acct.level * 1.5}rem">
												{acct.code}
											</td>

											<!-- Name -->
											<td class="px-4 py-2" style="padding-left: {1 + acct.level * 1.5}rem">
												<span class="dark:text-gray-200">{acct.name}</span>
												{#if acct.description}
													<span class="text-gray-400 ml-1 text-xs">- {acct.description}</span>
												{/if}
											</td>

											<!-- Type Badge -->
											<td class="px-4 py-2">
												<span
													class="text-xs font-medium px-[5px] rounded-lg uppercase whitespace-nowrap {TYPE_BADGE_CLASSES[
														acct.account_type
													] ?? ''}"
												>
													{acct.account_type}
												</span>
											</td>

											<!-- Opening DR -->
											<td class="px-4 py-1.5 text-right">
												<input
													type="number"
													step="0.01"
													class="w-24 text-right text-xs font-mono px-2 py-1 rounded-lg border bg-transparent
														{editedBalances[acct.id] ? 'border-emerald-400 dark:border-emerald-600' : 'border-gray-200 dark:border-gray-700'}
														focus:outline-none focus:ring-1 focus:ring-emerald-400 dark:focus:ring-emerald-600
														dark:text-gray-200"
													value={(editedBalances[acct.id] ?? openingBalances[acct.id] ?? {debit:0}).debit || ''}
													on:input={(e) => handleBalanceInput(acct.id, 'debit', e.currentTarget.value)}
													on:click|stopPropagation
												/>
											</td>

											<!-- Opening CR -->
											<td class="px-4 py-1.5 text-right">
												<input
													type="number"
													step="0.01"
													class="w-24 text-right text-xs font-mono px-2 py-1 rounded-lg border bg-transparent
														{editedBalances[acct.id] ? 'border-emerald-400 dark:border-emerald-600' : 'border-gray-200 dark:border-gray-700'}
														focus:outline-none focus:ring-1 focus:ring-emerald-400 dark:focus:ring-emerald-600
														dark:text-gray-200"
													value={(editedBalances[acct.id] ?? openingBalances[acct.id] ?? {credit:0}).credit || ''}
													on:input={(e) => handleBalanceInput(acct.id, 'credit', e.currentTarget.value)}
													on:click|stopPropagation
												/>
											</td>

											<!-- Status -->
											<td class="px-4 py-2">
												<Badge
													type={acct.is_active ? 'success' : 'muted'}
													content={acct.is_active ? $i18n.t('Active') : $i18n.t('Inactive')}
												/>
											</td>

											<!-- Actions -->
											<td class="px-4 py-2 text-right">
												{#if acct.is_active}
													<button
														class="p-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition text-red-500"
														on:click={(e) => confirmDeactivate(e, acct)}
														title={$i18n.t('Deactivate')}
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
												{/if}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						{/if}
					</div>
				{/if}
			{/each}
		</div>
	{/if}
</div>
