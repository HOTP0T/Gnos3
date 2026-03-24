<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getCategorizationRules, createCategorizationRule, deleteCategorizationRule, getAccounts } from '$lib/apis/accounting';

	const i18n = getContext('i18n');
	export let companyId: number;

	let rules: any[] = [];
	let accounts: any[] = [];
	let loading = true;
	let showForm = false;

	let newVendor = '';
	let newAccountCode = '';
	let newMatchType = 'contains';
	let newPriority = 0;
	let newDescKeywords = '';

	const load = async () => {
		loading = true;
		try {
			const [rulesData, accountsData] = await Promise.all([
				getCategorizationRules(companyId),
				getAccounts({ company_id: companyId })
			]);
			rules = rulesData ?? [];
			const accts = accountsData?.accounts ?? accountsData ?? [];
			accounts = Array.isArray(accts) ? accts : [];
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	onMount(load);

	const handleCreate = async () => {
		if (!newVendor || !newAccountCode) { toast.error($i18n.t('Vendor pattern and account are required')); return; }
		try {
			await createCategorizationRule(companyId, {
				vendor_name_pattern: newVendor,
				account_code: newAccountCode,
				match_type: newMatchType,
				priority: newPriority,
				description_keywords: newDescKeywords || null,
			});
			toast.success($i18n.t('Rule created'));
			newVendor = ''; newAccountCode = ''; newDescKeywords = ''; showForm = false;
			await load();
		} catch (err) { toast.error(`${err}`); }
	};

	const handleDelete = async (id: number) => {
		try {
			await deleteCategorizationRule(id);
			toast.success($i18n.t('Rule removed'));
			await load();
		} catch (err) { toast.error(`${err}`); }
	};

	const accountName = (code: string) => {
		const acct = accounts.find((a: any) => a.code === code);
		return acct ? `${code} — ${acct.name}` : code;
	};
</script>

<div class="space-y-3">
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-semibold dark:text-gray-200">{$i18n.t('Account Categorization Rules')}</h3>
		<button
			class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
			on:click={() => (showForm = !showForm)}
		>{showForm ? $i18n.t('Cancel') : $i18n.t('Add Rule')}</button>
	</div>

	<p class="text-xs text-gray-500 dark:text-gray-400">
		{$i18n.t('Rules map vendor names to accounts. The AI learns from your corrections automatically.')}
	</p>

	{#if showForm}
		<div class="p-3 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-800 space-y-2">
			<div class="grid grid-cols-2 gap-2">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Vendor Pattern')}</label>
					<input type="text" bind:value={newVendor} placeholder="e.g. amazon" class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Account')}</label>
					<select bind:value={newAccountCode} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="">—</option>
						{#each accounts as acct}
							<option value={acct.code}>{acct.code} — {acct.name}</option>
						{/each}
					</select>
				</div>
			</div>
			<div class="flex gap-2 items-end">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Match Type')}</label>
					<select bind:value={newMatchType} class="text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="exact">{$i18n.t('Exact')}</option>
						<option value="contains">{$i18n.t('Contains')}</option>
						<option value="prefix">{$i18n.t('Prefix')}</option>
						<option value="regex">{$i18n.t('Regex')}</option>
					</select>
				</div>
				<div class="flex-1">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Description Keywords</label>
					<input type="text" bind:value={newDescKeywords} placeholder="e.g. office,supplies" class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Priority')}</label>
					<input type="number" bind:value={newPriority} class="w-20 text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<button class="px-4 py-1.5 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition" on:click={handleCreate}>{$i18n.t('Save')}</button>
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="text-sm text-gray-400">{$i18n.t('Loading...')}</div>
	{:else if rules.length === 0}
		<div class="text-sm text-gray-400 italic">{$i18n.t('No rules yet. Rules are created automatically when you confirm invoice categories.')}</div>
	{:else}
		<div class="overflow-x-auto rounded-xl border border-gray-100/30 dark:border-gray-850/30">
			<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
				<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
					<tr>
						<th class="px-3 py-2">{$i18n.t('Vendor Pattern')}</th>
						<th class="px-2 py-2">{$i18n.t('Account')}</th>
						<th class="px-2 py-2 text-center">{$i18n.t('Match')}</th>
						<th class="px-2 py-2 text-center">{$i18n.t('Source')}</th>
						<th class="px-2 py-2 text-center">{$i18n.t('Used')}</th>
						<th class="px-2 py-2 w-16"></th>
					</tr>
				</thead>
				<tbody>
					{#each rules as rule}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
							<td class="px-3 py-1.5 font-medium">{rule.vendor_name_pattern}</td>
							<td class="px-2 py-1.5">{accountName(rule.account_code)}</td>
							<td class="px-2 py-1.5 text-center">
								<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.match_type === 'exact' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : rule.match_type === 'contains' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'}">
									{rule.match_type}
								</span>
							</td>
							<td class="px-2 py-1.5 text-center">
								<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.created_by === 'human' ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}">
									{rule.created_by === 'human' ? $i18n.t('Manual') : $i18n.t('AI')}
								</span>
							</td>
							<td class="px-2 py-1.5 text-center text-gray-400">{rule.times_used}x</td>
							<td class="px-2 py-1.5 text-center">
								<button class="text-red-500 hover:text-red-700 text-xs" on:click={() => handleDelete(rule.id)}>{$i18n.t('Remove')}</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
