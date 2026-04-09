<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getCategorizationRules, createCategorizationRule, updateCategorizationRule, approveCategorizationRule, rejectCategorizationRule, deleteCategorizationRule, applyCategorizationRule, getAccounts } from '$lib/apis/accounting';

	const i18n = getContext('i18n');
	export let companyId: number;

	let rules: any[] = [];
	let accounts: any[] = [];
	let loading = true;
	let showForm = false;

	let newVendor = '';
	let newAccountCode = '';
	let newCounterpartyCode = '';
	let newMatchType = 'contains';
	let newMatchField = 'vendor';
	let newPriority = 0;
	let newDescKeywords = '';
	let newAutoCreateEntry = true;
	let newTaxAccountCode = '';
	let newTaxRate = '';
	let newDefaultDescription = '';
	let newRuleType = 'invoice';

	$: pendingRules = rules.filter((r: any) => r.status === 'pending');
	$: invoiceRules = rules.filter((r: any) => (r.rule_type || 'invoice') === 'invoice' && r.status !== 'pending');
	$: bankFeeRules = rules.filter((r: any) => r.rule_type === 'bank_fee' && r.status !== 'pending');

	// Edit mode
	let editingRule: any = null;

	const startEdit = (rule: any) => {
		editingRule = { ...rule };
		newVendor = rule.vendor_name_pattern;
		newAccountCode = rule.account_code;
		newCounterpartyCode = rule.counterparty_account_code || '';
		newMatchType = rule.match_type;
		newMatchField = rule.match_field || 'vendor';
		newPriority = rule.priority;
		newDescKeywords = (rule.description_keywords || []).join(', ');
		newAutoCreateEntry = rule.auto_create_entry;
		newTaxAccountCode = rule.tax_account_code || '';
		newTaxRate = rule.tax_rate ? String(rule.tax_rate) : '';
		newDefaultDescription = rule.default_description || '';
		newRuleType = rule.rule_type || 'invoice';
		showForm = true;
	};

	const handleSaveEdit = async () => {
		if (!editingRule || !newVendor || !newAccountCode) { toast.error($i18n.t('Name pattern and account are required')); return; }
		try {
			await updateCategorizationRule(editingRule.id, {
				vendor_name_pattern: newVendor,
				account_code: newAccountCode,
				counterparty_account_code: newCounterpartyCode || null,
				match_field: newRuleType === 'bank_fee' ? 'vendor' : newMatchField,
				auto_create_entry: newAutoCreateEntry,
				tax_account_code: newTaxAccountCode || null,
				tax_rate: newTaxRate ? parseFloat(newTaxRate) : null,
				default_description: newDefaultDescription || null,
				match_type: newMatchType,
				priority: newPriority,
				description_keywords: newDescKeywords ? newDescKeywords.split(',').map((k: string) => k.trim()).filter(Boolean) : null,
				rule_type: newRuleType,
			});
			toast.success($i18n.t('Rule updated'));
			editingRule = null;
			showForm = false;
			newVendor = ''; newAccountCode = ''; newCounterpartyCode = ''; newDescKeywords = '';
			newTaxAccountCode = ''; newTaxRate = ''; newDefaultDescription = '';
			newAutoCreateEntry = true; newRuleType = 'invoice';
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	const cancelEdit = () => {
		editingRule = null;
		showForm = false;
		newVendor = ''; newAccountCode = ''; newCounterpartyCode = ''; newDescKeywords = '';
		newTaxAccountCode = ''; newTaxRate = ''; newDefaultDescription = '';
		newAutoCreateEntry = true; newRuleType = 'invoice';
	};

	const handleApprove = async (id: number) => {
		try {
			await approveCategorizationRule(id);
			toast.success($i18n.t('Rule approved'));
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	const handleReject = async (id: number) => {
		try {
			await rejectCategorizationRule(id);
			toast.success($i18n.t('Rule rejected'));
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

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
		if (!newVendor || !newAccountCode) { toast.error($i18n.t('Name pattern and account are required')); return; }
		try {
			await createCategorizationRule(companyId, {
				vendor_name_pattern: newVendor,
				account_code: newAccountCode,
				counterparty_account_code: newCounterpartyCode || null,
				match_field: newRuleType === 'bank_fee' ? 'vendor' : newMatchField,
				auto_create_entry: newAutoCreateEntry,
				tax_account_code: newTaxAccountCode || null,
				tax_rate: newTaxRate ? parseFloat(newTaxRate) : null,
				default_description: newDefaultDescription || null,
				match_type: newMatchType,
				priority: newPriority,
				description_keywords: newDescKeywords ? newDescKeywords.split(',').map((k: string) => k.trim()).filter(Boolean) : null,
				rule_type: newRuleType,
			});
			toast.success($i18n.t('Rule created'));
			newVendor = ''; newAccountCode = ''; newCounterpartyCode = ''; newDescKeywords = '';
			newTaxAccountCode = ''; newTaxRate = ''; newDefaultDescription = '';
			newAutoCreateEntry = true; newRuleType = 'invoice'; showForm = false;
			await load();
		} catch (err) { toast.error(`${err}`); }
	};

	const handleDelete = async (id: number) => {
		try {
			await deleteCategorizationRule(id);
			toast.success($i18n.t('Rule removed'));
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	let applyingRuleId: number | null = null;
	const handleApply = async (rule: any) => {
		applyingRuleId = rule.id;
		try {
			const res = await applyCategorizationRule(rule.id);
			const label = res.rule_type === 'bank_fee' ? $i18n.t('bank lines') : $i18n.t('invoices');
			toast.success(`${$i18n.t('Applied to')} ${res.applied} ${label}`);
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		applyingRuleId = null;
	};

	const accountName = (code: string) => {
		const acct = accounts.find((a: any) => a.code === code);
		return acct ? `${code} — ${acct.name}` : code;
	};
</script>

<div class="space-y-3">
	<div class="flex items-center justify-between">
		<div class="flex items-center gap-2">
			<h3 class="text-sm font-semibold dark:text-gray-200">{$i18n.t('Account Categorization Rules')}</h3>
			{#if pendingRules.length > 0}
				<span class="px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400">{pendingRules.length} {$i18n.t('pending')}</span>
			{/if}
		</div>
		<button
			class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
			on:click={() => { if (showForm) { cancelEdit(); } else { editingRule = null; showForm = true; } }}
		>{showForm ? $i18n.t('Cancel') : $i18n.t('Add Rule')}</button>
	</div>

	<p class="text-xs text-gray-500 dark:text-gray-400">
		{$i18n.t('Rules map vendor or client names to accounts. The AI learns from your corrections automatically.')}
	</p>

	{#if showForm}
		<div class="p-3 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-800 space-y-2">
			<!-- Rule Type toggle -->
			<div class="flex gap-1 mb-1">
				<button
					class="px-3 py-1 text-xs rounded-lg font-medium transition {newRuleType === 'invoice' ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'}"
					on:click={() => (newRuleType = 'invoice')}
				>{$i18n.t('Invoice Rule')}</button>
				<button
					class="px-3 py-1 text-xs rounded-lg font-medium transition {newRuleType === 'bank_fee' ? 'bg-amber-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-300 dark:hover:bg-gray-600'}"
					on:click={() => (newRuleType = 'bank_fee')}
				>{$i18n.t('Bank Fee Rule')}</button>
			</div>

			<!-- Row 1: Match Field + Name Pattern + Account -->
			<div class="grid grid-cols-[auto_1fr_1fr] gap-2">
				{#if newRuleType === 'invoice'}
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Match On')}</label>
					<select bind:value={newMatchField} class="text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="vendor">{$i18n.t('Vendor')}</option>
						<option value="client">{$i18n.t('Client')}</option>
						<option value="both">{$i18n.t('Both')}</option>
					</select>
				</div>
				{/if}
				<div class={newRuleType === 'bank_fee' ? 'col-span-2' : ''}>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{newRuleType === 'bank_fee' ? $i18n.t('Description Pattern') : newMatchField === 'client' ? $i18n.t('Client Pattern') : newMatchField === 'both' ? $i18n.t('Name Pattern') : $i18n.t('Vendor Pattern')}</label>
					<input type="text" bind:value={newVendor} placeholder={newRuleType === 'bank_fee' ? 'e.g. FRAIS BANCAIRES' : newMatchField === 'client' ? 'e.g. Dupont SARL' : 'e.g. amazon'} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Debit Account')}</label>
					<select bind:value={newAccountCode} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="">—</option>
						{#each accounts as acct}
							<option value={acct.code}>{acct.code} — {acct.name}</option>
						{/each}
					</select>
				</div>
			</div>
			<!-- Counterparty (Credit) Account -->
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Credit Account')}</label>
				<select bind:value={newCounterpartyCode} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
					<option value="">—</option>
					{#each accounts as acct}
						<option value={acct.code}>{acct.code} — {acct.name}</option>
					{/each}
				</select>
			</div>
			<!-- Row 2: Match Type + Keywords + Tax -->
			<div class="grid grid-cols-4 gap-2">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Match Type')}</label>
					<select bind:value={newMatchType} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="exact">{$i18n.t('Exact')}</option>
						<option value="contains">{$i18n.t('Contains')}</option>
						<option value="prefix">{$i18n.t('Prefix')}</option>
					</select>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Keywords')}</label>
					<input type="text" bind:value={newDescKeywords} placeholder="office,supplies" class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Tax Account')}</label>
					<select bind:value={newTaxAccountCode} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="">—</option>
						{#each accounts as acct}
							<option value={acct.code}>{acct.code} — {acct.name}</option>
						{/each}
					</select>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Tax Rate %')}</label>
					<input type="number" step="0.01" bind:value={newTaxRate} placeholder="e.g. 20" class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
			</div>
			<!-- Row 3: Description + Auto-entry + Priority + Save -->
			<div class="flex gap-2 items-end">
				<div class="flex-1">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Default Description')}</label>
					<input type="text" bind:value={newDefaultDescription} placeholder="e.g. Office rent" class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<label class="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400 whitespace-nowrap pb-1">
					<input type="checkbox" bind:checked={newAutoCreateEntry} class="rounded" />
					{$i18n.t('Auto-create draft entry')}
				</label>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Priority')}</label>
					<input type="number" bind:value={newPriority} class="w-20 text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<button class="px-4 py-1.5 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition whitespace-nowrap" on:click={() => editingRule ? handleSaveEdit() : handleCreate()}>{editingRule ? $i18n.t('Update Rule') : $i18n.t('Save')}</button>
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="text-sm text-gray-400">{$i18n.t('Loading...')}</div>
	{:else if rules.length === 0}
		<div class="text-sm text-gray-400 italic">{$i18n.t('No rules yet. Rules are created automatically when you confirm invoice categories.')}</div>
	{:else}
		<!-- Pending Rules -->
		{#if pendingRules.length > 0}
		<div>
			<h4 class="text-xs font-semibold text-amber-600 dark:text-amber-400 uppercase mb-1 flex items-center gap-1">
				<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>
				{$i18n.t('Pending Approval')} ({pendingRules.length})
			</h4>
			<div class="overflow-x-auto rounded-xl border-2 border-amber-200 dark:border-amber-800/50 bg-amber-50/30 dark:bg-amber-950/10">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-amber-100/50 dark:bg-amber-900/20 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2">{$i18n.t('Pattern')}</th>
							<th class="px-2 py-2">{$i18n.t('Type')}</th>
							<th class="px-2 py-2">{$i18n.t('Debit')}</th>
							<th class="px-2 py-2">{$i18n.t('Credit')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Match')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Source')}</th>
							<th class="px-2 py-2 w-40"></th>
						</tr>
					</thead>
					<tbody>
						{#each pendingRules as rule}
							<tr class="border-b border-amber-100 dark:border-amber-900/20 hover:bg-amber-50/50 dark:hover:bg-amber-950/20">
								<td class="px-3 py-1.5 font-medium">{rule.vendor_name_pattern}</td>
								<td class="px-2 py-1.5">
									<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.rule_type === 'bank_fee' ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-400' : 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400'}">
										{rule.rule_type === 'bank_fee' ? $i18n.t('Bank Fee') : $i18n.t('Invoice')}
									</span>
								</td>
								<td class="px-2 py-1.5">{accountName(rule.account_code)}</td>
								<td class="px-2 py-1.5 text-gray-400">{rule.counterparty_account_code ? accountName(rule.counterparty_account_code) : '—'}</td>
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
								<td class="px-2 py-1.5 text-center whitespace-nowrap">
									<button class="px-2 py-0.5 text-[10px] font-medium rounded bg-green-600 text-white hover:bg-green-700 transition mr-1" on:click={() => handleApprove(rule.id)}>{$i18n.t('Approve')}</button>
									<button class="px-2 py-0.5 text-[10px] font-medium rounded bg-red-500 text-white hover:bg-red-600 transition mr-1" on:click={() => handleReject(rule.id)}>{$i18n.t('Reject')}</button>
									<button class="text-blue-600 hover:text-blue-800 dark:text-blue-400 text-[10px]" on:click={() => startEdit(rule)}>{$i18n.t('Edit')}</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
		{/if}

		<!-- Invoice Rules -->
		{#if invoiceRules.length > 0}
		<div>
			<h4 class="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase mb-1">{$i18n.t('Invoice Rules')}</h4>
			<div class="overflow-x-auto rounded-xl border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2">{$i18n.t('Name Pattern')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Field')}</th>
							<th class="px-2 py-2">{$i18n.t('Debit')}</th>
							<th class="px-2 py-2">{$i18n.t('Credit')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Match')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Auto')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Source')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Used')}</th>
							<th class="px-2 py-2 w-16"></th>
						</tr>
					</thead>
					<tbody>
						{#each invoiceRules as rule}
							<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
								<td class="px-3 py-1.5 font-medium">
									{rule.vendor_name_pattern}
									{#if rule.description_keywords?.length}
										<div class="text-[9px] text-gray-400 mt-0.5">{rule.description_keywords.join(', ')}</div>
									{/if}
								</td>
								<td class="px-2 py-1.5 text-center">
									<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.match_field === 'client' ? 'bg-indigo-100 dark:bg-indigo-900/30 text-indigo-700 dark:text-indigo-400' : rule.match_field === 'both' ? 'bg-teal-100 dark:bg-teal-900/30 text-teal-700 dark:text-teal-400' : 'bg-orange-100 dark:bg-orange-900/30 text-orange-700 dark:text-orange-400'}">
										{rule.match_field === 'client' ? $i18n.t('Client') : rule.match_field === 'both' ? $i18n.t('Both') : $i18n.t('Vendor')}
									</span>
								</td>
								<td class="px-2 py-1.5">{accountName(rule.account_code)}</td>
								<td class="px-2 py-1.5 text-gray-400">{rule.counterparty_account_code ? accountName(rule.counterparty_account_code) : '—'}</td>
								<td class="px-2 py-1.5 text-center">
									<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.match_type === 'exact' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : rule.match_type === 'contains' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'}">
										{rule.match_type}
									</span>
								</td>
								<td class="px-2 py-1.5 text-center">
									{#if rule.auto_create_entry}
										<span class="px-1.5 py-0.5 rounded text-[10px] font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400">{$i18n.t('Auto')}</span>
									{:else}
										<span class="text-[10px] text-gray-400">—</span>
									{/if}
								</td>
								<td class="px-2 py-1.5 text-center">
									<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.created_by === 'system' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' : rule.created_by === 'human' ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-400' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}">
										{rule.created_by === 'system' ? $i18n.t('System') : rule.created_by === 'human' ? $i18n.t('Manual') : $i18n.t('AI')}
									</span>
								</td>
								<td class="px-2 py-1.5 text-center text-gray-400">{rule.times_used}x</td>
								<td class="px-2 py-1.5 text-center whitespace-nowrap">
									<button class="text-blue-600 hover:text-blue-800 dark:text-blue-400 text-xs mr-1" on:click={() => startEdit(rule)}>{$i18n.t('Edit')}</button>
									<button
										class="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200 text-xs mr-1 disabled:opacity-50"
										disabled={applyingRuleId === rule.id}
										on:click={() => handleApply(rule)}
									>{applyingRuleId === rule.id ? $i18n.t('Applying...') : $i18n.t('Apply')}</button>
									<button class="text-red-500 hover:text-red-700 text-xs" on:click={() => handleDelete(rule.id)}>{$i18n.t('Remove')}</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
		{/if}

		<!-- Bank Fee Rules -->
		{#if bankFeeRules.length > 0}
		<div>
			<h4 class="text-xs font-semibold text-amber-600 dark:text-amber-400 uppercase mb-1">{$i18n.t('Bank Fee Rules')}</h4>
			<div class="overflow-x-auto rounded-xl border border-amber-100/50 dark:border-amber-900/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-amber-50/50 dark:bg-amber-950/20 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2">{$i18n.t('Description Pattern')}</th>
							<th class="px-2 py-2">{$i18n.t('Debit')}</th>
							<th class="px-2 py-2">{$i18n.t('Credit')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Match')}</th>
							<th class="px-2 py-2">{$i18n.t('Default Description')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Used')}</th>
							<th class="px-2 py-2 w-16"></th>
						</tr>
					</thead>
					<tbody>
						{#each bankFeeRules as rule}
							<tr class="border-b border-amber-50 dark:border-amber-950/20 hover:bg-amber-50/30 dark:hover:bg-amber-950/10">
								<td class="px-3 py-1.5 font-medium">{rule.vendor_name_pattern}</td>
								<td class="px-2 py-1.5">{accountName(rule.account_code)}</td>
								<td class="px-2 py-1.5 text-gray-400">{rule.counterparty_account_code ? accountName(rule.counterparty_account_code) : '—'}</td>
								<td class="px-2 py-1.5 text-center">
									<span class="px-1.5 py-0.5 rounded text-[10px] font-medium {rule.match_type === 'exact' ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400' : rule.match_type === 'contains' ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400' : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400'}">
										{rule.match_type}
									</span>
								</td>
								<td class="px-2 py-1.5 text-gray-400">{rule.default_description ?? '—'}</td>
								<td class="px-2 py-1.5 text-center text-gray-400">{rule.times_used}x</td>
								<td class="px-2 py-1.5 text-center whitespace-nowrap">
									<button class="text-blue-600 hover:text-blue-800 dark:text-blue-400 text-xs mr-1" on:click={() => startEdit(rule)}>{$i18n.t('Edit')}</button>
									<button
										class="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200 text-xs mr-1 disabled:opacity-50"
										disabled={applyingRuleId === rule.id}
										on:click={() => handleApply(rule)}
									>{applyingRuleId === rule.id ? $i18n.t('Applying...') : $i18n.t('Apply')}</button>
									<button class="text-red-500 hover:text-red-700 text-xs" on:click={() => handleDelete(rule.id)}>{$i18n.t('Remove')}</button>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		</div>
		{/if}
	{/if}
</div>
