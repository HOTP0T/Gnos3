<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		getChartTemplates,
		getPeriodTemplates,
		createPeriodTemplate,
		deleteChartTemplate,
		deletePeriodTemplate,
		downloadChartImportTemplate,
		downloadPeriodImportTemplate
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import ExcelImportModal from '$lib/components/accounting/ExcelImportModal.svelte';

	const i18n = getContext('i18n');

	// Tab state
	let activeTab: 'chart' | 'period' = 'chart';

	// Data
	let chartTemplates: any[] = [];
	let periodTemplates: any[] = [];
	let loadingCharts = true;
	let loadingPeriods = true;

	// Expanded template (by id)
	let expandedChartId: number | null = null;
	let expandedPeriodId: number | null = null;

	// Collapsed type sections within expanded template
	let collapsedTypes: Set<string> = new Set();
	const toggleTypeSection = (type: string) => {
		if (collapsedTypes.has(type)) collapsedTypes.delete(type);
		else collapsedTypes.add(type);
		collapsedTypes = collapsedTypes;
	};

	const ACCOUNT_TYPES = [
		{ key: 'asset', label: 'Assets', badge: 'bg-blue-500/20 text-blue-700 dark:text-blue-200', header: 'bg-blue-50/50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800' },
		{ key: 'liability', label: 'Liabilities', badge: 'bg-red-500/20 text-red-700 dark:text-red-200', header: 'bg-red-50/50 dark:bg-red-900/20 border-red-200 dark:border-red-800' },
		{ key: 'equity', label: 'Equity', badge: 'bg-purple-500/20 text-purple-700 dark:text-purple-200', header: 'bg-purple-50/50 dark:bg-purple-900/20 border-purple-200 dark:border-purple-800' },
		{ key: 'revenue', label: 'Revenue', badge: 'bg-green-500/20 text-green-700 dark:text-green-200', header: 'bg-green-50/50 dark:bg-green-900/20 border-green-200 dark:border-green-800' },
		{ key: 'expense', label: 'Expenses', badge: 'bg-yellow-500/20 text-yellow-700 dark:text-yellow-200', header: 'bg-yellow-50/50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800' },
	];

	const groupTemplateAccounts = (accounts: any[]) => {
		return ACCOUNT_TYPES.map((type) => ({
			...type,
			accounts: accounts
				.filter((a: any) => (a.account_type ?? a.type ?? '').toLowerCase() === type.key)
				.sort((a: any, b: any) => (a.code ?? '').localeCompare(b.code ?? ''))
		})).filter((g) => g.accounts.length > 0);
	};

	const getIndentLevel = (acct: any): number => {
		if (acct.parent_code) return (acct.parent_code.split('.').length);
		const dots = (acct.code ?? '').split('.').length - 1;
		return dots;
	};

	// Create template inline form
	let showCreateForm = false;
	let newTemplateName = '';
	let newTemplateCountry = '';
	let creatingTemplate = false;

	// Period template create form
	let showPeriodCreateForm = false;
	let newPeriodTemplateName = '';
	let newPeriodTemplateDesc = '';
	let periodEntries: Array<{ name: string; start_date: string; end_date: string }> = [];
	let creatingPeriodTemplate = false;

	// Excel import modal
	let showImportModal = false;
	let importType: 'chart' | 'period' = 'chart';

	// ─── Chart Templates ───────────────────────────────────────────────────────

	const loadChartTemplates = async () => {
		loadingCharts = true;
		try {
			const res = await getChartTemplates();
			chartTemplates = Array.isArray(res) ? res : res?.templates ?? [];
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load chart templates')}: ${err}`);
		}
		loadingCharts = false;
	};

	const handleDeleteChart = async (id: number) => {
		try {
			await deleteChartTemplate(id);
			toast.success($i18n.t('Chart template deleted'));
			await loadChartTemplates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to delete template') + ': ' + msg);
		}
	};

	const toggleChartExpand = (id: number) => {
		expandedChartId = expandedChartId === id ? null : id;
	};

	// ─── Period Templates ──────────────────────────────────────────────────────

	const loadPeriodTemplates = async () => {
		loadingPeriods = true;
		try {
			const res = await getPeriodTemplates();
			periodTemplates = Array.isArray(res) ? res : res?.templates ?? [];
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load period templates')}: ${err}`);
		}
		loadingPeriods = false;
	};

	const handleDeletePeriod = async (id: number) => {
		try {
			await deletePeriodTemplate(id);
			toast.success($i18n.t('Period template deleted'));
			await loadPeriodTemplates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to delete template') + ': ' + msg);
		}
	};

	const togglePeriodExpand = (id: number) => {
		expandedPeriodId = expandedPeriodId === id ? null : id;
	};

	// ─── Period Template Manual Create ────────────────────────────────────────

	const addPeriodEntry = () => {
		periodEntries = [...periodEntries, { name: '', start_date: '', end_date: '' }];
	};

	const removePeriodEntry = (index: number) => {
		periodEntries = periodEntries.filter((_, i) => i !== index);
	};

	const autoSuggestPeriodName = (index: number) => {
		const entry = periodEntries[index];
		if (entry.start_date && !entry.name) {
			const d = new Date(entry.start_date);
			const monthName = d.toLocaleString('en', { month: 'long' });
			periodEntries[index].name = `${monthName} ${d.getFullYear()}`;
			periodEntries = [...periodEntries];
		}
	};

	const handleCreatePeriodTemplate = async () => {
		if (!newPeriodTemplateName.trim()) return;
		const validEntries = periodEntries.filter(
			(e) => e.name && e.start_date && e.end_date
		);
		if (validEntries.length === 0) {
			toast.error($i18n.t('Add at least one period entry'));
			return;
		}

		creatingPeriodTemplate = true;
		try {
			await createPeriodTemplate({
				name: newPeriodTemplateName.trim(),
				description: newPeriodTemplateDesc.trim() || undefined,
				entries: validEntries.map((e, i) => ({
					name: e.name,
					start_date: e.start_date,
					end_date: e.end_date,
					sort_order: i
				}))
			});
			toast.success($i18n.t('Period template created'));
			showPeriodCreateForm = false;
			newPeriodTemplateName = '';
			newPeriodTemplateDesc = '';
			periodEntries = [];
			await loadPeriodTemplates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create template') + ': ' + msg);
		}
		creatingPeriodTemplate = false;
	};

	const resetPeriodForm = () => {
		showPeriodCreateForm = false;
		newPeriodTemplateName = '';
		newPeriodTemplateDesc = '';
		periodEntries = [];
	};

	// ─── Import ────────────────────────────────────────────────────────────────

	const openImportModal = (type: 'chart' | 'period') => {
		importType = type;
		showImportModal = true;
	};

	const handleImported = () => {
		if (importType === 'chart') {
			loadChartTemplates();
		} else {
			loadPeriodTemplates();
		}
	};

	// ─── Lifecycle ─────────────────────────────────────────────────────────────

	onMount(async () => {
		await Promise.all([loadChartTemplates(), loadPeriodTemplates()]);
	});
</script>

<ExcelImportModal
	bind:show={showImportModal}
	type={importType}
	on:imported={handleImported}
/>

<div class="py-2">
	<!-- Header -->
	<div class="pt-0.5 pb-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900">
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Accounting Settings')}</div>
		</div>
	</div>

	<!-- Tabs -->
	<div class="flex border-b border-gray-200 dark:border-gray-800 mt-2 mb-4">
		<button
			class="px-4 py-2 text-sm font-medium transition border-b-2 {activeTab === 'chart'
				? 'border-blue-500 text-blue-600 dark:text-blue-400'
				: 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'}"
			on:click={() => (activeTab = 'chart')}
		>
			{$i18n.t('Chart Templates')}
		</button>
		<button
			class="px-4 py-2 text-sm font-medium transition border-b-2 {activeTab === 'period'
				? 'border-blue-500 text-blue-600 dark:text-blue-400'
				: 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'}"
			on:click={() => (activeTab = 'period')}
		>
			{$i18n.t('Period Templates')}
		</button>
	</div>

	<!-- Chart Templates Tab -->
	{#if activeTab === 'chart'}
		<div class="flex gap-2 mb-3">
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-gray-900 hover:bg-gray-850 text-white dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium transition flex items-center gap-1.5"
				on:click={() => (showCreateForm = !showCreateForm)}
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
				{$i18n.t('Create Template')}
			</button>
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-500 dark:hover:bg-blue-600 font-medium transition flex items-center gap-1.5"
				on:click={() => openImportModal('chart')}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
					/>
				</svg>
				{$i18n.t('Import from Excel')}
			</button>
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white dark:bg-emerald-500 dark:hover:bg-emerald-600 font-medium transition flex items-center gap-1.5"
				on:click={downloadChartImportTemplate}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"
					/>
				</svg>
				{$i18n.t('Download Template')}
			</button>
		</div>

		<!-- Inline Create Form -->
		{#if showCreateForm}
			<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30 mb-3">
				<div class="text-sm font-medium dark:text-gray-200 mb-3">
					{$i18n.t('Create New Chart Template')}
				</div>
				<div class="grid grid-cols-1 md:grid-cols-3 gap-3 items-end">
					<div>
						<label
							for="template-name"
							class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>
							{$i18n.t('Name')} *
						</label>
						<input
							id="template-name"
							type="text"
							bind:value={newTemplateName}
							placeholder={$i18n.t('e.g. French PCG')}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						/>
					</div>
					<div>
						<label
							for="template-country"
							class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>
							{$i18n.t('Country')}
						</label>
						<input
							id="template-country"
							type="text"
							bind:value={newTemplateCountry}
							placeholder={$i18n.t('e.g. FR')}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						/>
					</div>
					<div class="flex gap-2">
						<button
							class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
							on:click={() => {
								showCreateForm = false;
								newTemplateName = '';
								newTemplateCountry = '';
							}}
						>
							{$i18n.t('Cancel')}
						</button>
						<button
							class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
							disabled={!newTemplateName || creatingTemplate}
							on:click={() => openImportModal('chart')}
						>
							{$i18n.t('Import Excel')}
						</button>
					</div>
				</div>
			</div>
		{/if}

		<!-- Chart Templates List -->
		{#if loadingCharts}
			<div class="flex justify-center my-10">
				<Spinner className="size-5" />
			</div>
		{:else if chartTemplates.length === 0}
			<div class="flex justify-center my-10 text-sm text-gray-500 dark:text-gray-400">
				{$i18n.t('No chart templates found. Import one from Excel to get started.')}
			</div>
		{:else}
			<div class="space-y-2">
				{#each chartTemplates as template (template.id)}
					<div class="rounded-xl border border-gray-100 dark:border-gray-850 overflow-hidden">
						<!-- Template Header -->
						<!-- svelte-ignore a11y-click-events-have-key-events -->
						<!-- svelte-ignore a11y-no-static-element-interactions -->
						<div
							class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
							on:click={() => toggleChartExpand(template.id)}
						>
							<div class="flex items-center gap-3">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									stroke-width="2"
									stroke="currentColor"
									class="size-3.5 transition-transform {expandedChartId === template.id ? '' : '-rotate-90'}"
								>
									<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
								</svg>
								<div>
									<span class="text-sm font-semibold dark:text-gray-200">
										{template.name}
									</span>
									{#if template.country}
										<span class="text-xs text-gray-400 ml-2">
											({template.country})
										</span>
									{/if}
								</div>
								<span class="text-xs text-gray-400">
									{template.accounts?.length ?? template.account_count ?? 0} {$i18n.t('accounts')}
								</span>
								{#if template.is_builtin}
									<span class="text-xs font-medium px-2 py-0.5 rounded-lg bg-purple-500/20 text-purple-700 dark:text-purple-200">
										{$i18n.t('Built-in')}
									</span>
								{/if}
							</div>
							<div class="flex items-center gap-2">
								{#if !template.is_builtin}
									<button
										class="p-1.5 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition text-red-500"
										on:click|stopPropagation={() => handleDeleteChart(template.id)}
										title={$i18n.t('Delete')}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											viewBox="0 0 24 24"
											stroke-width="2"
											stroke="currentColor"
											class="size-4"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
											/>
										</svg>
									</button>
								{/if}
							</div>
						</div>

						<!-- Expanded Accounts List — grouped by type with colors -->
						{#if expandedChartId === template.id && template.accounts?.length > 0}
							<div class="divide-y divide-gray-100 dark:divide-gray-850/50">
								{#each groupTemplateAccounts(template.accounts) as group}
									<!-- Type Section Header -->
									<!-- svelte-ignore a11y-click-events-have-key-events -->
									<!-- svelte-ignore a11y-no-static-element-interactions -->
									<div
										class="flex items-center justify-between px-4 py-2 cursor-pointer select-none {group.header}"
										on:click={() => toggleTypeSection(template.id + '-' + group.key)}
									>
										<div class="flex items-center gap-2">
											<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3 transition-transform {collapsedTypes.has(template.id + '-' + group.key) ? '-rotate-90' : ''}">
												<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
											</svg>
											<span class="text-xs font-semibold dark:text-gray-200">{$i18n.t(group.label)}</span>
											<span class="text-xs font-medium px-1.5 py-0.5 rounded-lg {group.badge}">{group.accounts.length}</span>
										</div>
									</div>

									{#if !collapsedTypes.has(template.id + '-' + group.key)}
										<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300 table-fixed">
											<colgroup>
												<col class="w-[90px]" />
												<col />
												<col class="w-[80px]" />
												<col class="w-[80px]" />
											</colgroup>
											<thead class="text-[10px] uppercase bg-gray-50/30 dark:bg-gray-850/20 text-gray-500 dark:text-gray-400">
												<tr>
													<th class="px-3 py-1.5">{$i18n.t('Code')}</th>
													<th class="px-3 py-1.5">{$i18n.t('Name')}</th>
													<th class="px-3 py-1.5 whitespace-nowrap">{$i18n.t('Balance')}</th>
													<th class="px-3 py-1.5 whitespace-nowrap">{$i18n.t('Parent')}</th>
												</tr>
											</thead>
											<tbody>
												{#each group.accounts as acct}
													<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30 transition">
														<td class="px-3 py-1.5 font-mono whitespace-nowrap">{acct.code}</td>
														<td class="px-3 py-1.5 truncate" style="padding-left: {12 + getIndentLevel(acct) * 14}px" title={acct.description ? `${acct.name} — ${acct.description}` : acct.name}>
															<span class="dark:text-gray-200">{acct.name}</span>
															{#if acct.description}
																<span class="text-[10px] text-gray-400 dark:text-gray-500 ml-1">— {acct.description}</span>
															{/if}
														</td>
														<td class="px-3 py-1.5 whitespace-nowrap">
															<span class="text-[10px] font-medium uppercase {group.badge} px-1.5 py-0.5 rounded">{acct.normal_balance ?? ''}</span>
														</td>
														<td class="px-3 py-1.5 font-mono text-gray-400 whitespace-nowrap">{acct.parent_code ?? ''}</td>
													</tr>
												{/each}
											</tbody>
										</table>
									{/if}
								{/each}
							</div>
						{:else if expandedChartId === template.id}
							<div class="px-4 py-3 text-xs text-gray-400 dark:text-gray-500">
								{$i18n.t('No accounts in this template.')}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	{/if}

	<!-- Period Templates Tab -->
	{#if activeTab === 'period'}
		<div class="flex gap-2 mb-3">
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-gray-900 hover:bg-gray-850 text-white dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium transition flex items-center gap-1.5"
				on:click={() => {
					showPeriodCreateForm = !showPeriodCreateForm;
					if (showPeriodCreateForm && periodEntries.length === 0) addPeriodEntry();
				}}
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
				{$i18n.t('Create Template')}
			</button>
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-blue-600 hover:bg-blue-700 text-white dark:bg-blue-500 dark:hover:bg-blue-600 font-medium transition flex items-center gap-1.5"
				on:click={() => openImportModal('period')}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5"
					/>
				</svg>
				{$i18n.t('Import from Excel')}
			</button>
			<button
				class="px-3.5 py-1.5 text-sm rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white dark:bg-emerald-500 dark:hover:bg-emerald-600 font-medium transition flex items-center gap-1.5"
				on:click={downloadPeriodImportTemplate}
			>
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="2"
					stroke="currentColor"
					class="size-4"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"
					/>
				</svg>
				{$i18n.t('Download Template')}
			</button>
		</div>

		<!-- Inline Period Create Form -->
		{#if showPeriodCreateForm}
			<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30 mb-3">
				<div class="text-sm font-medium dark:text-gray-200 mb-3">
					{$i18n.t('Create New Period Template')}
				</div>

				<div class="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
							{$i18n.t('Template Name')} *
						</label>
						<input
							type="text"
							bind:value={newPeriodTemplateName}
							placeholder={$i18n.t('e.g. Calendar Year 2025 Monthly')}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						/>
					</div>
					<div>
						<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
							{$i18n.t('Description')}
						</label>
						<input
							type="text"
							bind:value={newPeriodTemplateDesc}
							placeholder={$i18n.t('Optional description')}
							class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
						/>
					</div>
				</div>

				<!-- Period Entries -->
				<div class="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
					{$i18n.t('Period Entries')}
				</div>

				<div class="space-y-2 mb-3">
					{#each periodEntries as entry, idx}
						<div class="flex items-center gap-2">
							<input
								type="text"
								bind:value={entry.name}
								placeholder={$i18n.t('Period name')}
								class="flex-1 text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
							<input
								type="date"
								bind:value={entry.start_date}
								on:change={() => autoSuggestPeriodName(idx)}
								class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
							<input
								type="date"
								bind:value={entry.end_date}
								class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
							/>
							<button
								class="p-1 hover:bg-red-50 dark:hover:bg-red-900/20 rounded transition text-red-500"
								on:click={() => removePeriodEntry(idx)}
								title={$i18n.t('Remove')}
							>
								<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-4">
									<path stroke-linecap="round" stroke-linejoin="round" d="M6 18 18 6M6 6l12 12" />
								</svg>
							</button>
						</div>
					{/each}
				</div>

				<button
					class="text-xs text-blue-600 dark:text-blue-400 hover:underline mb-4"
					on:click={addPeriodEntry}
				>
					+ {$i18n.t('Add Period')}
				</button>

				<div class="flex gap-2 pt-2 border-t border-gray-100 dark:border-gray-800">
					<button
						class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
						on:click={resetPeriodForm}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
						disabled={!newPeriodTemplateName.trim() || periodEntries.length === 0 || creatingPeriodTemplate}
						on:click={handleCreatePeriodTemplate}
					>
						{creatingPeriodTemplate ? $i18n.t('Creating...') : $i18n.t('Create Template')}
					</button>
				</div>
			</div>
		{/if}

		<!-- Period Templates List -->
		{#if loadingPeriods}
			<div class="flex justify-center my-10">
				<Spinner className="size-5" />
			</div>
		{:else if periodTemplates.length === 0}
			<div class="flex justify-center my-10 text-sm text-gray-500 dark:text-gray-400">
				{$i18n.t('No period templates found. Create one manually or import from Excel.')}
			</div>
		{:else}
			<div class="space-y-2">
				{#each periodTemplates as template (template.id)}
					<div class="rounded-xl border border-gray-100 dark:border-gray-850 overflow-hidden">
						<!-- Template Header -->
						<!-- svelte-ignore a11y-click-events-have-key-events -->
						<!-- svelte-ignore a11y-no-static-element-interactions -->
						<div
							class="flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-850/50 cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-850 transition"
							on:click={() => togglePeriodExpand(template.id)}
						>
							<div class="flex items-center gap-3">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									stroke-width="2"
									stroke="currentColor"
									class="size-3.5 transition-transform {expandedPeriodId === template.id ? '' : '-rotate-90'}"
								>
									<path stroke-linecap="round" stroke-linejoin="round" d="m19.5 8.25-7.5 7.5-7.5-7.5" />
								</svg>
								<span class="text-sm font-semibold dark:text-gray-200">
									{template.name}
								</span>
								<span class="text-xs text-gray-400">
									{template.entries?.length ?? template.entry_count ?? 0} {$i18n.t('entries')}
								</span>
							</div>
							<div class="flex items-center gap-2">
								<button
									class="p-1.5 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition text-red-500"
									on:click|stopPropagation={() => handleDeletePeriod(template.id)}
									title={$i18n.t('Delete')}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="2"
										stroke="currentColor"
										class="size-4"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="m14.74 9-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 0 1-2.244 2.077H8.084a2.25 2.25 0 0 1-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 0 0-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 0 1 3.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 0 0-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 0 0-7.5 0"
										/>
									</svg>
								</button>
							</div>
						</div>

						<!-- Expanded Entries List -->
						{#if expandedPeriodId === template.id && template.entries?.length > 0}
							<div class="overflow-x-auto">
								<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
									<thead class="text-xs text-gray-500 dark:text-gray-400 uppercase bg-gray-50/50 dark:bg-gray-850/30">
										<tr class="border-b border-gray-100 dark:border-gray-850">
											<th scope="col" class="px-4 py-2">{$i18n.t('Name')}</th>
											<th scope="col" class="px-4 py-2 w-36">{$i18n.t('Start Date')}</th>
											<th scope="col" class="px-4 py-2 w-36">{$i18n.t('End Date')}</th>
										</tr>
									</thead>
									<tbody>
										{#each template.entries as entry}
											<tr class="border-b border-gray-50 dark:border-gray-850/50 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition">
												<td class="px-4 py-1.5 dark:text-gray-200">{entry.name}</td>
												<td class="px-4 py-1.5">{entry.start_date}</td>
												<td class="px-4 py-1.5">{entry.end_date}</td>
											</tr>
										{/each}
									</tbody>
								</table>
							</div>
						{:else if expandedPeriodId === template.id}
							<div class="px-4 py-3 text-xs text-gray-400 dark:text-gray-500">
								{$i18n.t('No entries in this template.')}
							</div>
						{/if}
					</div>
				{/each}
			</div>
		{/if}
	{/if}
</div>
