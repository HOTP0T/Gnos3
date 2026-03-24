<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getRecurringTemplates, createRecurringTemplate, deleteRecurringTemplate, generateRecurringNow, getAccounts } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let templates: any[] = [];
	let accounts: any[] = [];
	let loading = true;
	let showForm = false;

	// Form state
	let form = {
		name: '',
		frequency: 'monthly',
		day_of_month: 1,
		start_date: new Date().toISOString().slice(0, 10),
		end_date: '',
		currency: 'USD',
		reference_prefix: '',
		description: '',
		auto_post: false,
		lines: [
			{ account_code: '', debit: 0, credit: 0, description: '' },
			{ account_code: '', debit: 0, credit: 0, description: '' },
		],
	};

	const load = async () => {
		loading = true;
		try {
			const [tmplData, acctData] = await Promise.all([
				getRecurringTemplates(companyId),
				getAccounts({ company_id: companyId }),
			]);
			templates = tmplData ?? [];
			const accts = acctData?.accounts ?? acctData ?? [];
			accounts = Array.isArray(accts) ? accts : [];
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
		loading = false;
	};

	onMount(load);

	const addLine = () => { form.lines = [...form.lines, { account_code: '', debit: 0, credit: 0, description: '' }]; };
	const removeLine = (i: number) => { form.lines = form.lines.filter((_, idx) => idx !== i); };

	const handleCreate = async () => {
		if (!form.name) { toast.error($i18n.t('Name is required')); return; }
		if (form.lines.length < 2) { toast.error($i18n.t('At least 2 lines required')); return; }
		try {
			await createRecurringTemplate(companyId, {
				name: form.name,
				frequency: form.frequency,
				day_of_month: form.day_of_month,
				start_date: form.start_date,
				end_date: form.end_date || undefined,
				currency: form.currency,
				reference_prefix: form.reference_prefix || undefined,
				description: form.description || undefined,
				auto_post: form.auto_post,
				transaction_type: 'journal',
				lines_template: form.lines.filter(l => l.account_code),
			});
			toast.success($i18n.t('Template created'));
			showForm = false;
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	const handleDelete = async (id: number) => {
		try {
			await deleteRecurringTemplate(id);
			toast.success($i18n.t('Template deleted'));
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	const handleGenerate = async (id: number) => {
		try {
			const res = await generateRecurringNow(id);
			toast.success($i18n.t(`Generated ${res.generated} entries`));
			await load();
		} catch (err: any) { toast.error(err?.detail ?? `${err}`); }
	};

	const freqLabel = (f: string) => {
		const map: Record<string, string> = { weekly: 'Weekly', monthly: 'Monthly', quarterly: 'Quarterly', yearly: 'Yearly' };
		return map[f] || f;
	};
</script>

<div class="space-y-3">
	<div class="flex items-center justify-between">
		<h3 class="text-sm font-semibold dark:text-gray-200">{$i18n.t('Recurring Transactions')}</h3>
		<button class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition" on:click={() => (showForm = !showForm)}>
			{showForm ? $i18n.t('Cancel') : $i18n.t('New Template')}
		</button>
	</div>

	{#if showForm}
		<div class="p-4 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-800 space-y-3">
			<div class="grid grid-cols-2 md:grid-cols-4 gap-2">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Name')}</label>
					<input type="text" bind:value={form.name} placeholder="Monthly Rent" class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Frequency')}</label>
					<select bind:value={form.frequency} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
						<option value="weekly">{$i18n.t('Weekly')}</option>
						<option value="monthly">{$i18n.t('Monthly')}</option>
						<option value="quarterly">{$i18n.t('Quarterly')}</option>
						<option value="yearly">{$i18n.t('Yearly')}</option>
					</select>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Start Date')}</label>
					<input type="date" bind:value={form.start_date} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Currency')}</label>
					<input type="text" bind:value={form.currency} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
				</div>
			</div>
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Description')}</label>
				<input type="text" bind:value={form.description} class="w-full text-sm rounded-lg px-3 py-1.5 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
			</div>
			<div>
				<div class="flex items-center justify-between mb-1">
					<label class="text-xs font-medium text-gray-500 dark:text-gray-400">{$i18n.t('Journal Lines')}</label>
					<button class="text-xs text-blue-600 hover:text-blue-700" on:click={addLine}>+ {$i18n.t('Add Line')}</button>
				</div>
				{#each form.lines as line, i}
					<div class="flex gap-1 mb-1">
						<select bind:value={line.account_code} class="flex-1 text-xs rounded px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden">
							<option value="">{$i18n.t('Account...')}</option>
							{#each accounts as acct}
								<option value={acct.code}>{acct.code} — {acct.name}</option>
							{/each}
						</select>
						<input type="number" bind:value={line.debit} placeholder="Debit" class="w-24 text-xs rounded px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
						<input type="number" bind:value={line.credit} placeholder="Credit" class="w-24 text-xs rounded px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
						<input type="text" bind:value={line.description} placeholder="Desc" class="w-32 text-xs rounded px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden" />
						{#if form.lines.length > 2}
							<button class="text-red-500 text-xs px-1" on:click={() => removeLine(i)}>x</button>
						{/if}
					</div>
				{/each}
			</div>
			<div class="flex items-center gap-3">
				<label class="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400">
					<input type="checkbox" bind:checked={form.auto_post} class="rounded" />
					{$i18n.t('Auto-post generated entries')}
				</label>
				<button class="px-4 py-1.5 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition" on:click={handleCreate}>{$i18n.t('Create Template')}</button>
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="flex justify-center my-6"><Spinner className="size-5" /></div>
	{:else if templates.length === 0}
		<div class="text-sm text-gray-400 italic">{$i18n.t('No recurring templates. Create one for repeating entries like rent, salaries, or subscriptions.')}</div>
	{:else}
		<div class="space-y-2">
			{#each templates as tmpl}
				<div class="p-3 rounded-lg border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
					<div class="flex items-center justify-between">
						<div>
							<span class="text-sm font-medium dark:text-gray-200">{tmpl.name}</span>
							<span class="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400">{freqLabel(tmpl.frequency)}</span>
							{#if !tmpl.is_active}<span class="ml-1 text-[10px] px-1.5 py-0.5 rounded bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400">{$i18n.t('Inactive')}</span>{/if}
						</div>
						<div class="flex gap-2">
							<button class="text-xs text-blue-600 hover:text-blue-700" on:click={() => handleGenerate(tmpl.id)}>{$i18n.t('Generate Now')}</button>
							<button class="text-xs text-red-500 hover:text-red-700" on:click={() => handleDelete(tmpl.id)}>{$i18n.t('Delete')}</button>
						</div>
					</div>
					<div class="mt-1 text-xs text-gray-500 dark:text-gray-400 flex gap-4">
						<span>{$i18n.t('Next')}: {tmpl.next_run_date}</span>
						{#if tmpl.last_generated_date}<span>{$i18n.t('Last')}: {tmpl.last_generated_date}</span>{/if}
						<span>{tmpl.currency}</span>
						{#if tmpl.auto_post}<span class="text-green-600 dark:text-green-400">{$i18n.t('Auto-post')}</span>{/if}
					</div>
					{#if tmpl.description}<div class="mt-1 text-xs text-gray-400">{tmpl.description}</div>{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>
