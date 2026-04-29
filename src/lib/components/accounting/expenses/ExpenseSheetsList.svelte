<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';
	import {
		getExpenseSheets,
		getEmployees,
		generateExpenseSheet,
		getExpenseSheetCandidates,
		getPendingReimbursableInvoices
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import ExpenseSheetStatusBadge from './ExpenseSheetStatusBadge.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let sheets: any[] = [];
	let employees: any[] = [];
	let pendingGroups: any[] = [];
	let pendingTotal = 0;
	let loading = true;
	let showGenerate = false;
	let generating = false;

	let statusFilter = '';
	let employeeFilter: number | '' = '';
	let dateFrom = '';
	let dateTo = '';
	let search = '';

	const defaultPeriod = () => {
		const now = new Date();
		const start = new Date(now.getFullYear(), now.getMonth(), 1);
		const end = new Date(now.getFullYear(), now.getMonth() + 1, 0);
		return {
			start: start.toISOString().slice(0, 10),
			end: end.toISOString().slice(0, 10)
		};
	};

	let genForm = (() => {
		const p = defaultPeriod();
		return {
			employee_id: 0,
			period_start: p.start,
			period_end: p.end,
			title: '',
			include_uncategorized: true
		};
	})();
	let candidatePreview: { total: number } | null = null;

	const load = async () => {
		loading = true;
		try {
			const [sheetData, empData, pendingData] = await Promise.all([
				getExpenseSheets(companyId, {
					status: statusFilter || undefined,
					employee_id: typeof employeeFilter === 'number' ? employeeFilter : undefined,
					date_from: dateFrom || undefined,
					date_to: dateTo || undefined,
					search: search || undefined
				}),
				getEmployees(companyId, { active: true }),
				getPendingReimbursableInvoices(companyId)
			]);
			sheets = sheetData?.sheets ?? [];
			employees = empData?.employees ?? [];
			pendingGroups = pendingData?.groups ?? [];
			pendingTotal = pendingData?.total_invoices ?? 0;
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		loading = false;
	};

	const generateForEmployee = async (group: any) => {
		const periodStart = group.earliest_date ?? defaultPeriod().start;
		const periodEnd = group.latest_date ?? defaultPeriod().end;
		generating = true;
		try {
			const sheet = await generateExpenseSheet(companyId, {
				employee_id: group.employee_id,
				period_start: periodStart,
				period_end: periodEnd,
				include_uncategorized: true
			});
			toast.success($i18n.t('Expense sheet created'));
			goto(`/accounting/company/${companyId}/expenses/${sheet.id}`);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		generating = false;
	};

	onMount(load);

	const openGenerate = () => {
		showGenerate = true;
		candidatePreview = null;
		if (employees.length && !genForm.employee_id) {
			genForm.employee_id = employees[0].id;
		}
	};

	const previewCandidates = async () => {
		if (!genForm.employee_id) {
			toast.error($i18n.t('Pick an employee first'));
			return;
		}
		try {
			candidatePreview = await getExpenseSheetCandidates(companyId, {
				employee_id: genForm.employee_id,
				period_start: genForm.period_start,
				period_end: genForm.period_end
			});
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
	};

	const handleGenerate = async () => {
		if (!genForm.employee_id) {
			toast.error($i18n.t('Pick an employee'));
			return;
		}
		generating = true;
		try {
			const sheet = await generateExpenseSheet(companyId, {
				employee_id: genForm.employee_id,
				period_start: genForm.period_start,
				period_end: genForm.period_end,
				title: genForm.title || undefined,
				include_uncategorized: genForm.include_uncategorized
			});
			toast.success($i18n.t('Expense sheet created'));
			showGenerate = false;
			goto(`/accounting/company/${companyId}/expenses/${sheet.id}`);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		generating = false;
	};

	const money = (amount: any, currency: string) => {
		if (amount == null) return '';
		const n = typeof amount === 'number' ? amount : parseFloat(amount);
		if (!isFinite(n)) return '';
		try {
			return new Intl.NumberFormat(undefined, {
				style: 'currency',
				currency: currency || 'USD'
			}).format(n);
		} catch {
			return `${currency} ${n.toFixed(2)}`;
		}
	};
</script>

<div class="py-2 space-y-3">
	<div class="flex items-center justify-between flex-wrap gap-2">
		<div>
			<h2 class="text-lg font-semibold dark:text-gray-200">{$i18n.t('Expense Sheets')}</h2>
			<p class="text-xs text-gray-500 dark:text-gray-400">
				{$i18n.t(
					'Employee expense reports. Generate a sheet, submit for approval, then mark as paid.'
				)}
			</p>
		</div>
		<button
			class="px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition"
			on:click={openGenerate}
		>
			+ {$i18n.t('New Sheet')}
		</button>
	</div>

	<!-- Pending receipts -->
	{#if pendingTotal > 0}
		<div
			class="p-3 rounded-lg border border-amber-200 dark:border-amber-900/50 bg-amber-50 dark:bg-amber-950/20"
		>
			<div class="flex items-center justify-between flex-wrap gap-2 mb-2">
				<div>
					<div class="text-sm font-semibold text-amber-800 dark:text-amber-300">
						{pendingTotal}
						{$i18n.t('reimbursable receipt(s) waiting to be sheeted')}
					</div>
					<div class="text-xs text-amber-700/80 dark:text-amber-400/80">
						{$i18n.t(
							'These have an employee assigned but are not on any active expense sheet yet.'
						)}
					</div>
				</div>
			</div>
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
				{#each pendingGroups as g}
					<div
						class="flex items-center justify-between p-2 rounded bg-white dark:bg-gray-900 border border-amber-100 dark:border-amber-900/30"
					>
						<div class="min-w-0">
							<div class="text-sm font-medium dark:text-gray-200 truncate">
								{g.employee_name}
								<span class="text-xs text-gray-400">({g.employee_code})</span>
							</div>
							<div class="text-xs text-gray-500 dark:text-gray-400">
								{g.count} {$i18n.t('receipts')} · {money(g.total, g.currency || 'USD')}
								{#if g.earliest_date && g.latest_date}
									<span class="text-gray-400"
										>· {g.earliest_date} → {g.latest_date}</span
									>
								{/if}
							</div>
						</div>
						<button
							class="ml-2 px-2.5 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-60 whitespace-nowrap"
							disabled={generating}
							on:click={() => generateForEmployee(g)}
						>
							{$i18n.t('Generate')}
						</button>
					</div>
				{/each}
			</div>
		</div>
	{/if}

	<!-- Filters -->
	<div
		class="grid grid-cols-2 md:grid-cols-5 gap-2 p-3 rounded-lg bg-gray-50 dark:bg-gray-850 border border-gray-200 dark:border-gray-800"
	>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>{$i18n.t('Status')}</label
			>
			<select
				bind:value={statusFilter}
				on:change={load}
				class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
			>
				<option value="">{$i18n.t('Any')}</option>
				<option value="draft">{$i18n.t('Draft')}</option>
				<option value="submitted">{$i18n.t('Submitted')}</option>
				<option value="approved">{$i18n.t('Approved')}</option>
				<option value="paid">{$i18n.t('Paid')}</option>
				<option value="rejected">{$i18n.t('Rejected')}</option>
			</select>
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>{$i18n.t('Employee')}</label
			>
			<select
				bind:value={employeeFilter}
				on:change={load}
				class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
			>
				<option value="">{$i18n.t('Any')}</option>
				{#each employees as emp}
					<option value={emp.id}>{emp.full_name}</option>
				{/each}
			</select>
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>{$i18n.t('From')}</label
			>
			<input
				type="date"
				bind:value={dateFrom}
				on:change={load}
				class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
			/>
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>{$i18n.t('To')}</label
			>
			<input
				type="date"
				bind:value={dateTo}
				on:change={load}
				class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
			/>
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>{$i18n.t('Search')}</label
			>
			<input
				type="text"
				bind:value={search}
				on:input={load}
				placeholder="ES-2026-0001"
				class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
			/>
		</div>
	</div>

	{#if showGenerate}
		<div
			class="p-4 rounded-lg border border-blue-200 dark:border-blue-900/50 bg-blue-50 dark:bg-blue-950/20 space-y-3"
		>
			<h3 class="text-sm font-semibold dark:text-gray-200">{$i18n.t('Generate Sheet')}</h3>
			<div class="grid grid-cols-2 md:grid-cols-4 gap-2">
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Employee')}</label
					>
					<select
						bind:value={genForm.employee_id}
						on:change={() => (candidatePreview = null)}
						class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					>
						<option value={0}>{$i18n.t('Select...')}</option>
						{#each employees as emp}
							<option value={emp.id}>{emp.full_name}</option>
						{/each}
					</select>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('From')}</label
					>
					<input
						type="date"
						bind:value={genForm.period_start}
						on:change={() => (candidatePreview = null)}
						class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('To')}</label
					>
					<input
						type="date"
						bind:value={genForm.period_end}
						on:change={() => (candidatePreview = null)}
						class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
				<div>
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
						>{$i18n.t('Title (optional)')}</label
					>
					<input
						type="text"
						bind:value={genForm.title}
						class="w-full text-sm rounded-lg px-2 py-1 bg-white dark:bg-gray-900 dark:text-gray-200 border border-gray-200 dark:border-gray-700 outline-hidden"
					/>
				</div>
			</div>
			<label class="flex items-center gap-1.5 text-xs text-gray-600 dark:text-gray-400">
				<input type="checkbox" bind:checked={genForm.include_uncategorized} class="rounded" />
				{$i18n.t('Include uncategorized invoices')}
			</label>
			<div class="flex items-center gap-3">
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
					on:click={previewCandidates}
				>
					{$i18n.t('Preview Candidates')}
				</button>
				{#if candidatePreview}
					<span class="text-xs text-gray-600 dark:text-gray-400">
						{candidatePreview.total}
						{$i18n.t('invoice(s) will be included')}
					</span>
				{/if}
				<div class="flex-1"></div>
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
					on:click={() => (showGenerate = false)}
				>
					{$i18n.t('Cancel')}
				</button>
				<button
					class="px-4 py-1.5 text-sm font-medium rounded-lg bg-green-600 text-white hover:bg-green-700 transition disabled:opacity-60"
					disabled={generating}
					on:click={handleGenerate}
				>
					{generating ? $i18n.t('Generating...') : $i18n.t('Generate')}
				</button>
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="flex justify-center my-8"><Spinner className="size-5" /></div>
	{:else if sheets.length === 0}
		<div class="text-sm text-gray-400 italic py-6 text-center">
			{$i18n.t('No expense sheets yet. Click New Sheet to create one.')}
		</div>
	{:else}
		<div class="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
			<table class="w-full text-sm">
				<thead class="text-xs text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-850">
					<tr>
						<th class="text-left py-2 px-3">{$i18n.t('Reference')}</th>
						<th class="text-left py-2 px-3">{$i18n.t('Employee')}</th>
						<th class="text-left py-2 px-3">{$i18n.t('Period')}</th>
						<th class="text-right py-2 px-3">{$i18n.t('Total')}</th>
						<th class="text-center py-2 px-3">{$i18n.t('Lines')}</th>
						<th class="text-left py-2 px-3">{$i18n.t('Status')}</th>
						<th class="text-left py-2 px-3">{$i18n.t('Created')}</th>
					</tr>
				</thead>
				<tbody>
					{#each sheets as s}
						<!-- svelte-ignore a11y-click-events-have-key-events -->
						<!-- svelte-ignore a11y-no-static-element-interactions -->
						<tr
							class="border-t border-gray-100 dark:border-gray-850 hover:bg-gray-50 dark:hover:bg-gray-900/50 cursor-pointer"
							on:click={() => goto(`/accounting/company/${companyId}/expenses/${s.id}`)}
						>
							<td class="py-2 px-3 font-mono text-xs dark:text-gray-200">{s.reference}</td>
							<td class="py-2 px-3 dark:text-gray-200">
								{s.employee_name ?? `#${s.employee_id}`}
								{#if s.employee_code}
									<span class="text-xs text-gray-400 ml-1">({s.employee_code})</span>
								{/if}
							</td>
							<td class="py-2 px-3 text-gray-600 dark:text-gray-400">
								{s.period_start} → {s.period_end}
							</td>
							<td class="py-2 px-3 text-right font-medium dark:text-gray-200"
								>{money(s.total_amount, s.currency)}</td
							>
							<td class="py-2 px-3 text-center text-gray-500">{s.line_count}</td>
							<td class="py-2 px-3">
								<ExpenseSheetStatusBadge status={s.status} />
							</td>
							<td class="py-2 px-3 text-xs text-gray-500"
								>{new Date(s.created_at).toLocaleDateString()}</td
							>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
