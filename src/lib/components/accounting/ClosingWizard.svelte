<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		getClosingChecklist,
		yearEndClose,
		generateDepreciation,
		getPeriods
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	// State
	let loading = false;
	let checking = false;
	let closing = false;
	let generatingYearEnd = false;
	let checklist: any = null;

	// Month selector
	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; from: string; to: string; fiscalStart: string }> = [];

	// Year-end preview
	let showYearEndPreview = false;
	let yearEndPreview: any = null;

	// ─── Helpers ────────────────────────────────────────────────────────────────

	function buildMonthOptions(periods: any[]) {
		const options: typeof monthOptions = [];
		for (const p of periods) {
			const start = new Date(p.start_date);
			const end = new Date(p.end_date);
			const fiscalStart = p.start_date;
			let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
			while (cursor <= end) {
				const y = cursor.getFullYear();
				const m = cursor.getMonth();
				const from = `${y}-${String(m + 1).padStart(2, '0')}-01`;
				const lastDay = new Date(y, m + 1, 0).getDate();
				const to = `${y}-${String(m + 1).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
				const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				options.push({ value: `${y}-${String(m + 1).padStart(2, '0')}`, label, from, to, fiscalStart });
				cursor = new Date(y, m + 1, 1);
			}
		}
		const seen = new Map<string, (typeof options)[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
	}

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '0.00';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	// ─── Data loading ───────────────────────────────────────────────────────────

	onMount(async () => {
		loading = true;
		try {
			const res = await getPeriods({ company_id: companyId });
			const periods = res.periods ?? res ?? [];
			monthOptions = buildMonthOptions(periods);

			const now = new Date();
			const curKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
			const match = monthOptions.find((o) => o.value === curKey);
			if (match) {
				selectedMonth = match.value;
			} else if (monthOptions.length > 0) {
				selectedMonth = monthOptions[0].value;
			}
		} catch (err) {
			console.error('Failed to load periods:', err);
		}
		loading = false;
	});

	// ─── Run Checklist ──────────────────────────────────────────────────────────

	const handleRunChecklist = async () => {
		const opt = monthOptions.find((o) => o.value === selectedMonth);
		if (!opt) {
			toast.error($i18n.t('Please select a period'));
			return;
		}
		checking = true;
		checklist = null;
		try {
			checklist = await getClosingChecklist({
				company_id: companyId,
				period_start: opt.from,
				period_end: opt.to
			});
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to run checklist') + ': ' + msg);
		}
		checking = false;
	};

	// ─── Action handlers ────────────────────────────────────────────────────────

	const handleCheckAction = async (check: any) => {
		if (check.action === 'generate_depreciation') {
			const opt = monthOptions.find((o) => o.value === selectedMonth);
			if (!opt) return;
			try {
				const result = await generateDepreciation(companyId, opt.to);
				const count = result?.entries_created ?? result?.count ?? 0;
				toast.success(`${count} ${$i18n.t('depreciation entries created as Draft')}`);
				await handleRunChecklist();
			} catch (err: any) {
				const msg = err?.detail ?? err?.message ?? String(err);
				toast.error($i18n.t('Failed to generate depreciation') + ': ' + msg);
			}
		} else if (check.action === 'generate_tax_entry') {
			toast.info($i18n.t('Please use the Tax Declaration tab to generate tax entries'));
		}
	};

	// ─── Close Period ───────────────────────────────────────────────────────────

	const handleClosePeriod = async () => {
		// Close is handled via the Periods management; here we just show a message
		toast.info($i18n.t('Use the Accounting Periods section in Settings to close the period'));
	};

	// ─── Year-End Close ─────────────────────────────────────────────────────────

	$: isDecember = selectedMonth.endsWith('-12');
	$: selectedOpt = monthOptions.find((o) => o.value === selectedMonth);

	const handleYearEndClose = async () => {
		if (!selectedOpt) return;
		generatingYearEnd = true;
		try {
			const result = await yearEndClose(companyId, {
				fiscal_year_start: selectedOpt.fiscalStart,
				fiscal_year_end: selectedOpt.to
			});
			yearEndPreview = result;
			const txId = result?.id ?? result?.transaction_id ?? '';
			toast.success(
				$i18n.t('Year-end closing entry created as Draft') + (txId ? ` (ID: ${txId})` : '')
			);
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create year-end closing entry') + ': ' + msg);
		}
		generatingYearEnd = false;
	};

	const statusIcon = (status: string): string => {
		if (status === 'ok' || status === 'pass') return '\u2705';
		if (status === 'warning' || status === 'warn') return '\u26A0\uFE0F';
		return '\u274C';
	};
</script>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">
				{$i18n.t('Closing Wizard')}
			</div>
		</div>
	</div>

	<!-- Description -->
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t('Run pre-closing checks and generate period/year-end closing entries.')}
	</div>

	{#if loading}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else}
		<!-- Month selector + Run Checklist -->
		<div class="flex flex-wrap gap-3 items-end mb-4">
			<div>
				<label
					for="closing-month"
					class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
				>
					{$i18n.t('Period')}
				</label>
				{#if monthOptions.length > 0}
					<select
						id="closing-month"
						bind:value={selectedMonth}
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
				disabled={!selectedMonth || checking}
				on:click={handleRunChecklist}
			>
				{$i18n.t('Run Checklist')}
			</button>
		</div>

		<!-- AI Loading Banner -->
		{#if checking}
			<div
				class="relative overflow-hidden rounded-xl border border-blue-200/50 dark:border-blue-800/30 bg-blue-50 dark:bg-blue-900/20 p-4 mb-4"
			>
				<div
					class="absolute top-0 left-0 h-1 bg-blue-500 animate-pulse"
					style="width: 100%;"
				/>
				<div class="flex items-center gap-3">
					<Spinner className="size-5 text-blue-600 dark:text-blue-400" />
					<span class="text-sm font-medium text-blue-700 dark:text-blue-300">
						{$i18n.t('AI is reviewing period...')}
					</span>
				</div>
			</div>
		{/if}

		<!-- Checklist Results -->
		{#if checklist && !checking}
			<div
				class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-4"
			>
				<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850">
					<div class="text-sm font-medium dark:text-gray-200">
						{$i18n.t('Pre-Closing Checklist')}
					</div>
				</div>

				<!-- Blockers -->
				{#if checklist.blockers?.length > 0}
					<div class="px-4 py-3 bg-red-50 dark:bg-red-900/20 border-b border-red-100 dark:border-red-800/30">
						<div class="text-xs font-medium text-red-700 dark:text-red-300 mb-1">
							{$i18n.t('Blockers')}
						</div>
						{#each checklist.blockers as blocker}
							<div class="text-xs text-red-600 dark:text-red-400">
								{blocker}
							</div>
						{/each}
					</div>
				{/if}

				<!-- Checks list -->
				<div class="divide-y divide-gray-100 dark:divide-gray-850">
					{#each checklist.checks ?? [] as check}
						<div class="px-4 py-3 flex items-start gap-3">
							<div class="text-base flex-shrink-0 mt-0.5">
								{statusIcon(check.status)}
							</div>
							<div class="flex-1 min-w-0">
								<div class="text-sm font-medium dark:text-gray-200">
									{check.label ?? check.name ?? ''}
								</div>
								{#if check.detail}
									<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
										{check.detail}
									</div>
								{/if}
							</div>
							{#if check.action && check.status !== 'ok' && check.status !== 'pass'}
								<button
									class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/20 dark:text-blue-300 dark:hover:bg-blue-900/40 transition flex-shrink-0"
									on:click={() => handleCheckAction(check)}
								>
									{check.action_label ?? $i18n.t('Fix')}
								</button>
							{/if}
						</div>
					{/each}
				</div>

				<!-- Footer: Close Period -->
				<div class="px-4 py-3 border-t border-gray-100 dark:border-gray-850 flex items-center gap-3">
					<button
						class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
						disabled={!(checklist.can_close ?? false) || closing}
						on:click={handleClosePeriod}
					>
						{$i18n.t('Close Period')}
					</button>
					{#if !(checklist.can_close ?? false)}
						<span class="text-xs text-gray-400 dark:text-gray-500">
							{$i18n.t('Resolve all blockers before closing')}
						</span>
					{/if}
				</div>
			</div>

			<!-- Year-End Section -->
			{#if isDecember}
				<div
					class="bg-white dark:bg-gray-900 rounded-xl border border-amber-200/50 dark:border-amber-800/30 mb-4"
				>
					<div class="px-4 py-3 border-b border-amber-100 dark:border-amber-800/30 bg-amber-50/50 dark:bg-amber-900/10">
						<div class="text-sm font-medium text-amber-800 dark:text-amber-300">
							{$i18n.t('Year-End Closing')}
						</div>
						<div class="text-xs text-amber-600 dark:text-amber-400 mt-0.5">
							{$i18n.t('Generate year-end closing entry to zero all revenue/expense accounts into retained earnings.')}
						</div>
					</div>

					{#if yearEndPreview?.lines?.length > 0}
						<div class="overflow-x-auto">
							<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
								<thead
									class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
								>
									<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
										<th class="px-3 py-2">{$i18n.t('Account Code')}</th>
										<th class="px-3 py-2">{$i18n.t('Description')}</th>
										<th class="px-3 py-2 text-right">{$i18n.t('Debit')}</th>
										<th class="px-3 py-2 text-right">{$i18n.t('Credit')}</th>
									</tr>
								</thead>
								<tbody>
									{#each yearEndPreview.lines as line}
										<tr
											class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
										>
											<td class="px-3 py-2 font-mono font-medium dark:text-gray-200">
												{line.account_code ?? ''}
											</td>
											<td class="px-3 py-2">
												{line.description ?? line.account_name ?? ''}
											</td>
											<td class="px-3 py-2 text-right font-mono">
												{line.debit ? fmt(line.debit) : ''}
											</td>
											<td class="px-3 py-2 text-right font-mono">
												{line.credit ? fmt(line.credit) : ''}
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					{/if}

					<div class="px-4 py-3 border-t border-amber-100 dark:border-amber-800/30">
						<button
							class="px-4 py-2 text-sm font-medium rounded-lg bg-amber-600 text-white hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600 transition disabled:opacity-50"
							disabled={generatingYearEnd}
							on:click={handleYearEndClose}
						>
							{generatingYearEnd
								? $i18n.t('Generating...')
								: $i18n.t('Generate Year-End Closing Entry')}
						</button>
					</div>
				</div>
			{/if}
		{/if}
	{/if}
</div>
