<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { buildMonthOptions, type MonthOption } from '$lib/utils/fiscalYear';
	import { toast } from 'svelte-sonner';

	import {
		getClosingChecklist,
		yearEndClose,
		carryForwardBalances,
		generateDepreciation,
		getPeriods,
		getCompany,
		bulkPostTransactions,
		autoMatchBankStatements,
		getBankAccounts,
		closePeriod
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	// State
	let loading = false;
	let checking = false;
	let closing = false;
	let generatingYearEnd = false;
	let carryingForward = false;
	let checklist: any = null;

	// Month selector
	let selectedMonth = '';
	let monthOptions: MonthOption[] = [];

	// Year-end preview
	let showYearEndPreview = false;
	let yearEndPreview: any = null;

	// ─── Helpers ────────────────────────────────────────────────────────────────

	let fiscalStartMonth = 1;
	let currency = '';

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '0.00';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	// The checklist API speaks English. The base steps are rebuilt here from their
	// stable `step` + `count` so the wizard follows the UI language; anything else
	// (country-specific checks, free-text details) goes through t() and falls back
	// to the API text when no translation exists.
	const checkLabel = (check: any): string => $i18n.t(check.label ?? check.name ?? '');

	const checkDetail = (check: any): string => {
		const n = Number(check.count ?? 0);
		switch (check.step) {
			case 'draft_entries':
				return n > 0
					? $i18n.t('{{count}} draft entries', { count: n })
					: $i18n.t('All entries are posted');
			case 'bank_reconciliation':
				if (n > 0) return $i18n.t('{{count}} unmatched bank lines', { count: n });
				break;
			case 'trial_balance':
				if (check.status !== 'ok' && check.difference != null) {
					return $i18n.t('Unbalanced: difference of {{amount}}', {
						amount: `${fmt(check.difference)} ${currency}`.trim()
					});
				}
				break;
		}
		return check.detail ? $i18n.t(check.detail) : '';
	};

	const checkActionLabel = (check: any): string =>
		check.action_label ? $i18n.t(check.action_label) : $i18n.t('Fix');

	// Checks that stop the period from closing (`blocking` from the API); the
	// API's `blockers` strings stay the fallback for older payloads.
	const blockingChecks = (list: any): any[] =>
		(list?.checks ?? []).filter((c: any) => c.blocking);

	// ─── Data loading ───────────────────────────────────────────────────────────

	onMount(async () => {
		loading = true;
		try {
			const [res, co] = await Promise.all([getPeriods({ company_id: companyId }), getCompany(companyId).catch(() => null)]);
			const periods = res.periods ?? res ?? [];
			fiscalStartMonth = Number(co?.fiscal_year_start_month ?? 1) || 1;
			currency = co?.currency ?? '';
			monthOptions = buildMonthOptions(periods, fiscalStartMonth);

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
		const opt = monthOptions.find((o) => o.value === selectedMonth);
		if (!opt) return;

		if (check.action === 'bulk_post_drafts' || check.step === 'draft_entries') {
			try {
				const result = await bulkPostTransactions({
					company_id: companyId,
					period_start: opt.from,
					period_end: opt.to
				});
				toast.success(`${result?.posted ?? 0} ${$i18n.t('entries posted')}${result?.skipped ? `, ${result.skipped} ${$i18n.t('skipped')}` : ''}`);
				await handleRunChecklist();
			} catch (err: any) {
				toast.error(err?.detail ?? `${err}`);
			}
		} else if (check.action === 'auto_reconcile' || check.step === 'bank_reconciliation') {
			try {
				const banks = await getBankAccounts(companyId);
				const bankList = banks?.accounts ?? banks ?? [];
				let totalMatched = 0;
				for (const ba of bankList) {
					const result = await autoMatchBankStatements(ba.id);
					totalMatched += result?.matched ?? 0;
				}
				toast.success(`${totalMatched} ${$i18n.t('bank lines matched')}`);
				await handleRunChecklist();
			} catch (err: any) {
				toast.error(err?.detail ?? `${err}`);
			}
		} else if (check.action === 'generate_depreciation') {
			try {
				const result = await generateDepreciation(companyId, opt.to);
				const count = result?.entries_created ?? result?.count ?? 0;
				toast.success(`${count} ${$i18n.t('depreciation entries created as Draft')}`);
				await handleRunChecklist();
			} catch (err: any) {
				toast.error(err?.detail ?? `${err}`);
			}
		} else if (check.action === 'generate_tax_entry') {
			toast.info($i18n.t('Please use the Tax Declaration tab to generate tax entries'));
		}
	};

	// ─── Close Period ───────────────────────────────────────────────────────────

	const handleClosePeriod = async () => {
		if (!checklist?.period_id) {
			toast.info($i18n.t('Use the Accounting Periods section in Settings to close the period'));
			return;
		}
		closing = true;
		try {
			await closePeriod(checklist.period_id);
			toast.success($i18n.t('Period closed'));
			await handleRunChecklist();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		closing = false;
	};

	// ─── Year-End Close ─────────────────────────────────────────────────────────

	$: selectedOpt = monthOptions.find((o) => o.value === selectedMonth);
	// The year-end close is offered in the last month of the company's fiscal year (December, or March for an April – March year).
	$: isFiscalYearEnd = !!selectedOpt?.isFiscalYearEnd;

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

	const handleCarryForward = async () => {
		if (!selectedOpt) return;
		// Opening date = day after the fiscal year end (start of next fiscal year).
		const close = new Date(selectedOpt.to + 'T00:00:00');
		const next = new Date(close);
		next.setDate(next.getDate() + 1);
		const openingDate = `${next.getFullYear()}-${String(next.getMonth() + 1).padStart(2, '0')}-${String(next.getDate()).padStart(2, '0')}`;
		carryingForward = true;
		try {
			const result = await carryForwardBalances(companyId, {
				closing_date: selectedOpt.to,
				opening_date: openingDate
			});
			const txId = result?.transaction_id ?? result?.id ?? '';
			if (!txId) {
				// Nothing to carry: the result account is flat (year-end entry not posted yet).
				toast.warning(result?.message ?? $i18n.t('Nothing to carry forward'));
			} else {
				toast.success(
					$i18n.t('Report à nouveau created as a draft — review it in Entries and post it') +
						` — ${result.amount} ${result.from_account} → ${result.to_account} (ID: ${txId})`
				);
			}
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create report à nouveau') + ': ' + msg);
		}
		carryingForward = false;
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
						{#if blockingChecks(checklist).length > 0}
							{#each blockingChecks(checklist) as check}
								<div class="text-xs text-red-600 dark:text-red-400">
									{checkLabel(check)}{checkDetail(check) ? ` — ${checkDetail(check)}` : ''}
								</div>
							{/each}
						{:else}
							{#each checklist.blockers as blocker}
								<div class="text-xs text-red-600 dark:text-red-400">
									{$i18n.t(blocker)}
								</div>
							{/each}
						{/if}
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
									{checkLabel(check)}
								</div>
								{#if checkDetail(check)}
									<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
										{checkDetail(check)}
									</div>
								{/if}
							</div>
							{#if check.action && check.status !== 'ok' && check.status !== 'pass'}
								<button
									class="px-3 py-1 text-xs font-medium rounded-lg bg-blue-50 text-blue-700 hover:bg-blue-100 dark:bg-blue-900/20 dark:text-blue-300 dark:hover:bg-blue-900/40 transition flex-shrink-0"
									on:click={() => handleCheckAction(check)}
								>
									{checkActionLabel(check)}
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
			{#if isFiscalYearEnd}
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

					<div class="px-4 py-3 border-t border-amber-100 dark:border-amber-800/30 flex flex-wrap items-center gap-3">
						<button
							class="px-4 py-2 text-sm font-medium rounded-lg bg-amber-600 text-white hover:bg-amber-700 dark:bg-amber-500 dark:hover:bg-amber-600 transition disabled:opacity-50"
							disabled={generatingYearEnd}
							on:click={handleYearEndClose}
						>
							{generatingYearEnd
								? $i18n.t('Generating...')
								: $i18n.t('Generate Year-End Closing Entry')}
						</button>
						<button
							class="px-4 py-2 text-sm font-medium rounded-lg border border-amber-500 text-amber-700 hover:bg-amber-50 dark:text-amber-300 dark:hover:bg-amber-900/20 transition disabled:opacity-50"
							disabled={carryingForward}
							on:click={handleCarryForward}
							title={$i18n.t(
								'Creates the report à nouveau as a draft: it moves the closed result into retained earnings on the first day of the next year. You review and post it yourself. Balance-sheet accounts already open at their closing balance, so they are not re-posted. Run this after posting the year-end closing entry.'
							)}
						>
							{carryingForward
								? $i18n.t('Creating...')
								: $i18n.t('Create Report à Nouveau draft (result → retained earnings)')}
						</button>
					</div>
				</div>
			{/if}
		{/if}
	{/if}
</div>
