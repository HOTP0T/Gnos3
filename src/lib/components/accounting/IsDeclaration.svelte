<script lang="ts">
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import {
		computeCitDeclaration,
		saveTaxFiling,
		exportTaxWorksheet,
		createTaxEntry,
		getTaxConfig,
		getTaxFilings,
		getAccounts,
		recordProvisionalTaxPayment,
		type CitAdjustment
	} from '$lib/apis/accounting';
	import { fiscalYearStart, fiscalYearEnd, fiscalYearLabel } from '$lib/utils/fiscalYear';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import TaxFilingsList from './TaxFilingsList.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let companyId: number;

	// ── Country shape ──────────────────────────────────────────────────
	// The country config says how this tax works: its name (IS / 企业所得税 /
	// Profits Tax), flat or tiered rates, whether the tax office bills a
	// provisional prepayment, and the company's fiscal year.
	let cfg: any = null;
	$: citLabel = cfg?.cit_label || 'IS';
	$: provisional = !!cfg?.cit_provisional;
	$: fyStartMonth = Number(cfg?.fiscal_year_start_month ?? 1) || 1;

	let calculating = false;
	let saving = false;
	let creating = false;
	let periodStart = '';
	let periodEnd = '';
	let priorYearLosses = '0';
	// Blank = let the backend net off the CIT filings already paid inside the period.
	let citAlreadyPaid = '';
	// Blank = read the prepaid-tax account's balance from the ledger.
	let provisionalPaid = '';
	// The accountant's tax computation: nothing here is inferred.
	let adjustments: CitAdjustment[] = [];
	let result: any = null;
	let filingsList: any;
	let savedFilings: any[] = [];

	// Quick period choices: the last three fiscal years plus the current year to date.
	type Choice = { key: string; label: string; start: string; end: string };
	let choices: Choice[] = [];
	let choice = '';

	const today = () => new Date().toISOString().slice(0, 10);

	const buildChoices = () => {
		const t = today();
		const curStart = fiscalYearStart(t, fyStartMonth);
		const out: Choice[] = [];
		let start = curStart;
		for (let i = 0; i < 4; i++) {
			const end = fiscalYearEnd(start, fyStartMonth);
			const label = fiscalYearLabel(start, end);
			out.push({
				key: start,
				label: i === 0 ? `${$i18n.t('FY')} ${label} (${$i18n.t('year to date')})` : `${$i18n.t('FY')} ${label}`,
				start,
				end: i === 0 ? t : end
			});
			// previous fiscal year
			const [y, m] = start.split('-').map(Number);
			start = `${y - 1}-${String(m).padStart(2, '0')}-01`;
		}
		choices = out;
		// Default: the last completed year while its return is still being prepared
		// (up to 8 months after year-end), otherwise the current year to date.
		const prev = out[1];
		const monthsSincePrevEnd =
			(new Date(t).getFullYear() - new Date(prev.end).getFullYear()) * 12 +
			(new Date(t).getMonth() - new Date(prev.end).getMonth());
		const pick = cfg?.cit_frequency === 'yearly' && monthsSincePrevEnd <= 8 ? prev : out[0];
		choice = pick.key;
		periodStart = pick.start;
		periodEnd = pick.end;
	};

	const onChoice = () => {
		const c = choices.find((x) => x.key === choice);
		if (!c) return;
		periodStart = c.start;
		periodEnd = c.end;
		prefillFromFiling();
	};

	// A saved filing for the same period carries the adjustments typed last time.
	const prefillFromFiling = () => {
		const f = savedFilings.find((x) => x.period_start === periodStart && x.period_end === periodEnd);
		const d = f?.details;
		if (!d) return;
		if (Array.isArray(d.adjustments)) {
			adjustments = d.adjustments.map((a: any) => ({ label: a.label ?? '', amount: Number(a.amount) || 0, kind: a.kind === 'deduct' ? 'deduct' : 'add' }));
		}
		if (d.prior_year_losses) priorYearLosses = String(d.prior_year_losses);
		if (d.provisional_paid_source === 'manual') provisionalPaid = String(d.provisional_paid ?? '');
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};
	const pct = (r: any) => (r === null || r === undefined ? '—' : `${(Number(r) * 100).toFixed(2).replace(/\.?0+$/, '')}%`);

	// ── Bank accounts for the provisional-tax payment ──────────────────
	let accounts: any[] = [];
	$: parentIds = new Set(accounts.map((a: any) => a.parent_id).filter(Boolean));
	$: bankAccounts = accounts.filter((a: any) => !parentIds.has(a.id) && (!a.account_type || a.account_type === 'asset'));
	let provAmount = '';
	let provDate = '';
	let provBank: number | '' = '';
	let provRef = '';
	let recording = false;
	let showProvisional = false;

	onMount(async () => {
		try {
			cfg = await getTaxConfig(companyId);
		} catch {
			cfg = null;
		}
		try {
			savedFilings = await getTaxFilings({ company_id: companyId, tax_type: 'cit' });
		} catch {
			savedFilings = [];
		}
		try {
			const a = await getAccounts({ company_id: companyId, active: true });
			accounts = Array.isArray(a) ? a : (a?.items ?? a?.accounts ?? []);
		} catch {
			accounts = [];
		}
		buildChoices();
		prefillFromFiling();
		provDate = today();
	});

	// ── Adjustments editor ─────────────────────────────────────────────
	const PRESETS: Array<{ label: string; kind: 'add' | 'deduct' }> = [
		{ label: 'Depreciation per accounts', kind: 'add' },
		{ label: 'Non-deductible expenses (fines, private expenses)', kind: 'add' },
		{ label: 'Entertainment over the deductible limit', kind: 'add' },
		{ label: 'General provisions', kind: 'add' },
		{ label: 'Depreciation allowances (tax)', kind: 'deduct' },
		{ label: 'Bank interest income (exempt)', kind: 'deduct' },
		{ label: 'Dividend income (exempt)', kind: 'deduct' },
		{ label: 'Offshore / non-taxable income', kind: 'deduct' },
		{ label: 'Capital gains (not taxable)', kind: 'deduct' }
	];
	let preset = '';
	const addAdjustment = (kind: 'add' | 'deduct', label = '') => {
		adjustments = [...adjustments, { label, amount: 0, kind }];
	};
	const addPreset = () => {
		const p = PRESETS.find((x) => x.label === preset);
		if (p) addAdjustment(p.kind, $i18n.t(p.label));
		preset = '';
	};
	const removeAdjustment = (i: number) => {
		adjustments = adjustments.filter((_, idx) => idx !== i);
	};

	const payload = () => ({
		period_start: periodStart,
		period_end: periodEnd,
		prior_year_losses: parseFloat(priorYearLosses) || 0,
		cit_already_paid: citAlreadyPaid.trim() === '' ? undefined : parseFloat(citAlreadyPaid) || 0,
		provisional_paid: provisionalPaid.trim() === '' ? undefined : parseFloat(provisionalPaid) || 0,
		adjustments: adjustments.filter((a) => Number(a.amount)).map((a) => ({ ...a, amount: Number(a.amount) }))
	});

	const calc = async () => {
		if (!periodStart || !periodEnd) {
			toast.error($i18n.t('Select a period'));
			return;
		}
		calculating = true;
		result = null;
		try {
			result = await computeCitDeclaration(companyId, payload());
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to calculate')}: ${err?.detail ?? err}`);
		}
		calculating = false;
	};

	const filingDetails = () => ({ ...result, adjustments: payload().adjustments });

	const createEntry = async () => {
		if (!result?.suggested_entry) return;
		creating = true;
		try {
			const r = await createTaxEntry(companyId, {
				tax_type: 'cit',
				period_start: result.period_start,
				period_end: result.period_end,
				entry: result.suggested_entry,
				tax_amount: result.cit_due ?? 0,
				currency: result.currency,
				details: filingDetails()
			});
			toast.success(
				$i18n.t('Accrual entry created as Draft and linked to the filing') +
					(r?.transaction_id ? ` (ID: ${r.transaction_id})` : '')
			);
			filingsList?.reload();
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to create accrual entry')}: ${err?.detail ?? err}`);
		}
		creating = false;
	};

	const save = async () => {
		if (!result) return;
		saving = true;
		try {
			await saveTaxFiling(companyId, {
				tax_type: 'cit',
				period_start: result.period_start,
				period_end: result.period_end,
				tax_amount: result.cit_due ?? 0,
				currency: result.currency,
				details: filingDetails()
			});
			toast.success($i18n.t('Filing saved'));
			filingsList?.reload();
			savedFilings = await getTaxFilings({ company_id: companyId, tax_type: 'cit' });
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save filing')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	const doExport = async () => {
		if (!result) return;
		try {
			const p = payload();
			await exportTaxWorksheet({
				company_id: companyId,
				tax_type: 'cit',
				period_start: result.period_start,
				period_end: result.period_end,
				prior_year_losses: p.prior_year_losses,
				cit_already_paid: p.cit_already_paid,
				provisional_paid: p.provisional_paid,
				adjustments: p.adjustments
			});
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to export')}: ${err?.detail ?? err}`);
		}
	};

	const recordProvisional = async () => {
		const amount = parseFloat(provAmount);
		if (!amount || amount <= 0 || !provBank) {
			toast.error($i18n.t('Enter the amount and the bank account'));
			return;
		}
		recording = true;
		try {
			const r = await recordProvisionalTaxPayment(companyId, {
				amount,
				bank_account_id: Number(provBank),
				paid_date: provDate || undefined,
				reference: provRef || undefined
			});
			toast.success(
				$i18n.t('Provisional tax payment recorded') + (r?.entry_number ? ` (${r.entry_number})` : r?.entry_status === 'draft' ? ` (${$i18n.t('draft')})` : '')
			);
			provAmount = '';
			provRef = '';
			showProvisional = false;
			if (result) await calc();
		} catch (err: any) {
			toast.error(`${err?.detail ?? err}`);
		}
		recording = false;
	};

	const inputCls =
		'text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden';
	const lblCls = 'block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1';
	const rowCls = 'border-b border-gray-100 dark:border-gray-850';
	const btnPrimary =
		'px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50';
	const btnSecondary =
		'px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition disabled:opacity-50';
</script>

<div class="py-2">
	<div class="flex md:self-center text-lg font-medium px-0.5 gap-2 mb-1">
		<div class="flex-shrink-0 dark:text-gray-200">{citLabel}</div>
	</div>
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t(
			'Accounting profit comes from the ledger. Type the adjustments of the tax computation below (add-backs and deductions); rates and reductions come from Settings → Countries.'
		)}
	</div>

	<!-- Period + inputs -->
	<div class="flex flex-wrap gap-3 items-end mb-3">
		<div>
			<label class={lblCls} for="cit-fy">{$i18n.t('Fiscal year')}</label>
			<select id="cit-fy" bind:value={choice} on:change={onChoice} class={inputCls}>
				{#each choices as c}<option value={c.key}>{c.label}</option>{/each}
				<option value="">{$i18n.t('Custom period')}</option>
			</select>
		</div>
		<div>
			<label class={lblCls} for="cit-start">{$i18n.t('Period start')}</label>
			<input id="cit-start" type="date" bind:value={periodStart} on:change={() => (choice = '')} class={inputCls} />
		</div>
		<div>
			<label class={lblCls} for="cit-end">{$i18n.t('Period end')}</label>
			<input id="cit-end" type="date" bind:value={periodEnd} on:change={() => (choice = '')} class={inputCls} />
		</div>
		<div>
			<label class={lblCls} for="cit-losses">{$i18n.t('Losses brought forward')}</label>
			<input id="cit-losses" type="number" step="0.01" bind:value={priorYearLosses} class={inputCls} />
		</div>
		{#if provisional}
			<div>
				<label class={lblCls} for="cit-prov">{$i18n.t('Provisional tax paid for this year')}</label>
				<input
					id="cit-prov"
					type="number"
					step="0.01"
					bind:value={provisionalPaid}
					placeholder={$i18n.t('from the ledger')}
					class={inputCls}
				/>
			</div>
		{/if}
		<div>
			<label class={lblCls} for="cit-paid">{$i18n.t('{{tax}} already paid', { tax: citLabel })}</label>
			<input
				id="cit-paid"
				type="number"
				step="0.01"
				bind:value={citAlreadyPaid}
				placeholder={$i18n.t('from paid filings')}
				class={inputCls}
			/>
		</div>
	</div>

	<!-- Adjustments -->
	<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-3">
		<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 flex flex-wrap items-center justify-between gap-2">
			<div>
				<div class="text-sm font-medium dark:text-gray-200">{$i18n.t('Tax computation adjustments')}</div>
				<div class="text-[11px] text-gray-400 dark:text-gray-500">
					{$i18n.t('Add back what the tax office refuses (book depreciation, fines…); deduct what it does not tax (bank interest, dividends) and the depreciation allowances.')}
				</div>
			</div>
			<div class="flex items-center gap-2">
				<select bind:value={preset} on:change={addPreset} class="{inputCls} text-xs">
					<option value="">{$i18n.t('— add a usual line —')}</option>
					{#each PRESETS as p}<option value={p.label}>{p.kind === 'add' ? '+' : '−'} {$i18n.t(p.label)}</option>{/each}
				</select>
				<button class={btnSecondary} on:click={() => addAdjustment('add')}>+ {$i18n.t('Add-back')}</button>
				<button class={btnSecondary} on:click={() => addAdjustment('deduct')}>− {$i18n.t('Deduction')}</button>
			</div>
		</div>
		{#if adjustments.length === 0}
			<div class="px-4 py-3 text-xs text-gray-400 dark:text-gray-500">{$i18n.t('No adjustments — the accounting profit is used as is.')}</div>
		{:else}
			<table class="w-full text-sm text-left">
				<tbody class="text-gray-900 dark:text-gray-100">
					{#each adjustments as a, i}
						<tr class={rowCls}>
							<td class="px-3 py-1.5 w-28">
								<select bind:value={a.kind} class="{inputCls} text-xs py-1">
									<option value="add">{$i18n.t('Add back')}</option>
									<option value="deduct">{$i18n.t('Deduct')}</option>
								</select>
							</td>
							<td class="px-3 py-1.5">
								<input type="text" bind:value={a.label} placeholder={$i18n.t('Description')} class="{inputCls} w-full py-1" />
							</td>
							<td class="px-3 py-1.5 w-44 text-right">
								<input type="number" step="0.01" bind:value={a.amount} class="{inputCls} w-40 text-right py-1 font-mono" />
							</td>
							<td class="px-2 py-1.5 w-10 text-right">
								<button class="text-gray-400 hover:text-red-500 text-sm" title={$i18n.t('Remove')} on:click={() => removeAdjustment(i)}>✕</button>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</div>

	<div class="flex flex-wrap gap-2 items-center mb-4">
		<button
			class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
			disabled={calculating}
			on:click={calc}
		>
			{$i18n.t('Calculate')}
		</button>
		{#if provisional}
			<button class={btnSecondary} on:click={() => (showProvisional = !showProvisional)}>
				{$i18n.t('Record a provisional tax payment')}
			</button>
		{/if}
	</div>

	{#if provisional && showProvisional}
		<div class="bg-white dark:bg-gray-900 rounded-xl border border-blue-200/50 dark:border-blue-800/30 mb-4 p-4">
			<div class="text-sm font-medium dark:text-gray-200 mb-1">{$i18n.t('Provisional tax payment')}</div>
			<div class="text-[11px] text-gray-400 dark:text-gray-500 mb-3">
				{$i18n.t(
					"The tax office's demand note asks for next year's tax in advance (usually 75% then 25%). Record each instalment here from the note: it is booked as prepaid tax and netted at the year-end. The final balance of a saved return is paid from the filings list below."
				)}
			</div>
			<div class="flex flex-wrap gap-3 items-end">
				<div>
					<label class={lblCls} for="prov-amount">{$i18n.t('Amount')}</label>
					<input id="prov-amount" type="number" step="0.01" bind:value={provAmount} class={inputCls} />
				</div>
				<div>
					<label class={lblCls} for="prov-date">{$i18n.t('Paid on')}</label>
					<input id="prov-date" type="date" bind:value={provDate} class={inputCls} />
				</div>
				<div>
					<label class={lblCls} for="prov-bank">{$i18n.t('Bank account')}</label>
					<select id="prov-bank" bind:value={provBank} class={inputCls}>
						<option value="">{$i18n.t('— select —')}</option>
						{#each bankAccounts as b}<option value={b.id}>{b.code} - {b.name}</option>{/each}
					</select>
				</div>
				<div>
					<label class={lblCls} for="prov-ref">{$i18n.t('Reference')}</label>
					<input id="prov-ref" type="text" bind:value={provRef} placeholder={$i18n.t('demand note no.')} class={inputCls} />
				</div>
				<button class={btnPrimary} disabled={recording} on:click={recordProvisional}>
					{recording ? $i18n.t('Recording...') : $i18n.t('Record payment')}
				</button>
			</div>
		</div>
	{/if}

	{#if calculating}
		<div class="flex items-center gap-3 my-6">
			<Spinner className="size-5 text-blue-600 dark:text-blue-400" />
			<span class="text-sm text-blue-700 dark:text-blue-300">{$i18n.t('Calculating...')}</span>
		</div>
	{/if}

	{#if result && !calculating}
		<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-3 overflow-x-auto">
			<table class="w-full text-sm text-left">
				<tbody class="text-gray-900 dark:text-gray-100">
					<tr class={rowCls}><td class="px-4 py-2">{$i18n.t('Total Revenue')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.revenue)}</td></tr>
					<tr class={rowCls}><td class="px-4 py-2">{$i18n.t('Total Cost')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.cost)}</td></tr>
					<tr class="{rowCls} font-medium"><td class="px-4 py-2">{$i18n.t('Profit before tax per accounts')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.income_before_tax)}</td></tr>
					{#if result.booked_tax_added_back}
						<tr class={rowCls}><td class="px-4 py-2 pl-8">{$i18n.t('Add: income tax charged in the accounts')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.booked_tax_added_back)}</td></tr>
					{/if}
					{#each result.adjustments ?? [] as a}
						<tr class={rowCls}>
							<td class="px-4 py-2 pl-8">{a.kind === 'add' ? $i18n.t('Add') : $i18n.t('Less')}: {a.label}</td>
							<td class="px-4 py-2 text-right font-mono">{a.kind === 'add' ? '' : '−'}{fmt(a.amount)}</td>
						</tr>
					{/each}
					<tr class="{rowCls} font-medium"><td class="px-4 py-2">{$i18n.t('Adjusted profit')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.adjusted_profit)}</td></tr>
					<tr class={rowCls}><td class="px-4 py-2">{$i18n.t('Less: losses brought forward')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.prior_year_losses)}</td></tr>
					<tr class="{rowCls} font-semibold bg-gray-50 dark:bg-gray-800/40"><td class="px-4 py-2">{$i18n.t('Assessable profit')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.profit_to_declare)}</td></tr>

					{#if (result.cit_tiers ?? []).length > 1}
						{#each result.cit_tiers as t}
							<tr class={rowCls}>
								<td class="px-4 py-2 pl-8">
									{citLabel} @ {pct(t.rate)} {$i18n.t('on')} {fmt(t.base)}
									<span class="text-xs text-gray-400">({t.up_to !== null && t.up_to !== undefined ? $i18n.t('up to {{cap}}', { cap: fmt(t.up_to) }) : $i18n.t('remainder')})</span>
								</td>
								<td class="px-4 py-2 text-right font-mono">{fmt(t.tax)}</td>
							</tr>
						{/each}
						<tr class="{rowCls} font-medium"><td class="px-4 py-2">{citLabel} {$i18n.t('before reductions')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.cit)}</td></tr>
					{:else}
						<tr class={rowCls}><td class="px-4 py-2">{citLabel} ({pct(result.cit_rate)})</td><td class="px-4 py-2 text-right font-mono">{fmt(result.cit)}</td></tr>
					{/if}
					{#if result.preferential}
						<tr class={rowCls}><td class="px-4 py-2">{$i18n.t('Less: preferential exemption')} ({pct(result.cit_preferential_rate)})</td><td class="px-4 py-2 text-right font-mono">{fmt(result.preferential)}</td></tr>
					{/if}
					{#if result.reduction_rate}
						<tr class={rowCls}>
							<td class="px-4 py-2">
								{$i18n.t('Less: tax reduction')} ({pct(result.reduction_rate)}{result.reduction_cap !== null && result.reduction_cap !== undefined ? `, ${$i18n.t('capped at')} ${fmt(result.reduction_cap)}` : ''})
							</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(result.reduction)}</td>
						</tr>
					{/if}
					<tr class="{rowCls} font-semibold bg-gray-50 dark:bg-gray-800/40"><td class="px-4 py-2">{citLabel} {$i18n.t('for the period')}</td><td class="px-4 py-2 text-right font-mono">{fmt(result.tax_for_period)}</td></tr>
					{#if result.cit_provisional}
						<tr class={rowCls}>
							<td class="px-4 py-2">
								{$i18n.t('Less: provisional tax already paid')}
								<span class="text-xs text-gray-400">
									({result.provisional_paid_source === 'ledger' ? $i18n.t('from the ledger') : result.provisional_paid_source === 'manual' ? $i18n.t('typed') : $i18n.t('no prepaid-tax account mapped')})
								</span>
							</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(result.provisional_paid)}</td>
						</tr>
					{/if}
					{#if result.cit_already_paid || citAlreadyPaid.trim() !== ''}
						<tr class={rowCls}>
							<td class="px-4 py-2">
								{$i18n.t('Less: {{tax}} already paid', { tax: citLabel })}
								{#if citAlreadyPaid.trim() === ''}
									<span class="text-xs text-gray-400">({$i18n.t('paid filings in the period')}: {fmt(result.cit_paid_from_filings)})</span>
								{/if}
							</td>
							<td class="px-4 py-2 text-right font-mono">{fmt(result.cit_already_paid)}</td>
						</tr>
					{/if}
					<tr class="font-bold bg-blue-50/50 dark:bg-blue-900/20">
						<td class="px-4 py-2">{result.cit_due < 0 ? $i18n.t('Refund / carried forward') : $i18n.t('Balance to be paid')}</td>
						<td class="px-4 py-2 text-right font-mono">{fmt(Math.abs(result.cit_due))} {result.currency ?? ''}</td>
					</tr>
				</tbody>
			</table>
		</div>

		{#if (result.warnings ?? []).length > 0}
			<div class="bg-amber-50 dark:bg-amber-900/20 border border-amber-200/50 dark:border-amber-800/30 rounded-xl p-3 mb-3 text-xs text-amber-800 dark:text-amber-200 space-y-1">
				{#each result.warnings as w}<div>{w}</div>{/each}
			</div>
		{/if}

		{#if (result.entry_blockers ?? []).length > 0}
			<div class="bg-white dark:bg-gray-900 rounded-xl border border-red-200/50 dark:border-red-800/30 mb-3 p-4">
				<div class="text-sm font-medium dark:text-gray-200 mb-1">{$i18n.t('Accrual entry not available')}</div>
				<ul class="text-xs text-red-700 dark:text-red-300 list-disc pl-4 space-y-0.5">
					{#each result.entry_blockers as b}<li>{b}</li>{/each}
				</ul>
				<button class="mt-2 {btnSecondary}" on:click={() => dispatch('gotoAccounts')}>{$i18n.t('Open Tax accounts')}</button>
			</div>
		{:else if result.suggested_entry?.lines?.length > 0}
			<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-3">
				<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 text-sm font-medium dark:text-gray-200">
					{$i18n.t('Suggested accrual entry')} <span class="text-xs text-gray-400 font-normal">({result.suggested_entry.date})</span>
				</div>
				<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
					<thead class="text-xs font-bold uppercase bg-gray-100 dark:bg-gray-800">
						<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
							<th class="px-3 py-2">{$i18n.t('Account Code')}</th>
							<th class="px-3 py-2">{$i18n.t('Description')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Debit')}</th>
							<th class="px-3 py-2 text-right">{$i18n.t('Credit')}</th>
						</tr>
					</thead>
					<tbody>
						{#each result.suggested_entry.lines as line}
							<tr class="{rowCls} text-xs">
								<td class="px-3 py-2 font-mono font-medium">{line.account_code} <span class="font-sans text-gray-500">{line.account_name ?? ''}</span></td>
								<td class="px-3 py-2">{line.description ?? ''}</td>
								<td class="px-3 py-2 text-right font-mono">{line.debit ? fmt(line.debit) : ''}</td>
								<td class="px-3 py-2 text-right font-mono">{line.credit ? fmt(line.credit) : ''}</td>
							</tr>
						{/each}
					</tbody>
				</table>
			</div>
		{/if}

		<div class="flex flex-wrap gap-2 mb-2">
			<button class={btnPrimary} disabled={saving} on:click={save}>
				{saving ? $i18n.t('Saving...') : $i18n.t('Save as Filing')}
			</button>
			{#if result.suggested_entry?.lines?.length > 0}
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
					disabled={creating}
					on:click={createEntry}
				>
					{creating ? $i18n.t('Creating...') : $i18n.t('Create Accrual Entry (Draft)')}
				</button>
			{/if}
			<button class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition" on:click={doExport}>
				{$i18n.t('Export Excel')}
			</button>
		</div>
	{/if}

	<TaxFilingsList {companyId} taxType="cit" bind:this={filingsList} />
</div>
