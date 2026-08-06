<script lang="ts">
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import {
		getAPAging,
		getARAging,
		exportAPAging,
		exportARAging,
		getTransactionInvoices,
		setTransactionInvoices,
		getUnmatchedBankLines,
		matchBankStatement,
		type AgingParams
	} from '$lib/apis/accounting';
	import K4miDocLink from '$lib/components/common/K4miDocLink.svelte';
	import InvoiceSelector from '$lib/components/accounting/InvoiceSelector.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import type { Writable } from 'svelte/store';
	import { convertAmount } from '$lib/utils/currency';
	import ReportAmount from '$lib/components/accounting/ReportAmount.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;
	export let reportType: 'ap' | 'ar' = 'ap';

	let loading = false;
	let exporting = false;
	let rawData: any = null;
	let asOf = new Date().toISOString().slice(0, 10);

	// Display-currency conversion (company-wide selector). Aging is in base currency;
	// scale the summary buckets + per-row balances by the base→display rate as of `asOf`.
	const displayCurrency = getContext<Writable<string>>('displayCurrency');
	const exchangeRates = getContext<Writable<any[]>>('exchangeRates');
	const companyCurrency = getContext<Writable<string>>('companyCurrency');
	$: baseCcy = rawData?.currency || $companyCurrency || 'EUR';
	$: converting = !!($displayCurrency && baseCcy && $displayCurrency !== baseCcy);
	$: fx = converting && rawData
		? convertAmount(1, baseCcy, $displayCurrency, $exchangeRates ?? [], asOf)
		: null;
	$: factor = converting ? (fx?.hasRate ? fx.rate : null) : 1;
	$: noRate = converting && !fx?.hasRate;
	$: displayCcy = converting ? $displayCurrency : baseCcy;
	$: fxProps = { factor, converting, displayCcy, baseCcy, zeroText: '—' };
	$: data = rawData;

	// Filters
	let q = '';
	let bucket = '';
	let minBalance: number | null = null;
	let maxBalance: number | null = null;
	// Sort
	let sortBy: 'name' | 'balance' | 'days' | 'bucket' = 'balance';
	let sortDir: 'asc' | 'desc' = 'desc';

	// Drill-down modal
	let modalRow: any = null;

	// Invoice linking (from the drill-down, for journal items)
	let showInvoiceSelector = false;
	let linkTargetItem: any = null;
	let linking = false;

	const openLinkPicker = (item: any) => {
		linkTargetItem = item;
		showInvoiceSelector = true;
	};

	const handleLinkSelect = async (e: CustomEvent) => {
		const invoice = e.detail;
		if (!linkTargetItem?.transaction_id || !invoice?.id) return;
		linking = true;
		try {
			const existing = (linkTargetItem.linked_invoices ?? []).map((li: any) => li.invoice_id);
			const next = Array.from(new Set([...existing, invoice.id]));
			await setTransactionInvoices(linkTargetItem.transaction_id, next);
			toast.success($i18n.t('Invoice linked'));
			await refreshModal();
		} catch (err) {
			toast.error(`${err}`);
		}
		linking = false;
	};

	const unlinkInvoice = async (item: any, invoiceId: number) => {
		linking = true;
		try {
			const next = (item.linked_invoices ?? [])
				.map((li: any) => li.invoice_id)
				.filter((id: number) => id !== invoiceId);
			await setTransactionInvoices(item.transaction_id, next);
			toast.success($i18n.t('Invoice unlinked'));
			await refreshModal();
		} catch (err) {
			toast.error(`${err}`);
		}
		linking = false;
	};

	// After a link/match change, reload the report and re-point the open modal at the fresh row.
	const refreshModal = async () => {
		const party = modalRow?.vendor_or_customer;
		await load();
		if (party && data?.rows) {
			modalRow = data.rows.find((r: any) => r.vendor_or_customer === party) ?? null;
		}
	};

	// Reconcile a journal item to a bank statement line (closes it in the aging).
	let reconcileItem: any = null;
	let bankCandidates: any[] = [];
	let loadingCandidates = false;

	const openReconcile = async (item: any) => {
		reconcileItem = reconcileItem?.transaction_id === item.transaction_id ? null : item;
		bankCandidates = [];
		if (!reconcileItem) return;
		loadingCandidates = true;
		try {
			// Prefer exact-amount lines; the match endpoint requires the amounts to balance.
			bankCandidates = await getUnmatchedBankLines(companyId, {
				amount: Number(item.balance),
				tolerance: 0.01,
				limit: 25
			});
		} catch (err) {
			toast.error(`${err}`);
		}
		loadingCandidates = false;
	};

	const doReconcile = async (line: any) => {
		if (!reconcileItem?.transaction_id) return;
		linking = true;
		try {
			await matchBankStatement(line.id, reconcileItem.transaction_id);
			toast.success($i18n.t('Reconciled to bank line'));
			reconcileItem = null;
			bankCandidates = [];
			await refreshModal();
		} catch (err) {
			toast.error(`${err}`);
		}
		linking = false;
	};

	const entityLabel = reportType === 'ap' ? 'Vendor' : 'Customer';
	const title = reportType === 'ap' ? 'Accounts Payable Aging' : 'Accounts Receivable Aging';

	const buildParams = (): AgingParams => ({
		company_id: companyId,
		as_of: asOf,
		q: q || undefined,
		bucket: bucket || undefined,
		min_balance: minBalance ?? undefined,
		max_balance: maxBalance ?? undefined,
		sort_by: sortBy,
		sort_dir: sortDir
	});

	const load = async () => {
		loading = true;
		try {
			const fn = reportType === 'ap' ? getAPAging : getARAging;
			rawData = await fn(buildParams());
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const doExport = async () => {
		exporting = true;
		try {
			await (reportType === 'ap' ? exportAPAging : exportARAging)(buildParams());
		} catch (err) {
			toast.error(`${err}`);
		}
		exporting = false;
	};

	// Reload when sort or the select/number filters change (only if we've loaded once).
	const reloadIfLoaded = () => {
		if (data) load();
	};

	// Debounced reload for the free-text search.
	let qTimer: any;
	const onSearchInput = () => {
		clearTimeout(qTimer);
		qTimer = setTimeout(() => {
			if (data) load();
		}, 350);
	};

	const setSort = (col: 'name' | 'balance' | 'days' | 'bucket') => {
		if (sortBy === col) {
			sortDir = sortDir === 'asc' ? 'desc' : 'asc';
		} else {
			sortBy = col;
			sortDir = col === 'name' ? 'asc' : 'desc';
		}
		reloadIfLoaded();
	};

	const sortIcon = (col: string) => (sortBy === col ? (sortDir === 'asc' ? '▲' : '▼') : '');

	const clearFilters = () => {
		q = '';
		bucket = '';
		minBalance = null;
		maxBalance = null;
		reloadIfLoaded();
	};

	const toggleBucket = (b: string) => {
		bucket = bucket === b ? '' : b;
		reloadIfLoaded();
	};

	$: hasFilters = q || bucket || minBalance != null || maxBalance != null;

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '—';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const bucketColor = (b: string) => {
		switch (b) {
			case 'current':
				return 'text-green-600 dark:text-green-400';
			case '31-60':
				return 'text-yellow-600 dark:text-yellow-400';
			case '61-90':
				return 'text-orange-600 dark:text-orange-400';
			case '90+':
				return 'text-red-600 dark:text-red-400';
			default:
				return '';
		}
	};
</script>

<div class="space-y-3">
	<!-- Controls -->
	<div class="flex flex-wrap gap-3 items-end">
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('As of')}</label>
			<input
				type="date"
				bind:value={asOf}
				on:change={reloadIfLoaded}
				class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
			/>
		</div>
		<div class="grow min-w-[160px]">
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Search')} {$i18n.t(entityLabel)}</label>
			<input
				type="text"
				bind:value={q}
				on:input={onSearchInput}
				placeholder={$i18n.t('Name contains…')}
				class="w-full text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
			/>
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Min balance')}</label>
			<input type="number" bind:value={minBalance} on:change={reloadIfLoaded} class="w-28 text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Max balance')}</label>
			<input type="number" bind:value={maxBalance} on:change={reloadIfLoaded} class="w-28 text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
		</div>
		<button class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition" on:click={load}>
			{$i18n.t('Generate')}
		</button>
		{#if data}
			<button
				class="px-3 py-1.5 text-sm font-medium rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-850 transition inline-flex items-center gap-1.5 disabled:opacity-50"
				on:click={doExport}
				disabled={exporting}
			>
				{#if exporting}<Spinner className="size-3.5" />{/if}
				{$i18n.t('Export Excel')}
			</button>
		{/if}
		{#if hasFilters}
			<button class="px-3 py-1.5 text-sm rounded-lg text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 transition" on:click={clearFilters}>
				{$i18n.t('Clear filters')}
			</button>
		{/if}
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if data}
		{#if displayCcy}
			<div class="text-[11px] text-gray-400 dark:text-gray-500 px-0.5 mb-1">
				{$i18n.t('Currency')}: {displayCcy}
				{#if noRate}<span class="text-amber-600 dark:text-amber-400">({$i18n.t('no rate for')} {$displayCurrency}, {$i18n.t('showing')} {baseCcy})</span>{/if}
			</div>
		{/if}
		<!-- Summary cards (click to filter by bucket) -->
		<div class="grid grid-cols-5 gap-2">
			{#each [{ k: 'current', label: $i18n.t('Current'), val: data.summary.current, bg: 'bg-green-50 dark:bg-green-900/20', fg: 'text-green-700 dark:text-green-400' }, { k: '31-60', label: '31–60', val: data.summary.days_31_60, bg: 'bg-yellow-50 dark:bg-yellow-900/20', fg: 'text-yellow-700 dark:text-yellow-400' }, { k: '61-90', label: '61–90', val: data.summary.days_61_90, bg: 'bg-orange-50 dark:bg-orange-900/20', fg: 'text-orange-700 dark:text-orange-400' }, { k: '90+', label: '90+', val: data.summary.over_90, bg: 'bg-red-50 dark:bg-red-900/20', fg: 'text-red-700 dark:text-red-400' }] as card}
				<button
					class="rounded-lg p-3 text-center transition border {bucket === card.k ? 'border-blue-500 ring-1 ring-blue-500' : 'border-transparent'} {card.bg}"
					on:click={() => toggleBucket(card.k)}
					title={$i18n.t('Filter by this bucket')}
				>
					<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{card.label}</div>
					<div class="text-sm font-bold font-mono {card.fg}"><ReportAmount value={card.val} {...fxProps} /></div>
				</button>
			{/each}
			<div class="rounded-lg p-3 bg-gray-50 dark:bg-gray-850 text-center border border-transparent">
				<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400">{$i18n.t('Total')}</div>
				<div class="text-sm font-bold font-mono dark:text-gray-200"><ReportAmount value={data.summary.total} {...fxProps} /></div>
			</div>
		</div>

		{#if data.rows.length === 0}
			<div class="text-sm text-gray-400 italic text-center py-6">{$i18n.t('No outstanding balances')}</div>
		{:else}
			<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2 cursor-pointer select-none" on:click={() => setSort('name')}>{$i18n.t(entityLabel)} {sortIcon('name')}</th>
							<th class="px-2 py-2 text-center">{$i18n.t('Items')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Original')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Paid')}</th>
							<th class="px-2 py-2 text-right cursor-pointer select-none" on:click={() => setSort('balance')}>{$i18n.t('Balance')} {sortIcon('balance')}</th>
							<th class="px-2 py-2 text-center cursor-pointer select-none" on:click={() => setSort('days')}>{$i18n.t('Days')} {sortIcon('days')}</th>
							<th class="px-2 py-2 text-center cursor-pointer select-none" on:click={() => setSort('bucket')}>{$i18n.t('Bucket')} {sortIcon('bucket')}</th>
							<th class="px-2 py-2"></th>
						</tr>
					</thead>
					<tbody>
						{#each data.rows as row}
							<tr
								class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30 cursor-pointer"
								on:click={() => (modalRow = row)}
							>
								<td class="px-3 py-1.5 font-medium max-w-[220px] truncate">{row.vendor_or_customer}</td>
								<td class="px-2 py-1.5 text-center">{row.item_count}</td>
								<td class="px-2 py-1.5 text-right font-mono"><ReportAmount value={row.original_amount} {...fxProps} /></td>
								<td class="px-2 py-1.5 text-right font-mono"><ReportAmount value={row.paid_amount} {...fxProps} /></td>
								<td class="px-2 py-1.5 text-right font-mono font-medium"><ReportAmount value={row.balance} {...fxProps} /></td>
								<td class="px-2 py-1.5 text-center">{row.days_outstanding}</td>
								<td class="px-2 py-1.5 text-center"><span class="font-medium {bucketColor(row.bucket)}">{row.bucket}</span></td>
								<td class="px-2 py-1.5 text-right text-blue-600 dark:text-blue-400">{$i18n.t('View')} →</td>
							</tr>
						{/each}
					</tbody>
					<tfoot class="font-bold bg-gray-50/50 dark:bg-gray-850/30 border-t-2 border-gray-200 dark:border-gray-700">
						<tr>
							<td class="px-3 py-2" colspan="4">{$i18n.t('Total')}</td>
							<td class="px-2 py-2 text-right font-mono"><ReportAmount value={data.summary.total} {...fxProps} /></td>
							<td colspan="3"></td>
						</tr>
					</tfoot>
				</table>
			</div>
		{/if}
	{/if}
</div>

<!-- Drill-down modal -->
{#if modalRow}
	<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" on:click|self={() => (modalRow = null)} role="presentation">
		<div class="bg-white dark:bg-gray-900 rounded-2xl shadow-xl w-full max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
			<div class="flex items-start justify-between px-5 py-4 border-b border-gray-100 dark:border-gray-850">
				<div>
					<div class="text-[10px] uppercase text-gray-400">{$i18n.t(title)}</div>
					<div class="text-base font-semibold dark:text-gray-100">{modalRow.vendor_or_customer}</div>
					<div class="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
						{modalRow.item_count} {$i18n.t('open items')} · {$i18n.t('Balance')} <span class="font-mono font-medium">{fmt(modalRow.balance)}</span>
					</div>
				</div>
				<button class="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xl leading-none" on:click={() => (modalRow = null)}>×</button>
			</div>
			<div class="overflow-y-auto px-5 py-3">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase text-gray-500 dark:text-gray-400 border-b border-gray-100 dark:border-gray-850">
						<tr>
							<th class="py-2 pr-2">{$i18n.t('Type')}</th>
							<th class="py-2 px-2">{$i18n.t('Reference')}</th>
							<th class="py-2 px-2">{$i18n.t('Date')}</th>
							<th class="py-2 px-2">{$i18n.t('Due')}</th>
							<th class="py-2 px-2 text-right">{$i18n.t('Original')}</th>
							<th class="py-2 px-2 text-right">{$i18n.t('Paid')}</th>
							<th class="py-2 px-2 text-right">{$i18n.t('Balance')}</th>
							<th class="py-2 px-2 text-center">{$i18n.t('Days')}</th>
							<th class="py-2 pl-2 text-center">{$i18n.t('Bucket')}</th>
						</tr>
					</thead>
					<tbody>
						{#each modalRow.items as it}
							<tr class="border-b border-gray-50 dark:border-gray-850/30">
								<td class="py-1.5 pr-2">
									<span class="px-1.5 py-0.5 rounded text-[10px] {it.is_invoice ? 'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300'}">
										{it.is_invoice ? $i18n.t('Invoice') : $i18n.t('Journal')}
									</span>
								</td>
								<td class="py-1.5 px-2 font-mono max-w-[220px]">
									{#if it.is_invoice && it.k4mi_document_id}
										<K4miDocLink docId={it.k4mi_document_id} title={$i18n.t('Open in K4mi')}>{it.reference ?? '—'}</K4miDocLink>
									{:else}
										<div class="truncate">{it.reference ?? '—'}</div>
										<!-- Journal item: show + manage linked invoices -->
										<div class="flex flex-wrap items-center gap-1 mt-0.5">
											{#each it.linked_invoices ?? [] as li}
												<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300 text-[10px]">
													<K4miDocLink docId={li.k4mi_document_id} title={$i18n.t('Open in K4mi')}>{li.invoice_number ?? `#${li.invoice_id}`}</K4miDocLink>
													<button class="text-blue-400 hover:text-red-500" title={$i18n.t('Unlink')} on:click|stopPropagation={() => unlinkInvoice(it, li.invoice_id)} disabled={linking}>×</button>
												</span>
											{/each}
											<button
												class="px-1.5 py-0.5 rounded border border-dashed border-gray-300 dark:border-gray-600 text-[10px] text-gray-500 hover:text-blue-600 hover:border-blue-400 transition disabled:opacity-50"
												on:click|stopPropagation={() => openLinkPicker(it)}
												disabled={linking}
											>+ {$i18n.t('Link invoice')}</button>
											{#if it.transaction_id}
												<button
													class="px-1.5 py-0.5 rounded border border-dashed border-gray-300 dark:border-gray-600 text-[10px] text-gray-500 hover:text-green-600 hover:border-green-400 transition disabled:opacity-50"
													on:click|stopPropagation={() => openReconcile(it)}
													disabled={linking}
												>⇄ {$i18n.t('Reconcile')}</button>
											{/if}
										</div>
									{/if}
								</td>
								<td class="py-1.5 px-2">{it.invoice_date ?? '—'}</td>
								<td class="py-1.5 px-2">{it.due_date ?? '—'}</td>
								<td class="py-1.5 px-2 text-right font-mono">{fmt(it.original_amount)}</td>
								<td class="py-1.5 px-2 text-right font-mono">{fmt(it.paid_amount)}</td>
								<td class="py-1.5 px-2 text-right font-mono font-medium">{fmt(it.balance)}</td>
								<td class="py-1.5 px-2 text-center">{it.days_outstanding}</td>
								<td class="py-1.5 pl-2 text-center"><span class="font-medium {bucketColor(it.bucket)}">{it.bucket}</span></td>
							</tr>
							{#if reconcileItem && reconcileItem.transaction_id === it.transaction_id}
								<tr class="bg-gray-50/70 dark:bg-gray-850/40">
									<td colspan="9" class="px-3 py-2">
										<div class="text-[10px] uppercase text-gray-500 dark:text-gray-400 mb-1">
											{$i18n.t('Match to an unmatched bank line')} ({fmt(it.balance)})
										</div>
										{#if loadingCandidates}
											<div class="flex items-center gap-2 text-gray-400"><Spinner className="size-3.5" /> {$i18n.t('Loading…')}</div>
										{:else if bankCandidates.length === 0}
											<div class="text-xs text-gray-400 italic">{$i18n.t('No matching unmatched bank lines found for this amount.')}</div>
										{:else}
											<div class="space-y-1">
												{#each bankCandidates as line}
													<button
														class="w-full flex items-center justify-between gap-3 px-2 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 hover:border-green-400 hover:bg-green-50/50 dark:hover:bg-green-900/10 transition text-left disabled:opacity-50"
														on:click|stopPropagation={() => doReconcile(line)}
														disabled={linking}
													>
														<div class="min-w-0">
															<div class="text-xs dark:text-gray-200 truncate">{line.description ?? line.reference ?? `#${line.id}`}</div>
															<div class="text-[10px] text-gray-400">{line.transaction_date ?? ''} · {line.bank_account_name ?? ''} · {line.match_status}</div>
														</div>
														<div class="font-mono text-xs font-medium whitespace-nowrap">{fmt(line.amount)}</div>
													</button>
												{/each}
											</div>
										{/if}
									</td>
								</tr>
							{/if}
						{/each}
					</tbody>
				</table>
			</div>
		</div>
	</div>
{/if}

<InvoiceSelector bind:show={showInvoiceSelector} on:select={handleLinkSelect} />
