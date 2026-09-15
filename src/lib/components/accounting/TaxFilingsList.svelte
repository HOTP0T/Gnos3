<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getTaxFilings,
		markTaxFilingPaid,
		deleteTaxFiling,
		getAccounts,
		getTaxPaymentPreview
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;
	export let taxType: string;

	let loading = true;
	let filings: any[] = [];
	let accounts: any[] = [];

	let payingId: number | null = null;
	let bankAccountId: number | '' = '';
	let payableAccountId: number | '' = '';
	let paidDate = '';
	// CIT: the year's final tax as printed on the assessment notice (blank = as accrued).
	let assessedAmount: string | number | null = '';
	let processing = false;
	// What the settlement will book (debit and credit legs) + roles still unmapped.
	let preview: { legs: any[]; missing: string[]; total: number; bank?: number; warnings?: string[]; over_under?: number; tax_accrued?: number; bank_fx?: any } | null = null;
	let previewLoading = false;
	const blank = (v: any) => v === '' || v === null || v === undefined;
	const assessed = () => (blank(assessedAmount) ? undefined : parseFloat(String(assessedAmount)) || 0);
	$: isCit = taxType === 'cit';
	// Foreign-currency bank account: what left the bank in its currency, and the
	// rate (prefilled from the rate table; typed when the bank's rate differs).
	let bankFcAmount: string | number | null = '';
	let bankRate: string | number | null = '';
	const numOr = (v: any) => (blank(v) ? undefined : parseFloat(String(v)) || undefined);
	$: bankFx = preview?.bank_fx ?? null;
	$: foreignBank = !!bankFx?.foreign;

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};
	const today = () => new Date().toISOString().slice(0, 10);
	const fmtDate = (d: any) => (d ? dayjs(d).format('YYYY-MM-DD') : '—');

	$: parentIds = new Set(accounts.map((a: any) => a.parent_id).filter(Boolean));
	$: leafAccounts = accounts.filter((a: any) => !parentIds.has(a.id));
	$: bankAccounts = leafAccounts.filter((a: any) => !a.account_type || a.account_type === 'asset');
	$: payableAccounts = leafAccounts.filter((a: any) => a.account_type === 'liability');

	export async function reload() {
		loading = true;
		try {
			const res = await getTaxFilings({ company_id: companyId, tax_type: taxType });
			filings = Array.isArray(res) ? res : [];
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to load filings')}: ${err?.detail ?? err}`);
		}
		loading = false;
	}

	onMount(async () => {
		try {
			const a = await getAccounts({ company_id: companyId, active: true });
			accounts = Array.isArray(a) ? a : a?.items ?? a?.accounts ?? [];
		} catch {
			accounts = [];
		}
		await reload();
	});

	const loadPreview = async (f: any) => {
		previewLoading = true;
		try {
			preview = await getTaxPaymentPreview(f.id, payableAccountId === '' ? undefined : Number(payableAccountId), isCit ? assessed() : undefined, {
				bank_account_id: bankAccountId === '' ? undefined : Number(bankAccountId),
				bank_fc_amount: numOr(bankFcAmount),
				rate: numOr(bankRate),
				paid_date: paidDate || undefined
			});
		} catch (err: any) {
			preview = null;
			toast.error(`${$i18n.t('Failed to preview payment')}: ${err?.detail ?? err}`);
		}
		previewLoading = false;
	};

	const startPay = async (f: any) => {
		payingId = f.id;
		bankAccountId = '';
		payableAccountId = '';
		assessedAmount = '';
		bankFcAmount = '';
		bankRate = '';
		paidDate = today();
		preview = null;
		await loadPreview(f);
	};

	const confirmPay = async (f: any) => {
		if (!bankAccountId) {
			toast.error($i18n.t('Select a bank account'));
			return;
		}
		processing = true;
		try {
			await markTaxFilingPaid(f.id, {
				bank_account_id: Number(bankAccountId),
				paid_date: paidDate || today(),
				payable_account_id: payableAccountId === '' ? undefined : Number(payableAccountId),
				assessed_amount: isCit ? assessed() : undefined,
				bank_fc_amount: foreignBank ? numOr(bankFcAmount) : undefined,
				rate: foreignBank ? numOr(bankRate) : undefined
			});
			toast.success((preview?.bank ?? preview?.total ?? 0) > 0 ? $i18n.t('Filing marked paid') : $i18n.t('Filing settled against provisional tax — no bank movement'));
			payingId = null;
			await reload();
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to mark paid')}: ${err?.detail ?? err}`);
		}
		processing = false;
	};

	const remove = async (f: any) => {
		try {
			await deleteTaxFiling(f.id);
			await reload();
		} catch (err: any) {
			toast.error(`${err?.detail ?? err}`);
		}
	};

	const statusBadge = (f: any) => {
		if (f.status === 'paid') return 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-300';
		if (f.overdue) return 'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300';
		return 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300';
	};
	const statusText = (f: any) =>
		f.status === 'paid' ? $i18n.t('Paid') : f.overdue ? $i18n.t('Overdue') : $i18n.t('Open');
	const entryText = (f: any) => {
		if (!f.settlement_transaction_id) return '—';
		const st = f.settlement_entry_status;
		const num = f.settlement_entry_number ? ` ${f.settlement_entry_number}` : ` #${f.settlement_transaction_id}`;
		return (st === 'posted' ? $i18n.t('Posted') : st === 'voided' ? $i18n.t('Voided') : $i18n.t('Draft')) + num;
	};
</script>

<div
	class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mt-4"
>
	<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850 text-sm font-medium dark:text-gray-200">
		{$i18n.t('Filings')}
	</div>

	{#if loading}
		<div class="flex justify-center my-6"><Spinner className="size-5" /></div>
	{:else if filings.length === 0}
		<div class="px-4 py-6 text-center text-sm text-gray-400 dark:text-gray-500">
			{$i18n.t('No saved filings yet. Calculate a period and Save it as a filing.')}
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead
					class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
				>
					<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
						<th class="px-3 py-2">{$i18n.t('Period')}</th>
						<th class="px-3 py-2">{$i18n.t('Due date')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Amount')}</th>
						<th class="px-3 py-2">{$i18n.t('Entry')}</th>
						<th class="px-3 py-2">{$i18n.t('Status')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each filings as f (f.id)}
						<tr
							class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs"
						>
							<td class="px-3 py-2">{fmtDate(f.period_start)} → {fmtDate(f.period_end)}</td>
							<td class="px-3 py-2">{fmtDate(f.due_date)}</td>
							<td class="px-3 py-2 text-right font-mono">{fmt(f.tax_amount)} {f.currency ?? ''}</td>
							<td class="px-3 py-2 text-[11px] text-gray-500 dark:text-gray-400 whitespace-nowrap">{entryText(f)}</td>
							<td class="px-3 py-2">
								<span class="px-2 py-0.5 rounded-full text-[10px] font-medium {statusBadge(f)}">
									{statusText(f)}
								</span>
							</td>
							<td class="px-3 py-2 text-right whitespace-nowrap">
								{#if f.status !== 'paid' && Number(f.tax_amount) <= 0}
									<span class="text-[10px] text-gray-400" title={$i18n.t('Nothing to pay: the tax office refunds or credits the excess — record the refund as a receipt against the prepaid-tax account.')}>{$i18n.t('Refund / nothing to pay')}</span>
									<button
										class="px-2.5 py-1 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 transition"
										on:click={() => remove(f)}
									>
										{$i18n.t('Delete')}
									</button>
								{:else if f.status !== 'paid'}
									<button
										class="px-2.5 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition"
										on:click={() => startPay(f)}
									>
										{isCit ? $i18n.t('Settle') : $i18n.t('Mark Paid')}
									</button>
									<button
										class="px-2.5 py-1 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 transition"
										on:click={() => remove(f)}
									>
										{$i18n.t('Delete')}
									</button>
								{:else}
									<span class="text-[10px] text-gray-400">{$i18n.t('Paid')} {fmtDate(f.paid_date)}</span>
								{/if}
							</td>
						</tr>
						{#if payingId === f.id}
							<tr class="bg-blue-50/40 dark:bg-blue-900/10 border-b border-gray-100 dark:border-gray-850">
								<td colspan="6" class="px-3 py-3">
									<div class="flex flex-wrap items-end gap-3">
										<div>
											<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-bank-{f.id}">
												{$i18n.t('Bank / cash account (credit)')}
											</label>
											<select
												id="pay-bank-{f.id}"
												bind:value={bankAccountId}
												on:change={() => { bankFcAmount = ''; bankRate = ''; loadPreview(f); }}
												class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
											>
												<option value="">{$i18n.t('Select account...')}</option>
												{#each bankAccounts as acct}
													<option value={acct.id}>{acct.code} - {acct.name}</option>
												{/each}
											</select>
										</div>
										<div>
											<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-date-{f.id}">
												{$i18n.t('Paid date')}
											</label>
											<input
												id="pay-date-{f.id}"
												type="date"
												bind:value={paidDate}
												on:change={() => loadPreview(f)}
												class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
											/>
										</div>
										{#if foreignBank}
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-fc-{f.id}">
													{$i18n.t('Amount debited')} ({bankFx.bank_currency})
												</label>
												<input
													id="pay-fc-{f.id}"
													type="number"
													step="0.01"
													bind:value={bankFcAmount}
													on:change={() => loadPreview(f)}
													placeholder={bankFx.bank_fc_suggested != null ? `${$i18n.t('at the rate')}: ${fmt(bankFx.bank_fc_suggested)}` : ''}
													class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden w-40"
												/>
											</div>
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-rate-{f.id}">
													{$i18n.t('Rate')} {bankFx.bank_currency} → {bankFx.base_currency}
												</label>
												<input
													id="pay-rate-{f.id}"
													type="number"
													step="0.00000001"
													bind:value={bankRate}
													on:change={() => loadPreview(f)}
													placeholder={bankFx.rate != null ? `${bankFx.rate} (${bankFx.rate_source})` : $i18n.t('no rate on file — type it')}
													class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden w-40"
												/>
											</div>
										{/if}
										<button
											class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 transition disabled:opacity-50"
											disabled={processing || !bankAccountId || (preview?.missing?.length ?? 0) > 0 || (foreignBank && numOr(bankFcAmount) === undefined)}
											on:click={() => confirmPay(f)}
										>
											{processing ? $i18n.t('Recording...') : $i18n.t('Record payment')}
										</button>
										<button
											class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-100 text-gray-700 dark:bg-gray-850 dark:text-gray-300 transition"
											on:click={() => (payingId = null)}
										>
											{$i18n.t('Cancel')}
										</button>
									</div>
									<div class="flex flex-wrap items-end gap-3 mt-3">
										<div>
											<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-payable-{f.id}">
												{$i18n.t('Tax payable account (debit)')}
											</label>
											<select
												id="pay-payable-{f.id}"
												bind:value={payableAccountId}
												on:change={() => loadPreview(f)}
												class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
											>
												<option value="">{$i18n.t('From Tax accounts mapping')}</option>
												{#each payableAccounts as acct}
													<option value={acct.id}>{acct.code} - {acct.name}</option>
												{/each}
											</select>
										</div>
										{#if isCit}
											<div>
												<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-assessed-{f.id}">
													{$i18n.t("Final tax per the assessment notice (the year's tax, before provisional tax)")}
												</label>
												<input
													id="pay-assessed-{f.id}"
													type="number"
													step="0.01"
													bind:value={assessedAmount}
													on:change={() => loadPreview(f)}
													placeholder={preview?.tax_accrued !== undefined ? `${$i18n.t('as accrued')}: ${fmt(preview.tax_accrued)}` : $i18n.t('as accrued')}
													class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden w-56"
												/>
											</div>
										{/if}
									</div>
									{#if previewLoading}
										<div class="text-[10px] text-gray-400 mt-2">{$i18n.t('Loading...')}</div>
									{:else if preview}
										<div class="mt-2 text-[11px]">
											<div class="text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('This will book:')}</div>
											<table class="text-[11px]">
												<tbody>
													{#each preview.legs as leg}
														<tr>
															<td class="pr-3 py-0.5 text-gray-600 dark:text-gray-300">{(leg.side ?? 'debit') === 'debit' ? 'DR' : 'CR'} {$i18n.t(leg.label)}</td>
															<td class="pr-3 py-0.5 font-mono {leg.account ? 'dark:text-gray-200' : 'text-red-600 dark:text-red-300'}">
																{leg.account ? `${leg.account.code} ${leg.account.name}` : $i18n.t('— not mapped —')}
															</td>
															<td class="py-0.5 text-right font-mono">{fmt(leg.amount)}</td>
														</tr>
													{/each}
													<tr class="font-medium">
														<td class="pr-3 py-0.5">{(preview.bank ?? preview.total) > 0 ? `CR ${$i18n.t('Bank')}` : $i18n.t('Nothing to pay from the bank')}</td>
														<td></td>
														<td class="py-0.5 text-right font-mono">{fmt(preview.bank ?? preview.total)}</td>
													</tr>
												</tbody>
											</table>
											{#if foreignBank}
												<div class="text-gray-500 dark:text-gray-400 mt-1">
													{$i18n.t('Paid from a {{ccy}} account: the obligations are cleared in {{base}} through the exchange clearing account, then the clearing is moved to the bank in {{ccy}}.', { ccy: bankFx.bank_currency, base: bankFx.base_currency })}
													{#if bankFx.fx_kind}
														— {$i18n.t(bankFx.fx_kind === 'loss' ? 'exchange loss' : 'exchange gain')} {fmt(Math.abs(bankFx.fx_base))} {bankFx.base_currency}
														→ {bankFx.fx_account ? `${bankFx.fx_account.code} ${bankFx.fx_account.name}` : $i18n.t('account not set')}
													{/if}
												</div>
											{/if}
											{#each preview.warnings ?? [] as w}
												<div class="text-amber-700 dark:text-amber-300 mt-1">{w}</div>
											{/each}
											{#if preview.missing.length > 0}
												<div class="text-red-700 dark:text-red-300 mt-1">
													{$i18n.t('Map the missing role(s) in the Accounts tab (or pick a tax payable account above) before recording.')}
												</div>
											{/if}
										</div>
									{/if}
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
