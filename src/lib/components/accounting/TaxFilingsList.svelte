<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getTaxFilings,
		markTaxFilingPaid,
		deleteTaxFiling,
		getAccounts
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
	let paidDate = '';
	let processing = false;

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};
	const today = () => new Date().toISOString().slice(0, 10);
	const fmtDate = (d: any) => (d ? dayjs(d).format('YYYY-MM-DD') : '—');

	$: bankAccounts = accounts.filter((a: any) => !a.account_type || a.account_type === 'asset');

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
			const a = await getAccounts({ company_id: companyId });
			accounts = Array.isArray(a) ? a : a?.items ?? a?.accounts ?? [];
		} catch {
			accounts = [];
		}
		await reload();
	});

	const startPay = (f: any) => {
		payingId = f.id;
		bankAccountId = '';
		paidDate = today();
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
				paid_date: paidDate || today()
			});
			toast.success($i18n.t('Filing marked paid'));
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
							<td class="px-3 py-2">
								<span class="px-2 py-0.5 rounded-full text-[10px] font-medium {statusBadge(f)}">
									{statusText(f)}
								</span>
							</td>
							<td class="px-3 py-2 text-right whitespace-nowrap">
								{#if f.status !== 'paid'}
									<button
										class="px-2.5 py-1 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition"
										on:click={() => startPay(f)}
									>
										{$i18n.t('Mark Paid')}
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
								<td colspan="5" class="px-3 py-3">
									<div class="flex flex-wrap items-end gap-3">
										<div>
											<label class="block text-[10px] font-medium text-gray-500 dark:text-gray-400 mb-1" for="pay-bank-{f.id}">
												{$i18n.t('Bank / cash account (credit)')}
											</label>
											<select
												id="pay-bank-{f.id}"
												bind:value={bankAccountId}
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
												class="text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
											/>
										</div>
										<button
											class="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 transition disabled:opacity-50"
											disabled={processing || !bankAccountId}
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
									<div class="text-[10px] text-gray-400 dark:text-gray-500 mt-2">
										{$i18n.t('Records a payment: debit the tax-payable account, credit the chosen bank account.')}
									</div>
								</td>
							</tr>
						{/if}
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
