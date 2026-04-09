<script lang="ts">
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getGeneralLedger, getFullGeneralLedger, getAccounts, exportGeneralLedger } from '$lib/apis/accounting';
	import { K4MI_BASE_URL } from '$lib/constants';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let loading = false;
	let loaded = false;
	let mode: 'journal' | 'account' = 'journal';

	let dateFrom = '';
	let dateTo = '';

	// Journal mode
	let journalEntries: any[] = [];
	let journalTotal = 0;
	let journalTotalDebit = 0;
	let journalTotalCredit = 0;
	let journalPage = 0;
	const journalPerPage = 50;

	// Account mode
	let accounts: Array<{ id: number; code: string; name: string }> = [];
	let selectedAccountId: number | '' = '';
	let accountData: any = null;
	let accountPage = 0;
	const accountPerPage = 50;

	const loadAccounts = async () => {
		try {
			const res = await getAccounts({ company_id: companyId, active: true });
			accounts = (Array.isArray(res) ? res : []).map((a: any) => ({ id: a.id, code: a.code, name: a.name }));
		} catch (err) { console.error(err); }
	};

	const load = async () => {
		loading = true;
		try {
			if (mode === 'journal') {
				const res = await getFullGeneralLedger({
					company_id: companyId,
					date_from: dateFrom || undefined,
					date_to: dateTo || undefined,
					limit: journalPerPage,
					offset: journalPage * journalPerPage
				});
				journalEntries = res.entries ?? [];
				journalTotal = res.total ?? 0;
				journalTotalDebit = parseFloat(res.total_debit ?? 0);
				journalTotalCredit = parseFloat(res.total_credit ?? 0);
			} else {
				if (!selectedAccountId) { loading = false; return; }
				accountData = await getGeneralLedger({
					company_id: companyId,
					account_id: selectedAccountId as number,
					date_from: dateFrom || undefined,
					date_to: dateTo || undefined,
					limit: accountPerPage,
					offset: accountPage * accountPerPage
				});
			}
		} catch (err) { toast.error(`${err}`); }
		loading = false;
		loaded = true;
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const fmtRate = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 1);
		if (n === 1) return '1';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 8 });
	};

	// Compute original amount for a line (max of debit, credit — the gross amount)
	const origAmount = (line: any): string => {
		const d = parseFloat(line.debit) || 0;
		const c = parseFloat(line.credit) || 0;
		return fmt(Math.max(d, c));
	};

	$: if (mode === 'account' && accounts.length === 0) loadAccounts();
</script>

<div class="space-y-3">
	<!-- Filters -->
	<div class="flex flex-wrap gap-3 items-end">
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('View')}</label>
			<select bind:value={mode} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden">
				<option value="journal">{$i18n.t('Journal View')}</option>
				<option value="account">{$i18n.t('Account View')}</option>
			</select>
		</div>
		{#if mode === 'account'}
			<div>
				<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Account')}</label>
				<select bind:value={selectedAccountId} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden max-w-[250px]">
					<option value="">{$i18n.t('Select account...')}</option>
					{#each accounts as acct}<option value={acct.id}>{acct.code} — {acct.name}</option>{/each}
				</select>
			</div>
		{/if}
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('From')}</label>
			<input type="date" bind:value={dateFrom} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
		</div>
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('To')}</label>
			<input type="date" bind:value={dateTo} class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden" />
		</div>
		<button class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition" on:click={() => { journalPage = 0; accountPage = 0; load(); }}>{$i18n.t('Load')}</button>
		{#if (mode === 'journal' && journalEntries.length > 0) || (mode === 'account' && accountData)}
			<button
				class="px-3 py-1.5 text-xs font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-700 transition flex items-center gap-1.5"
				on:click={() => exportGeneralLedger({ company_id: companyId, date_from: dateFrom || undefined, date_to: dateTo || undefined })}
			>
				<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="size-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3" /></svg>
				{$i18n.t('Export Excel')}
			</button>
		{/if}
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if mode === 'journal' && loaded}
		<!-- Journal View matching sample: Period | Reference | Description | Account Code | Account Name | Currency | Exch Rate | Orig Amount | Debit | Credit -->
		<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
			<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300 whitespace-nowrap">
				<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
					<tr>
						<th class="px-2 py-2">{$i18n.t('#')}</th>
						<th class="px-2 py-2">{$i18n.t('Period')}</th>
						<th class="px-2 py-2">{$i18n.t('Reference')}</th>
						<th class="px-2 py-2">{$i18n.t('Description')}</th>
						<th class="px-2 py-2">{$i18n.t('Account Code')}</th>
						<th class="px-2 py-2">{$i18n.t('Account Name')}</th>
						<th class="px-2 py-2">{$i18n.t('Currency')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Exch. Rate')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Orig. Amount')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Debit')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Credit')}</th>
					</tr>
				</thead>
				<tbody>
					{#if journalEntries.length === 0}
						<tr>
							<td colspan="11" class="px-4 py-8 text-center text-sm text-gray-400 italic">{$i18n.t('No journal entries found.')}</td>
						</tr>
					{:else}
						{#each journalEntries as entry, entryIdx}
							{#each entry.lines as line, lineIdx}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30 {lineIdx === 0 && entryIdx > 0 ? 'border-t-2 border-gray-200 dark:border-gray-700' : ''}">
									<td class="px-2 py-1.5 font-mono text-[10px]" title="ID: {entry.transaction_id}">{lineIdx === 0 ? (entry.entry_number || entry.transaction_id || '') : ''}</td>
								<td class="px-2 py-1.5 font-mono">{lineIdx === 0 ? entry.period : ''}</td>
									<td class="px-2 py-1.5">{#if lineIdx === 0 && entry.k4mi_document_id}<a href="{K4MI_BASE_URL}/documents/{entry.k4mi_document_id}/details" target="_blank" rel="noopener" class="text-blue-600 dark:text-blue-400 hover:underline" title={$i18n.t('Open in K4mi')}>{entry.reference ?? ''} <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3 h-3 inline mb-0.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" /></svg></a>{:else if lineIdx === 0}{entry.reference ?? ''}{/if}</td>
									<td class="px-2 py-1.5 max-w-[200px] truncate">{lineIdx === 0 ? (entry.description ?? '') : ''}</td>
									<td class="px-2 py-1.5 font-mono">{line.account_code}</td>
									<td class="px-2 py-1.5">{line.account_name}</td>
									<td class="px-2 py-1.5">{lineIdx === 0 ? entry.currency : ''}</td>
									<td class="px-2 py-1.5 text-right font-mono">{lineIdx === 0 ? fmtRate(entry.exchange_rate) : ''}</td>
									<td class="px-2 py-1.5 text-right font-mono">{origAmount(line)}</td>
									<td class="px-2 py-1.5 text-right font-mono">{fmt(line.debit)}</td>
									<td class="px-2 py-1.5 text-right font-mono">{fmt(line.credit)}</td>
								</tr>
							{/each}
						{/each}
					{/if}
				</tbody>
				<tfoot class="font-medium bg-gray-50 dark:bg-gray-850/50 text-gray-800 dark:text-gray-200">
					<tr class="border-t-2 border-gray-200 dark:border-gray-700">
						<td class="px-2 py-2" colspan="8">{$i18n.t('Total')}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(journalTotalDebit)}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(journalTotalCredit)}</td>
					</tr>
				</tfoot>
			</table>
		</div>

		{#if journalTotal > journalPerPage}
			<div class="flex justify-between items-center text-xs text-gray-500">
				<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={journalPage === 0} on:click={() => { journalPage--; load(); }}>{$i18n.t('Previous')}</button>
				<span>{$i18n.t('Page')} {journalPage + 1} / {Math.ceil(journalTotal / journalPerPage)}</span>
				<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={(journalPage + 1) * journalPerPage >= journalTotal} on:click={() => { journalPage++; load(); }}>{$i18n.t('Next')}</button>
			</div>
		{/if}

	{:else if mode === 'account' && accountData}
		{@const hasSubAccounts = accountData.entries.some((e) => e.account_code)}
		<div class="text-sm font-medium dark:text-gray-200">
			{accountData.account_code} — {accountData.account_name}
			{#if hasSubAccounts}<span class="text-[10px] text-gray-400 ml-1">({$i18n.t('incl. sub-accounts')})</span>{/if}
			<span class="text-xs text-gray-500 ml-2">{$i18n.t('Opening')}: {fmt(accountData.opening_balance)}</span>
		</div>

		<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
			<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
				<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
					<tr>
						<th class="px-2 py-2">{$i18n.t('Date')}</th>
						{#if hasSubAccounts}<th class="px-2 py-2">{$i18n.t('Account')}</th>{/if}
						<th class="px-2 py-2">{$i18n.t('Reference')}</th>
						<th class="px-2 py-2">{$i18n.t('Description')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Debit')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Credit')}</th>
						<th class="px-2 py-2 text-right">{$i18n.t('Running Balance')}</th>
					</tr>
				</thead>
				<tbody>
					{#each accountData.entries as entry}
						<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
							<td class="px-2 py-1.5">{entry.transaction_date}</td>
							{#if hasSubAccounts}
								<td class="px-2 py-1.5">
									{#if entry.account_code}
										<span class="font-mono text-[10px] px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400">{entry.account_code}</span>
										<span class="text-[10px] text-gray-400 ml-0.5">{entry.account_name ?? ''}</span>
									{/if}
								</td>
							{/if}
							<td class="px-2 py-1.5">{#if entry.k4mi_document_id}<a href="{K4MI_BASE_URL}/documents/{entry.k4mi_document_id}/details" target="_blank" rel="noopener" class="text-blue-600 dark:text-blue-400 hover:underline" title={$i18n.t('Open in K4mi')}>{entry.reference ?? ''} <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-3 h-3 inline mb-0.5"><path stroke-linecap="round" stroke-linejoin="round" d="M13.5 6H5.25A2.25 2.25 0 0 0 3 8.25v10.5A2.25 2.25 0 0 0 5.25 21h10.5A2.25 2.25 0 0 0 18 18.75V10.5m-10.5 6L21 3m0 0h-5.25M21 3v5.25" /></svg></a>{:else}{entry.reference ?? ''}{/if}</td>
							<td class="px-2 py-1.5">{entry.description ?? ''}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(entry.debit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(entry.credit)}</td>
							<td class="px-2 py-1.5 text-right font-mono font-medium">{fmt(entry.running_balance)}</td>
						</tr>
					{/each}
				</tbody>
				<tfoot class="font-medium bg-gray-50 dark:bg-gray-850/50 text-gray-800 dark:text-gray-200">
					<tr class="border-t-2 border-gray-200 dark:border-gray-700">
						<td class="px-2 py-2" colspan="{hasSubAccounts ? 6 : 5}">{$i18n.t('Closing Balance')}</td>
						<td class="px-2 py-2 text-right font-mono">{fmt(accountData.closing_balance)}</td>
					</tr>
				</tfoot>
			</table>
		</div>

		{#if accountData.total > accountPerPage}
			<div class="flex justify-between items-center text-xs text-gray-500">
				<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={accountPage === 0} on:click={() => { accountPage--; load(); }}>{$i18n.t('Previous')}</button>
				<span>{$i18n.t('Page')} {accountPage + 1} / {Math.ceil(accountData.total / accountPerPage)}</span>
				<button class="px-2 py-1 rounded hover:bg-gray-100 dark:hover:bg-gray-800 transition disabled:opacity-50" disabled={(accountPage + 1) * accountPerPage >= accountData.total} on:click={() => { accountPage++; load(); }}>{$i18n.t('Next')}</button>
			</div>
		{/if}
	{/if}
</div>
