<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getGlobalExchangeRates,
		createGlobalExchangeRate,
		deleteGlobalExchangeRate,
		bulkImportGlobalExchangeRates,
		fetchGlobalRatesNow,
		getGlobalRateCoverage,
		backfillGlobalRates,
		downloadExchangeRateTemplate
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	// A broad picker for manual entry. The auto-fetch covers whatever the source
	// publishes; this list is only for hand-adding a pair.
	const CURRENCIES = [
		'EUR', 'USD', 'GBP', 'CNY', 'JPY', 'CHF', 'CAD', 'AUD', 'NZD', 'HKD',
		'SGD', 'KRW', 'INR', 'THB', 'MYR', 'IDR', 'PHP', 'VND', 'TWD', 'AED',
		'SAR', 'QAR', 'KWD', 'BHD', 'ZAR', 'BRL', 'MXN', 'TRY', 'PLN', 'CZK',
		'HUF', 'RON', 'SEK', 'NOK', 'DKK', 'ILS', 'RUB', 'MAD'
	];

	const today = () => dayjs().format('YYYY-MM-DD');

	// State
	let loading = true;
	let rates: any[] = [];
	// Filters (all applied server-side via the list query).
	let dateFilter = ''; // month view filter ('' = all months)
	let filterFrom = ''; // from/base currency
	let filterTo = ''; // to/quote currency

	// Fetch-now controls
	let fetchSource = 'frankfurter';
	let fetchDate = today(); // which month "Fetch now" pulls (any past date works)
	let fetching = false;

	// Coverage: months with no rates on file. A gap is what makes a backdated
	// entry fall back to another month's rate, so it is worth surfacing.
	let coverage: { start?: string; end?: string; bases?: string[]; missing?: any[] } | null = null;
	let backfilling = false;

	$: missingMonths = coverage?.missing
		? [...new Set(coverage.missing.map((m: any) => m.month))].sort()
		: [];

	const loadCoverage = async () => {
		try {
			coverage = await getGlobalRateCoverage();
		} catch {
			coverage = null;
		}
	};

	const handleBackfill = async () => {
		backfilling = true;
		try {
			const res = await backfillGlobalRates({ source: fetchSource });
			const failed: string[] = res?.months_failed ?? [];
			if (res?.fetched)
				toast.success(
					$i18n.t('Filled {{count}} rates across {{months}} month(s)', {
						count: res.fetched,
						months: res.months
					})
				);
			else if (!failed.length) toast.success($i18n.t('Every month already covered'));
			if (failed.length)
				toast.error(
					$i18n.t('Could not reach the rate source for {{months}} — add those by hand below.', {
						months: failed.join(', ')
					})
				);
			await Promise.all([loadRates(), loadCoverage()]);
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		backfilling = false;
	};

	// Create form
	let showAddForm = false;
	let newFromCurrency = 'USD';
	let newToCurrency = 'EUR';
	let newRate = '';
	let newDate = today();
	let creating = false;

	// Import
	let importingCsv = false;
	let fileInput: HTMLInputElement;

	// Delete confirmation
	let showDeleteConfirm = false;
	let deleteTarget: any = null;

	// ─── Helpers ────────────────────────────────────────────────────────────────

	const formatRate = (val: any) => {
		if (val === null || val === undefined) return '-';
		return parseFloat(val).toLocaleString(undefined, {
			minimumFractionDigits: 4,
			maximumFractionDigits: 6
		});
	};

	// Rows are stored dated the 1st of their month.
	const formatDate = (val: any) => (val ? dayjs(val).format('YYYY-MM-DD') : '-');

	// ─── Data loading ───────────────────────────────────────────────────────────

	const loadRates = async () => {
		loading = true;
		try {
			// Empty strings are dropped by the API client, so this sends only the
			// active filters (server-side narrowing).
			const res = await getGlobalExchangeRates({
				from_currency: filterFrom,
				to_currency: filterTo,
				effective_date: dateFilter
			});
			rates = Array.isArray(res) ? res : res?.items ?? [];
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load exchange rates')}: ${err}`);
		}
		loading = false;
	};

	const clearFilters = () => {
		filterFrom = '';
		filterTo = '';
		dateFilter = '';
		loadRates();
	};

	// ─── Fetch now ────────────────────────────────────────────────────────────────

	const handleFetchNow = async () => {
		fetching = true;
		try {
			const res = await fetchGlobalRatesNow({
				source: fetchSource,
				force: true,
				effective_date: fetchDate || undefined
			});
			const fetched = res?.fetched ?? 0;
			const errs: string[] = res?.errors ?? [];
			if (fetched > 0)
				toast.success($i18n.t('Fetched {{count}} rates', { count: fetched }));
			else if (errs.length === 0)
				toast.success($i18n.t('Rates already up to date for this month'));
			if (errs.length > 0)
				toast.error(`${$i18n.t('Some bases could not be fetched')}: ${errs.join('; ')}`);
			// Show the month we just pulled.
			dateFilter = fetchDate;
			await loadRates();
		} catch (err: any) {
			toast.error(err?.detail ?? `${err}`);
		}
		fetching = false;
	};

	// ─── Create ─────────────────────────────────────────────────────────────────

	const handleCreate = async () => {
		if (!newFromCurrency || !newToCurrency || !newRate || !newDate) {
			toast.error($i18n.t('Please fill in all fields'));
			return;
		}
		if (newFromCurrency === newToCurrency) {
			toast.error($i18n.t('From and To currencies must be different'));
			return;
		}
		const rateVal = parseFloat(newRate);
		if (isNaN(rateVal) || rateVal <= 0) {
			toast.error($i18n.t('Rate must be a positive number'));
			return;
		}

		creating = true;
		try {
			await createGlobalExchangeRate({
				from_currency: newFromCurrency,
				to_currency: newToCurrency,
				rate: rateVal,
				effective_date: newDate
			});
			toast.success($i18n.t('Exchange rate created'));
			showAddForm = false;
			newRate = '';
			await loadRates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create exchange rate') + ': ' + msg);
		}
		creating = false;
	};

	// ─── Delete ─────────────────────────────────────────────────────────────────

	const confirmDelete = (rate: any) => {
		deleteTarget = rate;
		showDeleteConfirm = true;
	};

	const handleDelete = async () => {
		if (!deleteTarget) return;
		try {
			await deleteGlobalExchangeRate(deleteTarget.id);
			toast.success($i18n.t('Exchange rate deleted'));
			await loadRates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to delete exchange rate') + ': ' + msg);
		}
		deleteTarget = null;
	};

	// ─── CSV Import ────────────────────────────────────────────────────────────

	const handleFileImport = async (event: Event) => {
		const input = event.target as HTMLInputElement;
		const file = input?.files?.[0];
		if (!file) return;

		importingCsv = true;
		try {
			const text = await file.text();
			const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
			if (lines.length < 2) {
				toast.error($i18n.t('File must have a header row and at least one data row'));
				importingCsv = false;
				return;
			}

			const header = lines[0].split(/[,;\t]/).map((h) => h.trim().toLowerCase().replace(/['"]/g, ''));
			const dateIdx = header.findIndex((h) => h === 'date' || h === 'effective_date' || h === 'effective date');
			const fromIdx = header.findIndex((h) => h === 'from' || h === 'from_currency' || h === 'from currency');
			const toIdx = header.findIndex((h) => h === 'to' || h === 'to_currency' || h === 'to currency');
			const rateIdx = header.findIndex((h) => h === 'rate' || h === 'exchange_rate' || h === 'exchange rate');

			if (dateIdx < 0 || fromIdx < 0 || toIdx < 0 || rateIdx < 0) {
				toast.error($i18n.t('CSV must have columns: Date, From, To, Rate'));
				importingCsv = false;
				return;
			}

			const parsed: any[] = [];
			for (let i = 1; i < lines.length; i++) {
				const cols = lines[i].split(/[,;\t]/).map((c) => c.trim().replace(/['"]/g, ''));
				const dateVal = cols[dateIdx];
				const fromVal = cols[fromIdx]?.toUpperCase();
				const toVal = cols[toIdx]?.toUpperCase();
				const rateVal = parseFloat(cols[rateIdx]);
				if (!dateVal || !fromVal || !toVal || isNaN(rateVal) || rateVal <= 0) continue;
				parsed.push({ effective_date: dateVal, from_currency: fromVal, to_currency: toVal, rate: rateVal });
			}

			if (parsed.length === 0) {
				toast.error($i18n.t('No valid rates found in file'));
				importingCsv = false;
				return;
			}

			await bulkImportGlobalExchangeRates(parsed);
			toast.success($i18n.t('Imported {{count}} exchange rates', { count: parsed.length }));
			await loadRates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Import failed') + ': ' + msg);
		}
		importingCsv = false;
		if (input) input.value = '';
	};

	onMount(() => {
		loadRates();
		loadCoverage();
	});
</script>

<ConfirmDialog
	bind:show={showDeleteConfirm}
	on:confirm={handleDelete}
	title={$i18n.t('Delete Exchange Rate')}
	message={$i18n.t('Are you sure you want to delete this exchange rate? This action cannot be undone.')}
/>

<div class="py-2">
	<!-- Header -->
	<div class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between">
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Exchange Rates')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">{rates.length}</div>
		</div>

		<div class="flex gap-2 flex-wrap">
			<input type="file" accept=".csv,.txt" class="hidden" bind:this={fileInput} on:change={handleFileImport} />
			<button
				class="px-3 py-2 text-sm font-medium rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition"
				on:click={() => downloadExchangeRateTemplate()}
				title={$i18n.t('Download CSV template with example monthly rates')}
			>
				{$i18n.t('Template')}
			</button>
			<button
				class="px-3 py-2 text-sm font-medium rounded-lg border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 transition disabled:opacity-50"
				disabled={importingCsv}
				on:click={() => fileInput?.click()}
			>
				{importingCsv ? $i18n.t('Importing...') : $i18n.t('Import CSV')}
			</button>
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition"
				on:click={() => (showAddForm = !showAddForm)}
			>
				{showAddForm ? $i18n.t('Cancel') : $i18n.t('Add Rate')}
			</button>
		</div>
	</div>

	<!-- Description -->
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t(
			'Shared exchange rates for the whole platform. The 1st-of-month rate applies to the entire month. Each company uses these unless it overrides a rate in its own settings.'
		)}
	</div>

	<!-- Fetch controls -->
	<div class="flex items-center gap-2 mb-1 px-3 py-2.5 bg-gray-50 dark:bg-gray-850/50 rounded-xl border border-gray-100 dark:border-gray-800 flex-wrap">
		<label for="global-fetch-month" class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Month')}</label>
		<input
			id="global-fetch-month"
			type="date"
			bind:value={fetchDate}
			class="text-xs rounded-lg px-2 py-1.5 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300"
		/>
		<button
			class="px-3 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
			disabled={fetching}
			on:click={handleFetchNow}
		>
			{fetching ? $i18n.t('Fetching...') : $i18n.t('Fetch now')}
		</button>
		<select
			bind:value={fetchSource}
			class="text-xs rounded-lg px-2 py-1.5 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300"
		>
			<option value="frankfurter">{$i18n.t('ECB / Frankfurter — all currencies')}</option>
			<option value="erapi">{$i18n.t('open.er-api — 160+ currencies')}</option>
		</select>
	</div>
	<!-- Fetch helper -->
	<div class="text-[10px] text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t('“Fetch now” pulls rates for the selected month, per company base currency. Pick a past date to backfill previous months. Manual rates are never overwritten.')}
	</div>

	<!-- Coverage: which months have no rates at all -->
	{#if coverage}
		<div
			class="flex items-start gap-3 flex-wrap px-3 py-2.5 mb-3 rounded-lg border {missingMonths.length
				? 'border-amber-200 dark:border-amber-900/40 bg-amber-50/60 dark:bg-amber-950/20'
				: 'border-gray-100 dark:border-gray-850 bg-gray-50/60 dark:bg-gray-900/40'}"
		>
			<div class="flex-1 min-w-[16rem]">
				{#if missingMonths.length}
					<div class="text-xs font-medium text-amber-700 dark:text-amber-300">
						{$i18n.t('{{count}} month(s) have no rates on file', { count: missingMonths.length })}
					</div>
					<div class="text-[10px] text-amber-600/80 dark:text-amber-400/70 mt-0.5">
						{missingMonths.slice(0, 12).join(', ')}{missingMonths.length > 12 ? ' …' : ''}
					</div>
					<div class="text-[10px] text-gray-500 dark:text-gray-400 mt-1">
						{$i18n.t('An entry dated in one of these months has no rate of its own. Fill the gaps below, or add the pair by hand — entries will ask for the rate rather than borrow another month\'s.')}
					</div>
				{:else}
					<div class="text-xs text-gray-600 dark:text-gray-300">
						{$i18n.t('Every month from {{start}} to {{end}} has rates on file.', {
							start: coverage.start,
							end: coverage.end
						})}
					</div>
				{/if}
			</div>
			<button
				class="px-3 py-1.5 text-xs font-medium rounded-lg bg-amber-600 text-white hover:bg-amber-700 transition disabled:opacity-50 shrink-0"
				disabled={backfilling || !missingMonths.length}
				on:click={handleBackfill}
			>
				{backfilling ? $i18n.t('Filling...') : $i18n.t('Fill missing months')}
			</button>
		</div>
	{/if}

	<!-- Coverage caveat -->
	<div class="text-[11px] text-amber-600 dark:text-amber-400/80 px-0.5 mb-3">
		{$i18n.t(
			'ECB / Frankfurter covers ~31 major currencies. If a company base currency is outside that set (e.g. AED, TWD, SAR, VND), switch the source to open.er-api or add rates manually.'
		)}
	</div>

	<!-- Filters -->
	<div class="flex items-center gap-2 mb-3 px-3 py-2 bg-gray-50 dark:bg-gray-850/50 rounded-xl border border-gray-100 dark:border-gray-800 flex-wrap">
		<span class="text-xs font-medium text-gray-500 dark:text-gray-400">{$i18n.t('Filter')}</span>
		<select
			bind:value={filterFrom}
			on:change={loadRates}
			aria-label={$i18n.t('From currency')}
			class="text-xs rounded-lg px-2 py-1.5 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300"
		>
			<option value="">{$i18n.t('From: any')}</option>
			{#each CURRENCIES as cur}<option value={cur}>{cur}</option>{/each}
		</select>
		<span class="text-gray-400 dark:text-gray-500">→</span>
		<select
			bind:value={filterTo}
			on:change={loadRates}
			aria-label={$i18n.t('To currency')}
			class="text-xs rounded-lg px-2 py-1.5 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300"
		>
			<option value="">{$i18n.t('To: any')}</option>
			{#each CURRENCIES as cur}<option value={cur}>{cur}</option>{/each}
		</select>
		<div class="flex items-center gap-1.5">
			<label for="global-rate-month-view" class="text-xs text-gray-500 dark:text-gray-400">{$i18n.t('Month')}</label>
			<input
				id="global-rate-month-view"
				type="date"
				bind:value={dateFilter}
				on:change={loadRates}
				class="text-xs rounded-lg px-2 py-1.5 border border-gray-200 dark:border-gray-700 bg-transparent dark:text-gray-300"
			/>
		</div>
		{#if filterFrom || filterTo || dateFilter}
			<button
				class="text-xs text-gray-500 hover:text-gray-800 dark:text-gray-400 dark:hover:text-gray-100 underline decoration-dotted"
				on:click={clearFilters}
			>
				{$i18n.t('Clear filters')}
			</button>
		{/if}
		<span class="text-[10px] text-gray-400 dark:text-gray-500 ml-auto">{rates.length} {$i18n.t('shown')}</span>
	</div>

	<!-- Add Rate Form -->
	{#if showAddForm}
		<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30 mb-3">
			<div class="text-sm font-medium dark:text-gray-200 mb-3">{$i18n.t('Add Exchange Rate')}</div>
			<div class="grid grid-cols-1 md:grid-cols-5 gap-3 items-end">
				<div>
					<label for="grate-from" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('From Currency')} *</label>
					<select id="grate-from" bind:value={newFromCurrency} class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition">
						{#each CURRENCIES as cur}<option value={cur}>{cur}</option>{/each}
					</select>
				</div>
				<div>
					<label for="grate-to" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('To Currency')} *</label>
					<select id="grate-to" bind:value={newToCurrency} class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition">
						{#each CURRENCIES as cur}<option value={cur}>{cur}</option>{/each}
					</select>
				</div>
				<div>
					<label for="grate-value" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Rate')} *</label>
					<input id="grate-value" type="number" step="0.000001" min="0" bind:value={newRate} placeholder="1.0850" class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition" />
				</div>
				<div>
					<label for="grate-date" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Effective Date')} *</label>
					<input id="grate-date" type="date" bind:value={newDate} class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition" />
				</div>
				<button class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50" on:click={handleCreate} disabled={creating || !newFromCurrency || !newToCurrency || !newRate || !newDate}>
					{creating ? $i18n.t('Saving...') : $i18n.t('Save')}
				</button>
			</div>
			<div class="text-[10px] text-gray-400 dark:text-gray-500 mt-2">
				{$i18n.t('The date is applied to its whole calendar month (stored as the 1st).')}
			</div>
		</div>
	{/if}

	<!-- Rates Table -->
	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if rates.length === 0}
		<div class="bg-white dark:bg-gray-900 rounded-xl p-8 border border-gray-100/30 dark:border-gray-850/30 text-center">
			<div class="text-gray-400 dark:text-gray-500 text-sm mb-3">{$i18n.t('No exchange rates defined yet.')}</div>
			<div class="text-gray-400 dark:text-gray-500 text-xs">{$i18n.t('Click "Fetch now" to pull live rates, or add them manually.')}</div>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800">
					<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
						<th class="px-3 py-2">{$i18n.t('Effective Date')}</th>
						<th class="px-3 py-2">{$i18n.t('From')}</th>
						<th class="px-3 py-2">{$i18n.t('To')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Rate')}</th>
						<th class="px-3 py-2">{$i18n.t('Source')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each rates as rate (rate.id)}
						<tr class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition">
							<td class="px-3 py-2 font-medium dark:text-gray-200">{formatDate(rate.effective_date)}</td>
							<td class="px-3 py-2"><span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300">{rate.from_currency}</span></td>
							<td class="px-3 py-2"><span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300">{rate.to_currency}</span></td>
							<td class="px-3 py-2 text-right font-mono">{formatRate(rate.rate)}</td>
							<td class="px-3 py-2">
								{#if rate.source === 'manual' || rate.source === 'Manual'}
									<span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-200">{$i18n.t('Manual')}</span>
								{:else if rate.source === 'frankfurter' || rate.source === 'erapi' || rate.source === 'ecb' || rate.source === 'exchangerate'}
									<span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-200">{$i18n.t('Auto')}</span>
								{:else}
									<span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300">{rate.source ? $i18n.t(rate.source) : $i18n.t('Import')}</span>
								{/if}
							</td>
							<td class="px-3 py-2 text-right">
								<Tooltip content={$i18n.t('Delete this exchange rate')}>
									<button class="px-3 py-1 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 dark:hover:bg-red-900/40 transition" on:click={() => confirmDelete(rate)}>
										{$i18n.t('Delete')}
									</button>
								</Tooltip>
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
