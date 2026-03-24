<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import {
		getExchangeRates,
		createExchangeRate,
		deleteExchangeRate,
		bulkImportExchangeRates,
		downloadExchangeRateTemplate
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	const CURRENCIES = [
		'EUR',
		'USD',
		'GBP',
		'CNY',
		'JPY',
		'CAD',
		'AUD',
		'CHF',
		'BRL',
		'INR',
		'KRW',
		'MXN',
		'SGD',
		'HKD',
		'MAD'
	];

	// State
	let loading = true;
	let rates: any[] = [];

	// Create form
	let showAddForm = false;
	let newFromCurrency = 'USD';
	let newToCurrency = 'EUR';
	let newRate = '';
	let newEffectiveDate = '';
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

	const formatDate = (val: any) => {
		if (!val) return '-';
		return dayjs(val).format('YYYY-MM-DD');
	};

	// ─── Data loading ───────────────────────────────────────────────────────────

	const loadRates = async () => {
		loading = true;
		try {
			const res = await getExchangeRates({ company_id: companyId });
			rates = Array.isArray(res) ? res : res?.items ?? [];
			// Sort by effective_date descending
			rates.sort(
				(a: any, b: any) =>
					new Date(b.effective_date).getTime() - new Date(a.effective_date).getTime()
			);
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load exchange rates')}: ${err}`);
		}
		loading = false;
	};

	// ─── Create ─────────────────────────────────────────────────────────────────

	const handleCreate = async () => {
		if (!newFromCurrency || !newToCurrency || !newRate || !newEffectiveDate) {
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
			await createExchangeRate(companyId, {
				from_currency: newFromCurrency,
				to_currency: newToCurrency,
				rate: rateVal,
				effective_date: newEffectiveDate
			});
			toast.success($i18n.t('Exchange rate created'));
			showAddForm = false;
			newRate = '';
			newEffectiveDate = '';
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
			await deleteExchangeRate(deleteTarget.id);
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
			const lines = text.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
			if (lines.length < 2) {
				toast.error($i18n.t('File must have a header row and at least one data row'));
				importingCsv = false;
				return;
			}

			// Parse header to find column indices
			const header = lines[0].split(/[,;\t]/).map(h => h.trim().toLowerCase().replace(/['"]/g, ''));
			const dateIdx = header.findIndex(h => h === 'date' || h === 'effective_date' || h === 'effective date');
			const fromIdx = header.findIndex(h => h === 'from' || h === 'from_currency' || h === 'from currency');
			const toIdx = header.findIndex(h => h === 'to' || h === 'to_currency' || h === 'to currency');
			const rateIdx = header.findIndex(h => h === 'rate' || h === 'exchange_rate' || h === 'exchange rate');

			if (dateIdx < 0 || fromIdx < 0 || toIdx < 0 || rateIdx < 0) {
				toast.error($i18n.t('CSV must have columns: Date, From, To, Rate'));
				importingCsv = false;
				return;
			}

			const rates: any[] = [];
			for (let i = 1; i < lines.length; i++) {
				const cols = lines[i].split(/[,;\t]/).map(c => c.trim().replace(/['"]/g, ''));
				const dateVal = cols[dateIdx];
				const fromVal = cols[fromIdx]?.toUpperCase();
				const toVal = cols[toIdx]?.toUpperCase();
				const rateVal = parseFloat(cols[rateIdx]);

				if (!dateVal || !fromVal || !toVal || isNaN(rateVal) || rateVal <= 0) continue;

				rates.push({
					effective_date: dateVal,
					from_currency: fromVal,
					to_currency: toVal,
					rate: rateVal
				});
			}

			if (rates.length === 0) {
				toast.error($i18n.t('No valid rates found in file'));
				importingCsv = false;
				return;
			}

			await bulkImportExchangeRates(companyId, rates);
			toast.success($i18n.t('Imported {{count}} exchange rates', { count: rates.length }));
			await loadRates();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Import failed') + ': ' + msg);
		}
		importingCsv = false;
		// Reset input so same file can be re-selected
		if (input) input.value = '';
	};

	onMount(() => {
		loadRates();
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
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Exchange Rates')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{rates.length}
			</div>
		</div>

		<div class="flex gap-2">
			<input
				type="file"
				accept=".csv,.txt"
				class="hidden"
				bind:this={fileInput}
				on:change={handleFileImport}
			/>
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
				on:click={() => {
					showAddForm = !showAddForm;
				}}
			>
				{showAddForm ? $i18n.t('Cancel') : $i18n.t('Add Rate')}
			</button>
		</div>
	</div>

	<!-- Description -->
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t(
			'Monthly exchange rates for converting foreign currency transactions to your company currency.'
		)}
	</div>

	<!-- Add Rate Form -->
	{#if showAddForm}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30 mb-3"
		>
			<div class="text-sm font-medium dark:text-gray-200 mb-3">
				{$i18n.t('Add Exchange Rate')}
			</div>
			<div class="grid grid-cols-1 md:grid-cols-5 gap-3 items-end">
				<div>
					<label
						for="rate-from-currency"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('From Currency')} *
					</label>
					<select
						id="rate-from-currency"
						bind:value={newFromCurrency}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					>
						{#each CURRENCIES as cur}
							<option value={cur}>{cur}</option>
						{/each}
					</select>
				</div>
				<div>
					<label
						for="rate-to-currency"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('To Currency')} *
					</label>
					<select
						id="rate-to-currency"
						bind:value={newToCurrency}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					>
						{#each CURRENCIES as cur}
							<option value={cur}>{cur}</option>
						{/each}
					</select>
				</div>
				<div>
					<label
						for="rate-value"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Rate')} *
					</label>
					<input
						id="rate-value"
						type="number"
						step="0.000001"
						min="0"
						bind:value={newRate}
						placeholder="1.0850"
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="rate-effective-date"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Effective Date')} *
					</label>
					<input
						id="rate-effective-date"
						type="date"
						bind:value={newEffectiveDate}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
					on:click={handleCreate}
					disabled={creating || !newFromCurrency || !newToCurrency || !newRate || !newEffectiveDate}
				>
					{creating ? $i18n.t('Saving...') : $i18n.t('Save')}
				</button>
			</div>
		</div>
	{/if}

	<!-- Rates Table -->
	{#if loading}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if rates.length === 0}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-8 border border-gray-100/30 dark:border-gray-850/30 text-center"
		>
			<div class="text-gray-400 dark:text-gray-500 text-sm mb-3">
				{$i18n.t('No exchange rates defined yet.')}
			</div>
			<div class="text-gray-400 dark:text-gray-500 text-xs">
				{$i18n.t(
					'Add exchange rates to convert foreign currency transactions to your company currency.'
				)}
			</div>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead
					class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
				>
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
						<tr
							class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
						>
							<td class="px-3 py-2 font-medium dark:text-gray-200">
								{formatDate(rate.effective_date)}
							</td>
							<td class="px-3 py-2">
								<span
									class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
								>
									{rate.from_currency}
								</span>
							</td>
							<td class="px-3 py-2">
								<span
									class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300"
								>
									{rate.to_currency}
								</span>
							</td>
							<td class="px-3 py-2 text-right font-mono">
								{formatRate(rate.rate)}
							</td>
							<td class="px-3 py-2">
								{#if rate.source === 'manual' || rate.source === 'Manual'}
									<span
										class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-500/20 dark:text-purple-200"
									>
										{$i18n.t('Manual')}
									</span>
								{:else}
									<span
										class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300"
									>
										{rate.source ? $i18n.t(rate.source) : $i18n.t('Import')}
									</span>
								{/if}
							</td>
							<td class="px-3 py-2 text-right">
								<Tooltip content={$i18n.t('Delete this exchange rate')}>
									<button
										class="px-3 py-1 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 dark:hover:bg-red-900/40 transition"
										on:click={() => confirmDelete(rate)}
									>
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
