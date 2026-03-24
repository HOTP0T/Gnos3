<script lang="ts">
	import { onMount, getContext, setContext } from 'svelte';
	import { writable } from 'svelte/store';
	import { WEBUI_NAME, mobile, showSidebar } from '$lib/stores';
	import { page } from '$app/stores';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Sidebar from '$lib/components/icons/Sidebar.svelte';
	import { getCompany, getExchangeRates } from '$lib/apis/accounting';

	const i18n = getContext('i18n');

	let loaded = false;
	let companyName = '';
	let companyCurrency = 'EUR';

	// Global display currency — shared via context to all child components
	const displayCurrency = writable('');
	const exchangeRates = writable<any[]>([]);
	const ratesLoaded = writable(false);
	const companyCurrencyStore = writable('EUR');

	setContext('displayCurrency', displayCurrency);
	setContext('exchangeRates', exchangeRates);
	setContext('ratesLoaded', ratesLoaded);
	setContext('companyCurrency', companyCurrencyStore);

	$: companyId = parseInt($page.params.id, 10);
	$: basePath = `/accounting/company/${$page.params.id}`;

	setContext('companyId', {
		get id() {
			return companyId;
		}
	});

	const CURRENCIES = ['EUR', 'USD', 'GBP', 'CNY', 'JPY', 'CAD', 'AUD', 'CHF', 'BRL', 'INR', 'KRW', 'MXN', 'SGD', 'HKD', 'MAD'];

	const defaultTabs = [
		{ key: 'dashboard', label: 'Dashboard' },
		{ key: 'bank', label: 'Bank' },
		{ key: 'invoices', label: 'Invoices' },
		{ key: 'entries', label: 'Entries' },
		{ key: 'payments', label: 'Payments' },
		{ key: 'reports', label: 'Reports' },
		{ key: 'tax', label: 'Tax' },
		{ key: 'assets', label: 'Assets' },
		{ key: 'closing', label: 'Closing' },
		{ key: 'settings', label: 'Settings' }
	];

	// Load saved tab order from localStorage, fallback to default
	const TAB_ORDER_KEY = `accounting-tab-order-${companyId}`;
	let tabs = defaultTabs;

	function loadTabOrder() {
		try {
			const saved = localStorage.getItem(TAB_ORDER_KEY);
			if (saved) {
				const savedKeys: string[] = JSON.parse(saved);
				// Rebuild tabs in saved order, adding any new tabs not in saved order
				const tabMap = new Map(defaultTabs.map(t => [t.key, t]));
				const ordered: typeof defaultTabs = [];
				for (const key of savedKeys) {
					const tab = tabMap.get(key);
					if (tab) {
						ordered.push(tab);
						tabMap.delete(key);
					}
				}
				// Append any new tabs not in saved order
				for (const tab of tabMap.values()) {
					ordered.push(tab);
				}
				tabs = ordered;
			}
		} catch {}
	}

	function saveTabOrder() {
		try {
			localStorage.setItem(TAB_ORDER_KEY, JSON.stringify(tabs.map(t => t.key)));
		} catch {}
	}

	// Drag and drop state
	let dragIdx: number | null = null;
	let dragOverIdx: number | null = null;

	function handleDragStart(e: DragEvent, idx: number) {
		dragIdx = idx;
		if (e.dataTransfer) {
			e.dataTransfer.effectAllowed = 'move';
			e.dataTransfer.setData('text/plain', String(idx));
		}
	}

	function handleDragOver(e: DragEvent, idx: number) {
		e.preventDefault();
		if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
		dragOverIdx = idx;
	}

	function handleDrop(e: DragEvent, idx: number) {
		e.preventDefault();
		if (dragIdx === null || dragIdx === idx) {
			dragIdx = null;
			dragOverIdx = null;
			return;
		}
		// Reorder
		const newTabs = [...tabs];
		const [moved] = newTabs.splice(dragIdx, 1);
		newTabs.splice(idx, 0, moved);
		tabs = newTabs;
		saveTabOrder();
		dragIdx = null;
		dragOverIdx = null;
	}

	function handleDragEnd() {
		dragIdx = null;
		dragOverIdx = null;
	}

	// Check if we have rates for a given currency pair
	// Check if we have any rate for the selected currency pair
	$: hasRatesForDisplay = $displayCurrency && $displayCurrency !== companyCurrency && $exchangeRates.length > 0
		? $exchangeRates.some(r =>
			(r.from_currency === companyCurrency && r.to_currency === $displayCurrency) ||
			(r.from_currency === $displayCurrency && r.to_currency === companyCurrency)
		)
		: true;

	// Check if we have a rate for the CURRENT month specifically
	$: currentMonthKey = new Date().toISOString().slice(0, 7); // "2026-03"
	$: hasCurrentMonthRate = !($displayCurrency && $displayCurrency !== companyCurrency) || $exchangeRates.some(r =>
		r.effective_date?.startsWith(currentMonthKey) &&
		((r.from_currency === companyCurrency && r.to_currency === $displayCurrency) ||
		 (r.from_currency === $displayCurrency && r.to_currency === companyCurrency))
	);

	async function loadCompanyData() {
		try {
			const data = await getCompany(companyId);
			companyName = data.name || '';
			companyCurrency = data.currency || 'EUR';
			companyCurrencyStore.set(companyCurrency);
			displayCurrency.set(companyCurrency);

			// Load exchange rates
			try {
				const rates = await getExchangeRates({ company_id: companyId });
				exchangeRates.set(Array.isArray(rates) ? rates : []);
				ratesLoaded.set(true);
			} catch {
				exchangeRates.set([]);
				ratesLoaded.set(true);
			}
		} catch (err) {
			console.error('Failed to load company:', err);
		}
	}

	onMount(async () => {
		loadTabOrder();
		await loadCompanyData();
		loaded = true;
	});

	function handleCurrencyChange(e: Event) {
		const val = (e.target as HTMLSelectElement).value;
		displayCurrency.set(val);
	}
</script>

<svelte:head>
	<title>
		{companyName ? `${companyName} - ` : ''}{$i18n.t('Accounting')} &bull; {$WEBUI_NAME}
	</title>
</svelte:head>

{#if loaded}
	<div class="flex flex-col h-screen max-h-[100dvh] flex-1 transition-width duration-200 ease-in-out {$showSidebar ? 'md:max-w-[calc(100%-var(--sidebar-width))]' : 'md:max-w-[calc(100%-49px)]'} w-full max-w-full">
		<nav class="px-2.5 pt-1.5 backdrop-blur-xl drag-region">
			<div class="flex items-center gap-1">
				{#if $mobile}
					<div
						class="{$showSidebar
							? 'md:hidden'
							: ''} self-center flex flex-none items-center"
					>
						<Tooltip
							content={$showSidebar
								? $i18n.t('Close Sidebar')
								: $i18n.t('Open Sidebar')}
							interactive={true}
						>
							<button
								id="sidebar-toggle-button"
								class="cursor-pointer flex rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition"
								on:click={() => {
									showSidebar.set(!$showSidebar);
								}}
							>
								<div class="self-center p-1.5">
									<Sidebar />
								</div>
							</button>
						</Tooltip>
					</div>
				{/if}

				<div class="flex items-center gap-1.5 flex-1 min-w-0">
					<a
						class="text-sm text-gray-400 dark:text-gray-500 hover:text-gray-700 dark:hover:text-white transition whitespace-nowrap"
						href="/accounting/companies"
					>
						&larr; {$i18n.t('Companies')}
					</a>

					{#if companyName}
						<span class="text-gray-300 dark:text-gray-600 text-sm">/</span>
						<span class="text-base font-semibold truncate">{companyName}</span>
					{/if}

					<!-- Global Currency Switcher -->
					<div class="flex items-center gap-1 ml-2 flex-shrink-0">
						<select
							class="text-[11px] rounded-lg px-1.5 py-0.5 border
								{$displayCurrency !== companyCurrency
									? 'border-blue-400 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium'
									: 'border-gray-200 dark:border-gray-700 bg-transparent text-gray-500 dark:text-gray-400'}
								outline-none transition"
							value={$displayCurrency}
							on:change={handleCurrencyChange}
							title={$i18n.t('View amounts in a different currency')}
						>
							{#each CURRENCIES as cur}
								<option value={cur}>{cur}{cur === companyCurrency ? ' ●' : ''}</option>
							{/each}
						</select>

						{#if $displayCurrency !== companyCurrency}
							{#if !hasRatesForDisplay}
								<!-- No rates warning -->
								<Tooltip content={$i18n.t('No exchange rates found for') + ` ${companyCurrency}→${$displayCurrency}. ` + $i18n.t('Import rates in Settings → Exchange Rates.')}>
									<span class="text-amber-500 dark:text-amber-400 text-xs cursor-help" title="">
										<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-3.5 h-3.5">
											<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
										</svg>
									</span>
								</Tooltip>
							{:else}
								<!-- Converted indicator -->
								<span class="text-[10px] text-blue-500 dark:text-blue-400 whitespace-nowrap">
									{companyCurrency}→{$displayCurrency}
								</span>
							{/if}

							<button
								class="text-[10px] text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition"
								on:click={() => displayCurrency.set(companyCurrency)}
								title={$i18n.t('Reset to company currency')}
							>
								✕
							</button>
						{/if}
					</div>
				</div>

				<div class="">
					<div
						class="flex gap-1 scrollbar-none overflow-x-auto w-fit text-center text-sm font-medium rounded-full bg-transparent py-1"
					>
						{#each tabs as tab, idx (tab.key)}
							<!-- svelte-ignore a11y-no-static-element-interactions -->
							<a
								class="min-w-fit p-1.5 select-none cursor-grab active:cursor-grabbing rounded-md transition
									{$page.url.pathname.includes(`${basePath}/${tab.key}`)
										? 'font-semibold'
										: 'text-gray-300 dark:text-gray-600 hover:text-gray-700 dark:hover:text-white'}
									{dragOverIdx === idx && dragIdx !== idx ? 'ring-2 ring-blue-400 ring-offset-1' : ''}
									{dragIdx === idx ? 'opacity-40' : ''}"
								href="{basePath}/{tab.key}"
								draggable="true"
								on:dragstart={(e) => handleDragStart(e, idx)}
								on:dragover={(e) => handleDragOver(e, idx)}
								on:drop={(e) => handleDrop(e, idx)}
								on:dragend={handleDragEnd}
							>{$i18n.t(tab.label)}</a>
						{/each}
					</div>
				</div>
			</div>
		</nav>

		<!-- Current month rate warning -->
		{#if $displayCurrency !== companyCurrency && hasRatesForDisplay && !hasCurrentMonthRate}
			<div class="mx-3 mt-1 px-3 py-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800/50 rounded-lg text-xs text-blue-700 dark:text-blue-300 flex items-center gap-2">
				<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-4 h-4 flex-shrink-0">
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z" />
				</svg>
				<span>
					{$i18n.t('No exchange rate for the current month')} ({currentMonthKey}).
					{$i18n.t('Conversion uses the most recent available rate.')}
					<a href="{basePath}/settings" class="underline font-medium">{$i18n.t('Update rates')}</a>
				</span>
			</div>
		{/if}

		<!-- No rates banner (shown once, below nav) -->
		{#if $displayCurrency !== companyCurrency && !hasRatesForDisplay}
			<div class="mx-3 mt-1 px-3 py-2 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800/50 rounded-lg text-xs text-amber-700 dark:text-amber-300 flex items-center gap-2">
				<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-4 h-4 flex-shrink-0">
					<path stroke-linecap="round" stroke-linejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126ZM12 15.75h.007v.008H12v-.008Z" />
				</svg>
				<span>
					{$i18n.t('No exchange rates found for')} <strong>{companyCurrency}→{$displayCurrency}</strong>.
					{$i18n.t('Amounts cannot be converted.')}
					<a href="{basePath}/settings" class="underline font-medium hover:text-amber-900 dark:hover:text-amber-100">{$i18n.t('Import rates in Settings → Exchange Rates')}</a>
				</span>
			</div>
		{/if}

		<div class="pb-1 px-3 md:px-[18px] flex-1 max-h-full overflow-y-auto" id="accounting-container">
			<slot />
		</div>
	</div>
{/if}
