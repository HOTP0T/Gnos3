<script lang="ts">
	import { getContext, onMount } from 'svelte';

	import { getTaxAccounts, getTaxConfig } from '$lib/apis/accounting';
	import TaxDeclaration from './TaxDeclaration.svelte';
	import IsDeclaration from './IsDeclaration.svelte';
	import IrDeclaration from './IrDeclaration.svelte';
	import TaxAccountsSettings from './TaxAccountsSettings.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	type TaxTab = 'vat' | 'is' | 'ir' | 'accounts';
	let activeTab: TaxTab = 'vat';

	// The country decides which taxes exist: Hong Kong has no VAT and no salary
	// withholding, so those tabs are not shown at all, and the income-tax tab
	// takes the country's own name (IS / 企业所得税 / Profits Tax).
	// undefined = still loading (nothing rendered), null = could not load (all tabs shown).
	let cfg: any = undefined;
	$: hasVat = cfg ? cfg.has_vat !== false : true;
	$: withholdsIit = cfg ? cfg.withholds_iit !== false : true;
	$: citLabel = cfg?.cit_label || 'IS';
	$: tabs = [
		...(hasVat ? [{ id: 'vat' as TaxTab, label: cfg?.tax_name || 'VAT' }] : []),
		{ id: 'is' as TaxTab, label: citLabel },
		...(withholdsIit ? [{ id: 'ir' as TaxTab, label: 'IR' }] : []),
		{ id: 'accounts' as TaxTab, label: 'Accounts' }
	];
	$: if (cfg && !tabs.some((t) => t.id === activeTab)) activeTab = tabs[0].id;

	// Count of unmapped roles that exist in this jurisdiction — shown on the
	// Accounts tab so a blocked settlement entry has an obvious place to go.
	let missing = 0;
	const refreshMissing = async () => {
		try {
			const map = await getTaxAccounts(companyId);
			missing = (map.roles ?? []).filter((r: any) => r.source === 'missing' && r.applicable !== false).length;
		} catch {
			missing = 0;
		}
	};
	onMount(async () => {
		try {
			cfg = await getTaxConfig(companyId);
			if (cfg?.has_vat === false) activeTab = 'is';
		} catch {
			cfg = null;
		}
		await refreshMissing();
	});

	// Remount the declaration tabs after a mapping change so they re-resolve.
	let mapVersion = 0;
	const onSaved = () => {
		mapVersion += 1;
		refreshMissing();
	};
</script>

<div class="py-3 space-y-4">
	{#if cfg === undefined}
		<div class="text-xs text-gray-400 py-4">{$i18n.t('Loading...')}</div>
	{:else}
	<!-- Sub-tab bar -->
	<div class="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 w-fit">
		{#each tabs as tab}
			<button
				class="px-4 py-2 text-sm font-medium rounded-md transition flex items-center gap-1.5
					{activeTab === tab.id
					? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
					: 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'}"
				on:click={() => (activeTab = tab.id)}
			>
				{$i18n.t(tab.label)}
				{#if tab.id === 'accounts' && missing > 0}
					<span
						class="px-1.5 py-0.5 rounded-full text-[10px] font-semibold bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300"
						title={$i18n.t('Unmapped tax roles')}>{missing}</span
					>
				{/if}
			</button>
		{/each}
	</div>

	{#if cfg && (!hasVat || !withholdsIit)}
		<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5">
			{#if !hasVat}
				{$i18n.t('{{country}} levies no VAT / GST: there is no VAT return, and any tax printed on a supplier bill is booked as part of the cost.', { country: cfg.country })}
			{/if}
			{#if !withholdsIit}
				{$i18n.t('Employers do not withhold Salaries Tax — employees settle it themselves; the employer files the yearly Employer\'s Return (BIR56A / IR56B) in April.')}
			{/if}
		</div>
	{/if}

	{#key mapVersion}
		{#if activeTab === 'vat'}
			<TaxDeclaration {companyId} on:gotoAccounts={() => (activeTab = 'accounts')} />
		{:else if activeTab === 'is'}
			<IsDeclaration {companyId} on:gotoAccounts={() => (activeTab = 'accounts')} />
		{:else if activeTab === 'ir'}
			<IrDeclaration {companyId} />
		{:else if activeTab === 'accounts'}
			<TaxAccountsSettings {companyId} on:saved={onSaved} />
		{/if}
	{/key}
	{/if}
</div>
