<script lang="ts">
	import { getContext, onMount } from 'svelte';

	import { getTaxAccounts } from '$lib/apis/accounting';
	import TaxDeclaration from './TaxDeclaration.svelte';
	import IsDeclaration from './IsDeclaration.svelte';
	import IrDeclaration from './IrDeclaration.svelte';
	import TaxAccountsSettings from './TaxAccountsSettings.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	type TaxTab = 'vat' | 'is' | 'ir' | 'accounts';
	let activeTab: TaxTab = 'vat';

	const tabs: Array<{ id: TaxTab; label: string }> = [
		{ id: 'vat', label: 'VAT' },
		{ id: 'is', label: 'IS' },
		{ id: 'ir', label: 'IR' },
		{ id: 'accounts', label: 'Accounts' }
	];

	// Count of unmapped roles — shown on the Accounts tab so a blocked
	// settlement entry has an obvious place to go.
	let missing = 0;
	const refreshMissing = async () => {
		try {
			const map = await getTaxAccounts(companyId);
			missing = (map.roles ?? []).filter((r: any) => r.source === 'missing').length;
		} catch {
			missing = 0;
		}
	};
	onMount(refreshMissing);

	// Remount the declaration tabs after a mapping change so they re-resolve.
	let mapVersion = 0;
	const onSaved = () => {
		mapVersion += 1;
		refreshMissing();
	};
</script>

<div class="py-3 space-y-4">
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
</div>
