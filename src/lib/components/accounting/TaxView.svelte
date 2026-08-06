<script lang="ts">
	import { getContext } from 'svelte';

	import TaxDeclaration from './TaxDeclaration.svelte';
	import IsDeclaration from './IsDeclaration.svelte';
	import IrDeclaration from './IrDeclaration.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	type TaxTab = 'vat' | 'is' | 'ir';
	let activeTab: TaxTab = 'vat';

	const tabs: Array<{ id: TaxTab; label: string }> = [
		{ id: 'vat', label: 'VAT' },
		{ id: 'is', label: 'IS' },
		{ id: 'ir', label: 'IR' }
	];
</script>

<div class="py-3 space-y-4">
	<!-- Sub-tab bar -->
	<div class="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1 w-fit">
		{#each tabs as tab}
			<button
				class="px-4 py-2 text-sm font-medium rounded-md transition
					{activeTab === tab.id
					? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
					: 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300'}"
				on:click={() => (activeTab = tab.id)}
			>
				{$i18n.t(tab.label)}
			</button>
		{/each}
	</div>

	{#if activeTab === 'vat'}
		<TaxDeclaration {companyId} />
	{:else if activeTab === 'is'}
		<IsDeclaration {companyId} />
	{:else if activeTab === 'ir'}
		<IrDeclaration {companyId} />
	{/if}
</div>
