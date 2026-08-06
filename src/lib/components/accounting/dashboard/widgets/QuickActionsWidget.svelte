<script lang="ts">
	// Quick action buttons.
	import { getContext } from 'svelte';
	import { goto } from '$app/navigation';
	export let companyId: number;
	export let options: Record<string, any> = {}; // unused; part of widget contract
	const i18n: any = getContext('i18n');
	const T = (k: string) => (i18n?.t ? i18n.t(k) : k);
</script>

<div class="h-full flex flex-wrap items-center gap-2 content-center">
	<button class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition"
		on:click={() => goto(`/accounting/company/${companyId}/entries`)}>
		{T('New Journal Entry')}
	</button>
	<button class="px-3 py-1.5 text-xs font-medium rounded-lg bg-purple-600 text-white hover:bg-purple-700 dark:bg-purple-500 dark:hover:bg-purple-600 transition"
		on:click={() => goto(`/accounting/company/${companyId}/payments`)}>
		{T('Record Payment')}
	</button>
	<button class="px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-850 transition"
		on:click={() => goto(`/accounting/company/${companyId}/reports`)}>
		{T('Reports')}
	</button>
	<button class="px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-850 transition"
		on:click={() => goto(`/accounting/company/${companyId}/bank`)}>
		{T('Reconcile Bank')}
	</button>
</div>
