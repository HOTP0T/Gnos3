<script lang="ts">
	import { onMount, getContext } from 'svelte';

	import { getTaxConfig } from '$lib/apis/accounting';
	import TaxFilingsList from './TaxFilingsList.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	let cfg: any = null;

	onMount(async () => {
		try {
			cfg = await getTaxConfig(companyId);
		} catch {
			cfg = null;
		}
	});

	const pctOf = (r: any) =>
		r === null || r === undefined ? '—' : `${(Number(r) * 100).toFixed(2)}%`;
</script>

<div class="py-2">
	<div class="flex md:self-center text-lg font-medium px-0.5 gap-2 mb-1">
		<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Individual Income Tax (IR)')}</div>
	</div>

	<div
		class="bg-amber-50 dark:bg-amber-900/20 border border-amber-200/50 dark:border-amber-800/30 rounded-xl p-4 mb-3 text-sm text-amber-800 dark:text-amber-200"
	>
		{$i18n.t(
			'The IR worksheet is a scaffold. There is no reference calculation yet — provide a sample IR sheet and it will be built out like the VAT and IS tabs.'
		)}
	</div>

	<div
		class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30 mb-3"
	>
		<div class="text-sm dark:text-gray-200">
			{$i18n.t('Configured IR rate')}:
			<span class="font-semibold">{pctOf(cfg?.ir_rate)}</span>
		</div>
		<div class="text-xs text-gray-400 dark:text-gray-500 mt-1">
			{$i18n.t('Set per country in Settings → Countries.')}
		</div>
	</div>

	<TaxFilingsList {companyId} taxType="ir" />
</div>
