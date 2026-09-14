<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getCompany, updateCompany, getAccounts } from '$lib/apis/accounting';

	const i18n = getContext('i18n');

	export let companyId: number;

	let loading = true;
	let saving = false;
	let accounts: any[] = [];
	let apId: number | '' = '';
	let arId: number | '' = '';
	let savedAp: number | '' = '';
	let savedAr: number | '' = '';

	$: parentIds = new Set(accounts.map((a: any) => a.parent_id).filter(Boolean));
	$: leaf = accounts.filter((a: any) => !parentIds.has(a.id));
	$: liabilities = leaf.filter((a: any) => a.account_type === 'liability');
	$: assets = leaf.filter((a: any) => a.account_type === 'asset');
	$: dirty = apId !== savedAp || arId !== savedAr;

	onMount(async () => {
		try {
			const [co, accts] = await Promise.all([getCompany(companyId), getAccounts({ company_id: companyId, active: true })]);
			accounts = Array.isArray(accts) ? accts : (accts?.items ?? accts?.accounts ?? []);
			apId = savedAp = co?.default_ap_account_id ?? '';
			arId = savedAr = co?.default_ar_account_id ?? '';
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to load')}: ${err?.detail ?? err}`);
		}
		loading = false;
	});

	const save = async () => {
		saving = true;
		try {
			await updateCompany(companyId, {
				default_ap_account_id: apId === '' ? null : Number(apId),
				default_ar_account_id: arId === '' ? null : Number(arId)
			});
			savedAp = apId;
			savedAr = arId;
			toast.success($i18n.t('Booking defaults saved'));
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	const selectCls =
		'text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden max-w-full';
</script>

<div class="space-y-3">
	<p class="text-xs text-gray-400 dark:text-gray-500">
		{$i18n.t(
			'Counterparty accounts used when an invoice is booked and no categorisation rule names one for the vendor. Nothing is guessed: with no rule and no default, the entry is created with the counterparty line left open for you to fill.'
		)}
	</p>
	{#if loading}
		<div class="text-xs text-gray-400">{$i18n.t('Loading...')}</div>
	{:else}
		<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
			<div>
				<label for="bd-ap" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Default supplier (AP) account — purchases')}</label>
				<select id="bd-ap" class={selectCls} bind:value={apId}>
					<option value="">{$i18n.t('— none: ask every time —')}</option>
					{#each liabilities as a}<option value={a.id}>{a.code} - {a.name}</option>{/each}
				</select>
			</div>
			<div>
				<label for="bd-ar" class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Default customer (AR) account — sales')}</label>
				<select id="bd-ar" class={selectCls} bind:value={arId}>
					<option value="">{$i18n.t('— none: ask every time —')}</option>
					{#each assets as a}<option value={a.id}>{a.code} - {a.name}</option>{/each}
				</select>
			</div>
		</div>
		<button
			class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
			disabled={saving || !dirty}
			on:click={save}
		>
			{saving ? $i18n.t('Saving...') : $i18n.t('Save')}
		</button>
	{/if}
</div>
