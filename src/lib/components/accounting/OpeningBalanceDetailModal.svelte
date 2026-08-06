<script lang="ts">
	import { getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';
	import Modal from '$lib/components/common/Modal.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import {
		getArApAccounts,
		getOpeningBalanceDetails,
		setOpeningBalanceDetails,
		importOpeningBalanceDetails,
		downloadOpeningBalanceDetailTemplate
	} from '$lib/apis/accounting';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let companyId: number;

	let loading = false;
	let saving = false;
	let importing = false;
	let arApAccounts: any[] = []; // flattened ar + ap with side
	let selectedAccountId: number | null = null;
	let rows: any[] = [];
	let fileInput: HTMLInputElement;

	$: selectedAccount = arApAccounts.find((a) => a.id === selectedAccountId) ?? null;
	$: detailTotal = rows.reduce((s, r) => s + (parseFloat(r.amount) || 0), 0);
	$: openingLump = selectedAccount ? Number(selectedAccount.opening) : 0;
	$: delta = Math.round((detailTotal - openingLump) * 100) / 100;

	$: if (show && companyId && arApAccounts.length === 0) loadAccounts();

	const loadAccounts = async () => {
		loading = true;
		try {
			const res = await getArApAccounts(companyId);
			arApAccounts = [
				...(res.ar ?? []).map((a: any) => ({ ...a, side: 'ar' })),
				...(res.ap ?? []).map((a: any) => ({ ...a, side: 'ap' }))
			];
			if (arApAccounts.length && selectedAccountId == null) {
				selectedAccountId = arApAccounts[0].id;
				await loadDetails();
			}
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const loadDetails = async () => {
		if (!selectedAccountId) return;
		loading = true;
		try {
			const res = await getOpeningBalanceDetails(companyId, selectedAccountId);
			rows = (res ?? []).map((d: any) => ({
				party_name: d.party_name,
				reference: d.reference ?? '',
				amount: d.amount,
				item_date: d.item_date ?? '',
				due_date: d.due_date ?? ''
			}));
		} catch (err) {
			toast.error(`${err}`);
		}
		loading = false;
	};

	const onAccountChange = () => loadDetails();

	const addRow = () =>
		(rows = [...rows, { party_name: '', reference: '', amount: null, item_date: '', due_date: '' }]);
	const removeRow = (i: number) => (rows = rows.filter((_, idx) => idx !== i));

	const save = async () => {
		if (!selectedAccountId) return;
		const clean = rows
			.filter((r) => r.party_name && parseFloat(r.amount) > 0)
			.map((r) => ({
				party_name: r.party_name,
				reference: r.reference || null,
				amount: parseFloat(r.amount),
				item_date: r.item_date || null,
				due_date: r.due_date || null
			}));
		saving = true;
		try {
			await setOpeningBalanceDetails(companyId, selectedAccountId, clean);
			toast.success($i18n.t('Opening balance detail saved'));
			await loadAccounts();
			dispatch('save');
		} catch (err) {
			toast.error(`${err}`);
		}
		saving = false;
	};

	const onImport = async (e: Event) => {
		const file = (e.target as HTMLInputElement).files?.[0];
		if (!file) return;
		importing = true;
		try {
			const res = await importOpeningBalanceDetails(companyId, file);
			toast.success($i18n.t('Imported') + `: ${res.imported}, ` + $i18n.t('skipped') + `: ${res.skipped}`);
			arApAccounts = [];
			await loadAccounts();
			await loadDetails();
		} catch (err) {
			toast.error(`${err}`);
		}
		importing = false;
		if (fileInput) fileInput.value = '';
	};
</script>

<Modal bind:show size="lg">
	<div class="px-6 py-5">
		<div class="flex items-center justify-between mb-4">
			<h2 class="text-lg font-medium dark:text-gray-200">{$i18n.t('Opening Balance Detail (AR/AP)')}</h2>
			<button class="p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition" on:click={() => (show = false)}>✕</button>
		</div>

		<p class="text-xs text-gray-500 dark:text-gray-400 mb-3">
			{$i18n.t('Break an AR/AP account opening balance into per-customer/vendor lines so they appear individually in the aging report.')}
		</p>

		{#if loading && arApAccounts.length === 0}
			<div class="flex justify-center my-8"><Spinner className="size-5" /></div>
		{:else if arApAccounts.length === 0}
			<div class="text-sm text-gray-400 italic py-6 text-center">{$i18n.t('No AR/AP accounts found for this company.')}</div>
		{:else}
			<div class="flex flex-wrap items-end gap-3 mb-3">
				<div class="grow">
					<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Account')}</label>
					<select bind:value={selectedAccountId} on:change={onAccountChange} class="w-full text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800">
						{#each arApAccounts as a}
							<option value={a.id}>{a.side.toUpperCase()} · {a.code} — {a.name}</option>
						{/each}
					</select>
				</div>
				<button class="px-3 py-1.5 text-xs rounded-lg text-blue-600 dark:text-blue-400 hover:underline" on:click={downloadOpeningBalanceDetailTemplate}>
					{$i18n.t('Download template')}
				</button>
				<button class="px-3 py-1.5 text-xs rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-850 disabled:opacity-50" on:click={() => fileInput?.click()} disabled={importing}>
					{#if importing}<Spinner className="size-3.5" />{/if} {$i18n.t('Import Excel')}
				</button>
				<input type="file" accept=".xlsx,.xls" class="hidden" bind:this={fileInput} on:change={onImport} />
			</div>

			<!-- Reconciliation banner -->
			<div class="mb-3 text-xs flex items-center gap-4 px-3 py-2 rounded-lg {Math.abs(delta) < 0.01 ? 'bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-300' : 'bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-300'}">
				<span>{$i18n.t('Account opening')}: <b class="font-mono">{openingLump.toFixed(2)}</b></span>
				<span>{$i18n.t('Detail total')}: <b class="font-mono">{detailTotal.toFixed(2)}</b></span>
				<span>{$i18n.t('Δ')}: <b class="font-mono">{delta.toFixed(2)}</b></span>
				{#if Math.abs(delta) >= 0.01}<span class="italic">{$i18n.t('Detail lines should sum to the account opening balance.')}</span>{/if}
			</div>

			<div class="overflow-x-auto border border-gray-100 dark:border-gray-850 rounded-lg">
				<table class="w-full text-xs">
					<thead class="bg-gray-50 dark:bg-gray-850/50 text-gray-500 dark:text-gray-400 text-[10px] uppercase">
						<tr>
							<th class="px-2 py-2 text-left">{$i18n.t('Customer / Vendor')}</th>
							<th class="px-2 py-2 text-left">{$i18n.t('Reference')}</th>
							<th class="px-2 py-2 text-right">{$i18n.t('Amount')}</th>
							<th class="px-2 py-2 text-left">{$i18n.t('Item Date')}</th>
							<th class="px-2 py-2 text-left">{$i18n.t('Due Date')}</th>
							<th class="px-2 py-2"></th>
						</tr>
					</thead>
					<tbody>
						{#each rows as r, i}
							<tr class="border-t border-gray-50 dark:border-gray-850/40">
								<td class="px-2 py-1"><input bind:value={r.party_name} class="w-full bg-transparent border border-gray-200 dark:border-gray-700 rounded px-2 py-1 dark:text-gray-200" /></td>
								<td class="px-2 py-1"><input bind:value={r.reference} class="w-full bg-transparent border border-gray-200 dark:border-gray-700 rounded px-2 py-1 dark:text-gray-200" /></td>
								<td class="px-2 py-1"><input type="number" step="0.01" bind:value={r.amount} class="w-28 text-right bg-transparent border border-gray-200 dark:border-gray-700 rounded px-2 py-1 dark:text-gray-200 font-mono" /></td>
								<td class="px-2 py-1"><input type="date" bind:value={r.item_date} class="bg-transparent border border-gray-200 dark:border-gray-700 rounded px-2 py-1 dark:text-gray-200" /></td>
								<td class="px-2 py-1"><input type="date" bind:value={r.due_date} class="bg-transparent border border-gray-200 dark:border-gray-700 rounded px-2 py-1 dark:text-gray-200" /></td>
								<td class="px-2 py-1 text-center"><button class="text-red-400 hover:text-red-600" on:click={() => removeRow(i)}>✕</button></td>
							</tr>
						{/each}
						{#if rows.length === 0}
							<tr><td colspan="6" class="px-2 py-4 text-center text-gray-400 italic">{$i18n.t('No detail lines yet.')}</td></tr>
						{/if}
					</tbody>
				</table>
			</div>

			<div class="flex items-center justify-between mt-3">
				<button class="px-3 py-1.5 text-xs rounded-lg border border-dashed border-gray-300 dark:border-gray-600 text-gray-500 hover:text-blue-600 hover:border-blue-400" on:click={addRow}>+ {$i18n.t('Add line')}</button>
				<button class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50" on:click={save} disabled={saving}>
					{#if saving}<Spinner className="size-3.5" />{/if} {$i18n.t('Save')}
				</button>
			</div>
		{/if}
	</div>
</Modal>
