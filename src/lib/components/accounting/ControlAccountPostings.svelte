<script lang="ts">
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getControlAccountPostings, applyReclassification } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let companyId: number;

	let loading = true;
	let working = false;
	let data: any = null;
	// line_id → chosen target code. Nothing is pre-filled: the human decides.
	let targets: Record<number, string> = {};
	let perAccount: Record<string, string> = {};
	let preview: any = null;

	const fmt = (v: any) => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		return n ? n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) : '';
	};

	const load = async () => {
		loading = true;
		preview = null;
		try {
			data = await getControlAccountPostings(companyId);
			targets = {};
			perAccount = {};
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to load')}: ${err?.detail ?? err}`);
			data = null;
		}
		loading = false;
	};
	onMount(load);

	$: groups = (data?.by_account ?? []) as any[];
	$: postingsByAccount = (code: string) => (data?.postings ?? []).filter((p: any) => p.account_code === code);
	$: chosen = Object.entries(targets).filter(([, v]) => !!v).length;

	const applyToAccount = (code: string) => {
		const t = perAccount[code];
		if (!t) return;
		for (const p of postingsByAccount(code)) targets[p.line_id] = t;
		targets = { ...targets };
		preview = null;
	};

	const mappings = () =>
		Object.entries(targets)
			.filter(([, v]) => !!v)
			.map(([line_id, target_code]) => ({ line_id: Number(line_id), target_code }));

	const run = async (dryRun: boolean) => {
		if (chosen === 0) return;
		working = true;
		try {
			const res = await applyReclassification(companyId, mappings(), dryRun);
			if (dryRun) {
				preview = res;
				if (res.errors?.length) toast.error($i18n.t('{{n}} line(s) cannot be moved — see below', { n: res.errors.length }));
				else toast.success($i18n.t('{{n}} line(s) ready to move', { n: res.changes?.length ?? 0 }));
			} else {
				toast.success($i18n.t('{{n}} line(s) moved', { n: res.applied ?? 0 }));
				await load();
				dispatch('applied');
			}
		} catch (err: any) {
			toast.error(`${err?.detail ?? err}`);
		}
		working = false;
	};

	const selectCls =
		'text-xs rounded-lg px-2 py-1 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden max-w-full';
</script>

<div class="space-y-3">
	<p class="text-xs text-gray-400 dark:text-gray-500">
		{$i18n.t(
			'Postings that sit on a control account (one with sub-accounts) never appear in the sub-ledger. Choose the sub-account each line belongs on — per line, or once per account — then preview and apply. Nothing is chosen for you; a voided entry and its reversal should move together.'
		)}
	</p>

	{#if loading}
		<div class="flex justify-center my-6"><Spinner className="size-5" /></div>
	{:else if !data || data.total_lines === 0}
		<div class="text-sm text-gray-500 dark:text-gray-400">{$i18n.t('No postings on control accounts.')}</div>
	{:else}
		{#each groups as g (g.account_code)}
			<div class="rounded-xl border border-gray-200 dark:border-gray-800 overflow-hidden">
				<div class="flex flex-wrap items-center justify-between gap-2 px-3 py-2 bg-gray-50 dark:bg-gray-850/50">
					<div class="text-sm font-medium dark:text-gray-200">
						<span class="font-mono">{g.account_code}</span> {g.account_name}
						<span class="text-xs text-gray-400 ml-2">{g.lines} {$i18n.t('line(s)')} · DR {fmt(g.debit)} · CR {fmt(g.credit)}</span>
					</div>
					<div class="flex items-center gap-2">
						<select class={selectCls} bind:value={perAccount[g.account_code]}>
							<option value="">{$i18n.t('— sub-account for all lines —')}</option>
							{#each g.sub_accounts ?? [] as sa}
								<option value={sa.code}>{sa.code} - {sa.name}</option>
							{/each}
						</select>
						<button
							class="px-2.5 py-1 text-xs font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition disabled:opacity-50"
							disabled={!perAccount[g.account_code]}
							on:click={() => applyToAccount(g.account_code)}
						>
							{$i18n.t('Apply to all')}
						</button>
					</div>
				</div>
				<div class="overflow-x-auto">
					<table class="w-full text-xs text-left">
						<thead class="text-[10px] uppercase bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
							<tr>
								<th class="px-3 py-1.5">{$i18n.t('Entry')}</th>
								<th class="px-3 py-1.5">{$i18n.t('Date')}</th>
								<th class="px-3 py-1.5">{$i18n.t('Status')}</th>
								<th class="px-3 py-1.5">{$i18n.t('Description')}</th>
								<th class="px-3 py-1.5 text-right">{$i18n.t('Debit')}</th>
								<th class="px-3 py-1.5 text-right">{$i18n.t('Credit')}</th>
								<th class="px-3 py-1.5">{$i18n.t('Move to')}</th>
							</tr>
						</thead>
						<tbody class="dark:text-gray-200">
							{#each postingsByAccount(g.account_code) as p (p.line_id)}
								<tr class="border-t border-gray-100 dark:border-gray-800 {p.status === 'voided' ? 'opacity-70' : ''}">
									<td class="px-3 py-1.5 font-mono">{p.entry_number || '#' + p.line_id}</td>
									<td class="px-3 py-1.5 whitespace-nowrap">{p.date}</td>
									<td class="px-3 py-1.5">{p.status}</td>
									<td class="px-3 py-1.5 max-w-xs truncate" title={p.description}>{p.counterparty ? p.counterparty + ' · ' : ''}{p.description}</td>
									<td class="px-3 py-1.5 text-right font-mono">{fmt(p.debit)}</td>
									<td class="px-3 py-1.5 text-right font-mono">{fmt(p.credit)}</td>
									<td class="px-3 py-1.5">
										<select class={selectCls} bind:value={targets[p.line_id]} on:change={() => (preview = null)}>
											<option value="">{$i18n.t('— leave —')}</option>
											{#each g.sub_accounts ?? [] as sa}
												<option value={sa.code}>{sa.code} - {sa.name}</option>
											{/each}
										</select>
									</td>
								</tr>
							{/each}
						</tbody>
					</table>
				</div>
			</div>
		{/each}

		{#if preview}
			<div class="rounded-xl border {preview.errors?.length ? 'border-red-200 dark:border-red-800/40' : 'border-emerald-200 dark:border-emerald-800/40'} p-3 text-xs space-y-1">
				<div class="font-medium dark:text-gray-200">{$i18n.t('Preview')}: {preview.changes?.length ?? 0} {$i18n.t('line(s) will move')}</div>
				{#each preview.errors ?? [] as e}<div class="text-red-700 dark:text-red-300">{e}</div>{/each}
			</div>
		{/if}

		<div class="flex items-center gap-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition disabled:opacity-50"
				disabled={working || chosen === 0}
				on:click={() => run(true)}
			>
				{$i18n.t('Preview')}
			</button>
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
				disabled={working || chosen === 0 || !preview || (preview.errors?.length ?? 0) > 0}
				on:click={() => run(false)}
			>
				{working ? $i18n.t('Working...') : $i18n.t('Apply {{n}} move(s)', { n: chosen })}
			</button>
			<span class="text-xs text-gray-400">{$i18n.t('Preview first; apply is all-or-nothing and audited.')}</span>
		</div>
	{/if}
</div>
