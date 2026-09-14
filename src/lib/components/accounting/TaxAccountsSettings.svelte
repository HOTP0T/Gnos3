<script lang="ts">
	import { onMount, getContext, createEventDispatcher } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getTaxAccounts, updateTaxAccounts, getAccounts } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let companyId: number;

	type Acct = { id: number; code: string; name: string; account_type?: string; parent_id?: number | null };
	type Role = {
		role: string;
		label: string;
		group: string;
		multi: boolean;
		account_type: string | null;
		source: 'mapped' | 'default' | 'missing';
		accounts: Acct[];
	};

	let loading = true;
	let saving = false;
	let country = '';
	let roles: Role[] = [];
	let accounts: Acct[] = [];
	// Working copy: role → account ids. Only roles the user touched are sent.
	let draft: Record<string, number[]> = {};
	let dirty: Set<string> = new Set();
	// Per-role "add another" picker value (multi roles)
	let adder: Record<string, number | ''> = {};

	const GROUPS: Array<{ id: string; label: string; hint: string }> = [
		{ id: 'vat', label: 'VAT', hint: 'Accounts the VAT declaration reads (output / input) and the settlement entry posts to.' },
		{ id: 'surcharges', label: 'VAT surcharges', hint: 'Only used when the country config levies surcharges on VAT payable (e.g. China 城建税 / 教育费附加).' },
		{ id: 'cit', label: 'Corporate income tax', hint: 'Accrual entry (expense / payable) and the account the CIT payment debits.' },
		{ id: 'iit', label: 'Individual income tax', hint: 'Account the IIT payment debits.' }
	];

	$: parentIds = new Set(accounts.map((a) => a.parent_id).filter(Boolean));
	$: leafAccounts = accounts.filter((a) => !parentIds.has(a.id));
	$: byId = new Map(accounts.map((a) => [a.id, a]));

	const candidates = (role: Role): Acct[] => {
		// Prefer the role's account type; fall back to every leaf so an unusual chart still works.
		const typed = role.account_type ? leafAccounts.filter((a) => a.account_type === role.account_type) : [];
		return typed.length ? typed : leafAccounts;
	};

	const load = async () => {
		loading = true;
		try {
			const [map, accts] = await Promise.all([getTaxAccounts(companyId), getAccounts({ company_id: companyId, active: true })]);
			country = map.country ?? '';
			roles = map.roles ?? [];
			accounts = Array.isArray(accts) ? accts : (accts?.items ?? accts?.accounts ?? []);
			draft = Object.fromEntries(roles.map((r) => [r.role, r.accounts.map((a) => a.id)]));
			dirty = new Set();
			adder = {};
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to load tax accounts')}: ${err?.detail ?? err}`);
		}
		loading = false;
	};

	onMount(load);

	const touch = (role: string) => {
		dirty = new Set(dirty).add(role);
		draft = { ...draft };
	};

	const setSingle = (role: string, value: number | '') => {
		draft[role] = value === '' ? [] : [Number(value)];
		touch(role);
	};

	const addMulti = (role: string) => {
		const v = adder[role];
		if (v === '' || v === undefined) return;
		const id = Number(v);
		if (!draft[role].includes(id)) draft[role] = [...draft[role], id];
		adder[role] = '';
		touch(role);
	};

	const removeMulti = (role: string, id: number) => {
		draft[role] = draft[role].filter((x) => x !== id);
		touch(role);
	};

	const resetRole = (role: string) => {
		draft[role] = [];
		touch(role);
	};

	const save = async () => {
		if (dirty.size === 0) return;
		saving = true;
		try {
			const mappings: Record<string, number[]> = {};
			for (const r of dirty) mappings[r] = draft[r] ?? [];
			await updateTaxAccounts(companyId, mappings);
			toast.success($i18n.t('Tax accounts saved'));
			await load();
			dispatch('saved');
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save tax accounts')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	const sourceBadge = (r: Role, d: Set<string>) => {
		if (d.has(r.role)) return 'bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-300';
		if (r.source === 'mapped') return 'bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-300';
		if (r.source === 'default') return 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-300';
		return 'bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-300';
	};
	const sourceText = (r: Role, d: Set<string>) => {
		if (d.has(r.role)) return $i18n.t('unsaved');
		if (r.source === 'mapped') return $i18n.t('mapped');
		if (r.source === 'default') return $i18n.t('default');
		return $i18n.t('missing');
	};

	$: missingCount = roles.filter((r) => r.source === 'missing').length;
	$: defaultCount = roles.filter((r) => r.source === 'default').length;

	// Defaults are deterministic but unconfirmed: automation only uses mapped
	// accounts. One click turns every current default into an explicit mapping.
	const confirmDefaults = async () => {
		const mappings: Record<string, number[]> = {};
		for (const r of roles) if (r.source === 'default') mappings[r.role] = r.accounts.map((a) => a.id);
		if (Object.keys(mappings).length === 0) return;
		saving = true;
		try {
			await updateTaxAccounts(companyId, mappings);
			toast.success($i18n.t('Defaults confirmed'));
			await load();
			dispatch('saved');
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save tax accounts')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	const selectCls =
		'text-xs rounded-lg px-2 py-1.5 bg-white dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden max-w-full';
</script>

<div class="py-2">
	<div class="flex md:self-center text-lg font-medium px-0.5 gap-2 mb-1">
		<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Tax accounts')}</div>
		{#if country}
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">{country}</div>
		{/if}
	</div>
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t(
			'Which GL accounts play each tax role for this company. "default" is resolved from the country config and the chart; map an account to override it. Only detail (leaf) accounts can be mapped.'
		)}
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else}
		{#if defaultCount > 0}
			<div
				class="bg-amber-50 dark:bg-amber-900/20 border border-amber-200/50 dark:border-amber-800/30 rounded-xl p-3 mb-3 text-xs text-amber-800 dark:text-amber-200 flex flex-wrap items-center justify-between gap-2"
			>
				<span>{$i18n.t('{{count}} role(s) are resolved from the country defaults but not confirmed. Automatic booking (rules, expense sheets) only uses confirmed accounts.', { count: defaultCount })}</span>
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-amber-600 text-white hover:bg-amber-700 transition disabled:opacity-50"
					disabled={saving}
					on:click={confirmDefaults}
				>
					{$i18n.t('Confirm all defaults')}
				</button>
			</div>
		{/if}
		{#if missingCount > 0}
			<div
				class="bg-red-50 dark:bg-red-900/20 border border-red-200/50 dark:border-red-800/30 rounded-xl p-3 mb-3 text-xs text-red-800 dark:text-red-200"
			>
				{$i18n.t('{{count}} role(s) have no account. Declarations still compute, but settlement entries and payments that need them are blocked until mapped.', { count: missingCount })}
			</div>
		{/if}

		{#each GROUPS as g}
			{@const groupRoles = roles.filter((r) => r.group === g.id)}
			{#if groupRoles.length}
				<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 mb-3">
					<div class="px-4 py-3 border-b border-gray-100 dark:border-gray-850">
						<div class="text-sm font-medium dark:text-gray-200">{$i18n.t(g.label)}</div>
						<div class="text-[11px] text-gray-400 dark:text-gray-500">{$i18n.t(g.hint)}</div>
					</div>
					<div class="divide-y divide-gray-100 dark:divide-gray-850">
						{#each groupRoles as r (r.role)}
							<div class="px-4 py-2.5 flex flex-col md:flex-row md:items-center gap-2">
								<div class="md:w-72 shrink-0">
									<div class="text-xs font-medium dark:text-gray-200">{$i18n.t(r.label)}</div>
									<span class="inline-block mt-0.5 px-1.5 py-0.5 rounded-full text-[10px] font-medium {sourceBadge(r, dirty)}">
										{sourceText(r, dirty)}
									</span>
								</div>
								<div class="flex-1 flex flex-wrap items-center gap-2 min-w-0">
									{#if r.multi}
										{#each draft[r.role] ?? [] as id (id)}
											{@const a = byId.get(id)}
											<span
												class="inline-flex items-center gap-1 px-2 py-1 rounded-lg bg-gray-100 dark:bg-gray-800 text-xs font-mono dark:text-gray-200"
											>
												{a ? `${a.code} ${a.name}` : `#${id}`}
												<button
													class="text-gray-400 hover:text-red-600 ml-1"
													title={$i18n.t('Remove')}
													on:click={() => removeMulti(r.role, id)}>×</button
												>
											</span>
										{/each}
										<select class={selectCls} bind:value={adder[r.role]} on:change={() => addMulti(r.role)}>
											<option value="">{$i18n.t('Add account...')}</option>
											{#each candidates(r) as a}
												{#if !(draft[r.role] ?? []).includes(a.id)}
													<option value={a.id}>{a.code} - {a.name}</option>
												{/if}
											{/each}
										</select>
									{:else}
										<select
											class={selectCls}
											value={(draft[r.role] ?? [])[0] ?? ''}
											on:change={(e) => setSingle(r.role, (e.currentTarget as HTMLSelectElement).value === '' ? '' : Number((e.currentTarget as HTMLSelectElement).value))}
										>
											<option value="">{$i18n.t('— not mapped —')}</option>
											{#each candidates(r) as a}
												<option value={a.id}>{a.code} - {a.name}</option>
											{/each}
										</select>
									{/if}
									{#if (draft[r.role] ?? []).length > 0 || r.source === 'mapped'}
										<button
											class="text-[11px] text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 underline"
											on:click={() => resetRole(r.role)}
										>
											{$i18n.t('Reset to default')}
										</button>
									{/if}
								</div>
							</div>
						{/each}
					</div>
				</div>
			{/if}
		{/each}

		<div class="flex gap-2 items-center">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
				disabled={saving || dirty.size === 0}
				on:click={save}
			>
				{saving ? $i18n.t('Saving...') : $i18n.t('Save')}
			</button>
			{#if dirty.size > 0}
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
					on:click={load}
				>
					{$i18n.t('Discard')}
				</button>
				<span class="text-xs text-gray-400">{$i18n.t('{{count}} unsaved change(s)', { count: dirty.size })}</span>
			{/if}
		</div>
	{/if}
</div>
