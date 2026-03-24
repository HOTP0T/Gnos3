<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getBalanceSheet, getPeriods, getCompany } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let loading = false;
	let data: any = null;
	let isFrench = false;

	// Month picker — user picks a month, we derive as_of and period_start from fiscal year
	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; asOf: string; fiscalStart: string }> = [];

	function buildMonthOptions(periods: any[]) {
		const options: typeof monthOptions = [];
		for (const p of periods) {
			const start = new Date(p.start_date);
			const end = new Date(p.end_date);
			const fiscalStart = p.start_date;
			let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
			while (cursor <= end) {
				const y = cursor.getFullYear();
				const m = cursor.getMonth();
				const lastDay = new Date(y, m + 1, 0).getDate();
				const endDate = `${y}-${String(m + 1).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
				const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				options.push({ value: `${y}-${String(m + 1).padStart(2, '0')}`, label, asOf: endDate, fiscalStart });
				cursor = new Date(y, m + 1, 1);
			}
		}
		const seen = new Map<string, typeof options[0]>();
		for (const o of options) seen.set(o.value, o);
		return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
	}

	onMount(async () => {
		try {
			const [res, company] = await Promise.all([
				getPeriods({ company_id: companyId }),
				getCompany(companyId)
			]);
			const periods = res.periods ?? res ?? [];
			monthOptions = buildMonthOptions(periods);

			const country = (company?.country ?? '').trim().toLowerCase();
			isFrench = country === 'france' || country === 'fr';

			const now = new Date();
			const curKey = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
			const match = monthOptions.find(o => o.value === curKey);
			if (match) {
				selectedMonth = match.value;
			} else if (monthOptions.length > 0) {
				selectedMonth = monthOptions[0].value;
			}
		} catch (err) {
			console.error('Failed to load periods:', err);
		}
	});

	const load = async () => {
		const opt = monthOptions.find(o => o.value === selectedMonth);
		if (!opt) { toast.error($i18n.t('Please select a period')); return; }
		loading = true;
		try {
			data = await getBalanceSheet({
				company_id: companyId,
				as_of: opt.asOf,
				period_start: opt.fiscalStart
			});
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '—';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const isCurrentAsset = (code: string): boolean => {
		const root = code.replace(/\./g, '');
		const prefix = parseInt(root.substring(0, 2));
		return prefix < 20 || (prefix >= 31 && prefix <= 39);
	};

	const isCurrentLiability = (code: string): boolean => {
		const root = code.replace(/\./g, '');
		const prefix = parseInt(root.substring(0, 2));
		return prefix >= 40 && prefix <= 49;
	};

	$: assetGroups = data ? groupAssets(data.assets) : { current: [], nonCurrent: [] };
	$: liabGroups = data ? groupLiabilities(data.liabilities) : { current: [], nonCurrent: [] };
	$: selectedLabel = monthOptions.find(o => o.value === selectedMonth)?.label ?? '';

	function groupAssets(items: any[]) {
		const current = items.filter((i: any) => isCurrentAsset(i.account_code));
		const nonCurrent = items.filter((i: any) => !isCurrentAsset(i.account_code));
		return { current, nonCurrent };
	}

	function groupLiabilities(items: any[]) {
		const current = items.filter((i: any) => isCurrentLiability(i.account_code));
		const nonCurrent = items.filter((i: any) => !isCurrentLiability(i.account_code));
		return { current, nonCurrent };
	}

	const sumBeg = (items: any[]): number => items.reduce((s: number, i: any) => s + (parseFloat(i.beginning_balance) || 0), 0);
	const sumEnd = (items: any[]): number => items.reduce((s: number, i: any) => s + (parseFloat(i.balance) || 0), 0);

	// === French Bilan grouping ===

	function frPcgPrefix(code: string): number {
		const c = code.replace(/\./g, '');
		return parseInt(c.substring(0, 2)) || 0;
	}

	function frPcgPrefix3(code: string): number {
		const c = code.replace(/\./g, '');
		return parseInt(c.substring(0, 3)) || 0;
	}

	// Amortization/depreciation estimation (accounts 28x, 29x, 39x offset their parent asset)
	function isAmortAccount(code: string): boolean {
		const c = code.replace(/\./g, '');
		return c.startsWith('28') || c.startsWith('29') || c.startsWith('39');
	}

	const sumAmort = (items: any[]): number => {
		return items
			.filter((i: any) => isAmortAccount(i.account_code))
			.reduce((s: number, i: any) => s + Math.abs(parseFloat(i.balance) || 0), 0);
	};

	// French number formatting: space as thousands separator, blank for zero
	const fmtFr = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '';
		return n.toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	// Collapsible sections
	let collapsedSections = new Set<string>();

	function toggleSection(key: string) {
		if (collapsedSections.has(key)) {
			collapsedSections.delete(key);
		} else {
			collapsedSections.add(key);
		}
		collapsedSections = collapsedSections; // trigger reactivity
	}

	// --- French Bilan Actif grouping ---
	interface FrenchActifGroups {
		immoIncorporelles: any[];
		immoCorporelles: any[];
		immoFinancieres: any[];
		stocks: any[];
		creances: any[];
		valeursMob: any[];
		disponibilites: any[];
		chargesConstatees: any[];
		otherAssets: any[];
	}

	function buildFrenchActifGroups(items: any[]): FrenchActifGroups {
		const groups: FrenchActifGroups = {
			immoIncorporelles: [], immoCorporelles: [], immoFinancieres: [],
			stocks: [], creances: [], valeursMob: [], disponibilites: [],
			chargesConstatees: [], otherAssets: []
		};
		for (const item of items) {
			const p2 = frPcgPrefix(item.account_code);
			const p3 = frPcgPrefix3(item.account_code);
			if (p2 === 20 || (p2 === 21 && p3 <= 212)) {
				groups.immoIncorporelles.push(item);
			} else if ((p2 === 21 && p3 > 212) || p2 === 22 || p2 === 23) {
				groups.immoCorporelles.push(item);
			} else if (p2 === 26 || p2 === 27) {
				groups.immoFinancieres.push(item);
			} else if (p2 === 28 || p2 === 29) {
				// amortization accounts — attribute to fixed asset groups
				// For simplicity put them in corporelles (they affect Amort column)
				groups.immoCorporelles.push(item);
			} else if (p2 >= 31 && p2 <= 39) {
				groups.stocks.push(item);
			} else if (p2 >= 40 && p2 <= 49) {
				groups.creances.push(item);
			} else if (p2 === 50) {
				groups.valeursMob.push(item);
			} else if (p2 >= 51 && p2 <= 53) {
				groups.disponibilites.push(item);
			} else if (p2 === 48) {
				groups.chargesConstatees.push(item);
			} else {
				groups.otherAssets.push(item);
			}
		}
		return groups;
	}

	// --- French Bilan Passif grouping ---
	interface FrenchPassifGroups {
		capitauxPropres: any[];
		autresFondsPropres: any[];
		provisions: any[];
		empruntsEtDettes: any[];
		otherPassif: any[];
	}

	function buildFrenchPassifGroups(liabilities: any[], equity: any[]): FrenchPassifGroups {
		const groups: FrenchPassifGroups = {
			capitauxPropres: [], autresFondsPropres: [], provisions: [],
			empruntsEtDettes: [], otherPassif: []
		};
		// Equity items
		for (const item of equity) {
			const p2 = frPcgPrefix(item.account_code);
			if (p2 >= 10 && p2 <= 14) {
				groups.capitauxPropres.push(item);
			} else if (p2 === 15) {
				groups.provisions.push(item);
			} else {
				groups.autresFondsPropres.push(item);
			}
		}
		// Liability items
		for (const item of liabilities) {
			const p2 = frPcgPrefix(item.account_code);
			if (p2 === 15) {
				groups.provisions.push(item);
			} else if ((p2 >= 16 && p2 <= 17) || (p2 >= 40 && p2 <= 51)) {
				groups.empruntsEtDettes.push(item);
			} else {
				groups.otherPassif.push(item);
			}
		}
		return groups;
	}

	$: frActif = (data && isFrench) ? buildFrenchActifGroups(data.assets) : null;
	$: frPassif = (data && isFrench) ? buildFrenchPassifGroups(data.liabilities, data.equity) : null;

	// Helper: all fixed asset items
	$: frFixedItems = frActif ? [...frActif.immoIncorporelles, ...frActif.immoCorporelles, ...frActif.immoFinancieres] : [];
	$: frCurrentItems = frActif ? [...frActif.stocks, ...frActif.creances, ...frActif.valeursMob, ...frActif.disponibilites, ...frActif.chargesConstatees, ...frActif.otherAssets] : [];

	let lineNo = 0;
	function nextLine(): number { return ++lineNo; }
	function resetLines() { lineNo = 0; }
</script>

<div class="space-y-3">
	<div class="flex flex-wrap gap-3 items-end">
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('As of end of')}</label>
			{#if monthOptions.length > 0}
				<select
					bind:value={selectedMonth}
					class="text-sm rounded-lg px-3 py-1.5 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden"
				>
					{#each monthOptions as opt}
						<option value={opt.value}>{opt.label}</option>
					{/each}
				</select>
			{:else}
				<span class="text-xs text-gray-400 italic">{$i18n.t('No accounting periods defined')}</span>
			{/if}
		</div>
		<button
			class="px-4 py-1.5 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition disabled:opacity-50"
			disabled={!selectedMonth}
			on:click={load}
		>{$i18n.t('Generate')}</button>
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else if data}
		{@const _ = resetLines()}

		{#if isFrench && frActif && frPassif}
			<!-- ========== FRENCH FORMAT: Bilan Actif + Bilan Passif (HACANTHE reference) ========== -->
			{@const fixedGross = sumEnd(frFixedItems.filter(i => !isAmortAccount(i.account_code)))}
			{@const fixedAmort = sumAmort(frFixedItems)}
			{@const fixedNet = fixedGross - fixedAmort}
			{@const currentGross = sumEnd(frCurrentItems.filter(i => !isAmortAccount(i.account_code)))}
			{@const currentAmort = sumAmort(frCurrentItems)}
			{@const currentNet = currentGross - currentAmort}
			{@const totalPassif = parseFloat(data.total_liabilities) + parseFloat(data.total_equity)}

			<!-- ===== BILAN ACTIF ===== -->
			<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 overflow-hidden mb-4">
				<div class="px-3 py-2 text-center font-bold text-sm dark:text-gray-200 bg-gray-50 dark:bg-gray-850/50 border-b border-gray-200 dark:border-gray-700">
					{$i18n.t('Bilan Actif')}
					<span class="text-[10px] font-normal text-gray-500 ml-2">({$i18n.t('As of')} {data.as_of})</span>
				</div>
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-2 py-1.5">{$i18n.t('Bilan Actif')}</th>
							<th class="px-2 py-1.5 text-right w-28">{$i18n.t('Brut')}</th>
							<th class="px-2 py-1.5 text-right w-28">{$i18n.t('Amort. Prov.')}</th>
							<th class="px-2 py-1.5 text-right w-28">{$i18n.t('Net')}</th>
							<th class="px-2 py-1.5 text-right w-28">{$i18n.t('Net (N-1)')}</th>
						</tr>
					</thead>
					<tbody>
						<!-- ── Actif immobilise ── -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('actif-immobilise')}
						>
							<td class="px-2 py-1.5" colspan="5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('actif-immobilise') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Actif immobilise')}
								{#if collapsedSections.has('actif-immobilise')}
									<span class="float-right font-mono">{fmtFr(fixedNet)}</span>
								{/if}
							</td>
						</tr>
						{#if !collapsedSections.has('actif-immobilise')}
							<!-- Immobilisations incorporelles -->
							{#if frActif.immoIncorporelles.length > 0}
								<tr class="bg-gray-50/50 dark:bg-gray-850/20">
									<td class="px-2 py-1 font-bold text-gray-700 dark:text-gray-300" colspan="5" style="padding-left: 16px">{$i18n.t('Immobilisations incorporelles')}</td>
								</tr>
								{#each frActif.immoIncorporelles as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 28px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? 0 : item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? Math.abs(parseFloat(item.balance) || 0) : 0)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Immobilisations corporelles -->
							{#if frActif.immoCorporelles.length > 0}
								<tr class="bg-gray-50/50 dark:bg-gray-850/20">
									<td class="px-2 py-1 font-bold text-gray-700 dark:text-gray-300" colspan="5" style="padding-left: 16px">{$i18n.t('Immobilisations corporelles')}</td>
								</tr>
								{#each frActif.immoCorporelles as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 28px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? 0 : item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? Math.abs(parseFloat(item.balance) || 0) : 0)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? 0 : (parseFloat(item.balance) || 0))}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Immobilisations financieres -->
							{#if frActif.immoFinancieres.length > 0}
								<tr class="bg-gray-50/50 dark:bg-gray-850/20">
									<td class="px-2 py-1 font-bold text-gray-700 dark:text-gray-300" colspan="5" style="padding-left: 16px">{$i18n.t('Immobilisations financieres')}</td>
								</tr>
								{#each frActif.immoFinancieres as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 28px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? 0 : item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(isAmortAccount(item.account_code) ? Math.abs(parseFloat(item.balance) || 0) : 0)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Subtotal: ACTIF IMMOBILISE -->
							<tr class="border-t border-gray-300 dark:border-gray-600">
								<td class="px-2 py-1.5 text-right text-green-700 dark:text-green-400 font-bold">{$i18n.t('ACTIF IMMOBILISE')}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(fixedGross)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(fixedAmort)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(fixedNet)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(sumBeg(frFixedItems))}</td>
							</tr>
						{/if}

						<!-- ── Actif circulant ── -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('actif-circulant')}
						>
							<td class="px-2 py-1.5" colspan="5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('actif-circulant') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Actif circulant')}
								{#if collapsedSections.has('actif-circulant')}
									<span class="float-right font-mono">{fmtFr(currentNet)}</span>
								{/if}
							</td>
						</tr>
						{#if !collapsedSections.has('actif-circulant')}
							<!-- Stocks et en-cours -->
							{#if frActif.stocks.length > 0}
								<tr class="bg-gray-50/50 dark:bg-gray-850/20">
									<td class="px-2 py-1 font-bold text-gray-700 dark:text-gray-300" colspan="5" style="padding-left: 16px">{$i18n.t('Stocks et en-cours')}</td>
								</tr>
								{#each frActif.stocks as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 28px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono"></td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Creances -->
							{#if frActif.creances.length > 0}
								<tr class="bg-gray-50/50 dark:bg-gray-850/20">
									<td class="px-2 py-1 font-bold text-gray-700 dark:text-gray-300" colspan="5" style="padding-left: 16px">{$i18n.t('Creances')}</td>
								</tr>
								{#each frActif.creances as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 28px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono"></td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Valeurs mobilieres de placement -->
							{#if frActif.valeursMob.length > 0}
								<tr class="bg-gray-50/50 dark:bg-gray-850/20">
									<td class="px-2 py-1 font-bold text-gray-700 dark:text-gray-300" colspan="5" style="padding-left: 16px">{$i18n.t('Valeurs mobilieres de placement')}</td>
								</tr>
								{#each frActif.valeursMob as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 28px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono"></td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Disponibilites -->
							{#if frActif.disponibilites.length > 0}
								{#each frActif.disponibilites as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono"></td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Charges constatees d'avance -->
							{#if frActif.chargesConstatees.length > 0}
								{#each frActif.chargesConstatees as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono"></td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Other assets -->
							{#if frActif.otherAssets.length > 0}
								{#each frActif.otherAssets as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
										<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono"></td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
									</tr>
								{/each}
							{/if}

							<!-- Subtotal: ACTIF CIRCULANT -->
							<tr class="border-t border-gray-300 dark:border-gray-600">
								<td class="px-2 py-1.5 text-right text-green-700 dark:text-green-400 font-bold">{$i18n.t('ACTIF CIRCULANT')}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(currentGross)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(currentAmort)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(currentNet)}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(sumBeg(frCurrentItems))}</td>
							</tr>
						{/if}
					</tbody>
					<tfoot>
						<tr class="bg-green-600/15 text-green-700 dark:text-green-400 font-bold border-t-2 border-green-300 dark:border-green-700">
							<td class="px-2 py-2">{$i18n.t('TOTAL ACTIF')}</td>
							<td class="px-2 py-2 text-right font-mono">{fmtFr(fixedGross + currentGross)}</td>
							<td class="px-2 py-2 text-right font-mono">{fmtFr(fixedAmort + currentAmort)}</td>
							<td class="px-2 py-2 text-right font-mono">{fmtFr(data.total_assets)}</td>
							<td class="px-2 py-2 text-right font-mono">{fmtFr(sumBeg(data.assets))}</td>
						</tr>
					</tfoot>
				</table>
			</div>

			<!-- ===== BILAN PASSIF ===== -->
			<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 overflow-hidden">
				<div class="px-3 py-2 text-center font-bold text-sm dark:text-gray-200 bg-gray-50 dark:bg-gray-850/50 border-b border-gray-200 dark:border-gray-700">
					{$i18n.t('Bilan Passif')}
					<span class="text-xs font-medium ml-2 px-2 py-0.5 rounded-lg {data.is_balanced ? 'bg-green-500/20 text-green-700 dark:text-green-300' : 'bg-red-500/20 text-red-700 dark:text-red-300'}">{data.is_balanced ? $i18n.t('Balanced') : $i18n.t('Unbalanced')}</span>
				</div>
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-2 py-1.5">{$i18n.t('Bilan Passif')}</th>
							<th class="px-2 py-1.5 text-right w-32">{$i18n.t('Exercice N')}</th>
							<th class="px-2 py-1.5 text-right w-32">{$i18n.t('Exercice N-1')}</th>
						</tr>
					</thead>
					<tbody>
						<!-- ── Capitaux propres ── -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('capitaux-propres')}
						>
							<td class="px-2 py-1.5" colspan="3">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('capitaux-propres') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Capitaux propres')}
								{#if collapsedSections.has('capitaux-propres')}
									<span class="float-right font-mono">{fmtFr(sumEnd(frPassif.capitauxPropres))}</span>
								{/if}
							</td>
						</tr>
						{#if !collapsedSections.has('capitaux-propres')}
							{#each frPassif.capitauxPropres as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
									<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
								</tr>
							{/each}
							<!-- Subtotal -->
							<tr class="border-t border-gray-300 dark:border-gray-600">
								<td class="px-2 py-1.5 text-right text-green-700 dark:text-green-400 font-bold">{$i18n.t('CAPITAUX PROPRES')}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(sumEnd(frPassif.capitauxPropres))}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(sumBeg(frPassif.capitauxPropres))}</td>
							</tr>
						{/if}

						<!-- ── Autres fonds propres ── -->
						{#if frPassif.autresFondsPropres.length > 0}
							<tr class="bg-green-600 text-white text-xs font-semibold">
								<td class="px-2 py-1.5" colspan="3">{$i18n.t('Autres fonds propres')}</td>
							</tr>
							{#each frPassif.autresFondsPropres as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
									<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
								</tr>
							{/each}
						{/if}

						<!-- ── Provisions pour risques et charges ── -->
						{#if frPassif.provisions.length > 0}
							<tr class="bg-green-600 text-white text-xs font-semibold">
								<td class="px-2 py-1.5" colspan="3">{$i18n.t('Provisions pour risques et charges')}</td>
							</tr>
							{#each frPassif.provisions as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
									<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
								</tr>
							{/each}
						{/if}

						<!-- ── Emprunts et dettes ── -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('emprunts-dettes')}
						>
							<td class="px-2 py-1.5" colspan="3">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('emprunts-dettes') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Emprunts et dettes')}
								{#if collapsedSections.has('emprunts-dettes')}
									<span class="float-right font-mono">{fmtFr(sumEnd(frPassif.empruntsEtDettes))}</span>
								{/if}
							</td>
						</tr>
						{#if !collapsedSections.has('emprunts-dettes')}
							{#each frPassif.empruntsEtDettes as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
									<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
								</tr>
							{/each}
							<!-- Subtotal -->
							<tr class="border-t border-gray-300 dark:border-gray-600">
								<td class="px-2 py-1.5 text-right text-green-700 dark:text-green-400 font-bold">{$i18n.t('EMPRUNTS ET DETTES')}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(sumEnd(frPassif.empruntsEtDettes))}</td>
								<td class="px-2 py-1.5 text-right font-mono font-bold">{fmtFr(sumBeg(frPassif.empruntsEtDettes))}</td>
							</tr>
						{/if}

						<!-- Other passif -->
						{#if frPassif.otherPassif.length > 0}
							{#each frPassif.otherPassif as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50 dark:hover:bg-gray-850/30">
									<td class="px-2 py-1" style="padding-left: 16px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.balance)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.beginning_balance)}</td>
								</tr>
							{/each}
						{/if}
					</tbody>
					<tfoot>
						<tr class="bg-green-600/15 text-green-700 dark:text-green-400 font-bold border-t-2 border-green-300 dark:border-green-700">
							<td class="px-2 py-2">{$i18n.t('TOTAL PASSIF')}</td>
							<td class="px-2 py-2 text-right font-mono">{fmtFr(totalPassif)}</td>
							<td class="px-2 py-2 text-right font-mono">{fmtFr(sumBeg(data.liabilities) + sumBeg(data.equity))}</td>
						</tr>
					</tfoot>
				</table>
			</div>
		{:else}
			<!-- ========== DEFAULT FORMAT (US/other) ========== -->
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
				<!-- LEFT: Assets -->
				<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 overflow-hidden">
					<div class="px-3 py-2 bg-blue-50/50 dark:bg-blue-900/20 font-semibold text-sm dark:text-gray-200 flex justify-between items-center">
						<span>{$i18n.t('Assets')}</span>
						<span class="text-[10px] font-normal text-gray-500 uppercase">{$i18n.t('As of')} {data.as_of}</span>
					</div>
					<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
						<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
							<tr>
								<th class="px-2 py-1.5"></th>
								<th class="px-2 py-1.5 w-8">#</th>
								<th class="px-2 py-1.5 text-right">{$i18n.t('Beg. Balance')}</th>
								<th class="px-2 py-1.5 text-right">{$i18n.t('End. Balance')}</th>
							</tr>
						</thead>
						<tbody>
							{#if assetGroups.current.length > 0}
								<tr class="bg-blue-50/20 dark:bg-blue-900/5">
									<td class="px-2 py-1 font-medium text-gray-600 dark:text-gray-400 italic" colspan="4">{$i18n.t('Current assets:')}</td>
								</tr>
								{#each assetGroups.current as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30">
										<td class="px-2 py-1" style="padding-left: {8 + item.level * 10}px">{item.account_name}</td>
										<td class="px-2 py-1 text-gray-400 text-center">{nextLine()}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.beginning_balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.balance)}</td>
									</tr>
								{/each}
								<tr class="font-medium border-t border-gray-200 dark:border-gray-700 text-[10px]">
									<td class="px-2 py-1 italic" colspan="2">{$i18n.t('Sub-total current assets')}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumBeg(assetGroups.current))}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumEnd(assetGroups.current))}</td>
								</tr>
							{/if}

							{#if assetGroups.nonCurrent.length > 0}
								<tr class="bg-blue-50/20 dark:bg-blue-900/5">
									<td class="px-2 py-1 font-medium text-gray-600 dark:text-gray-400 italic" colspan="4">{$i18n.t('Non-current assets:')}</td>
								</tr>
								{#each assetGroups.nonCurrent as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30">
										<td class="px-2 py-1" style="padding-left: {8 + item.level * 10}px">{item.account_name}</td>
										<td class="px-2 py-1 text-gray-400 text-center">{nextLine()}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.beginning_balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.balance)}</td>
									</tr>
								{/each}
								<tr class="font-medium border-t border-gray-200 dark:border-gray-700 text-[10px]">
									<td class="px-2 py-1 italic" colspan="2">{$i18n.t('Sub-total non-current assets')}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumBeg(assetGroups.nonCurrent))}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumEnd(assetGroups.nonCurrent))}</td>
								</tr>
							{/if}
						</tbody>
						<tfoot class="font-bold bg-blue-50/30 dark:bg-blue-900/10">
							<tr class="border-t-2 border-blue-200 dark:border-blue-800">
								<td class="px-2 py-2" colspan="2">{$i18n.t('Total Assets')}</td>
								<td class="px-2 py-2 text-right font-mono">{fmt(sumBeg(data.assets))}</td>
								<td class="px-2 py-2 text-right font-mono">{fmt(data.total_assets)}</td>
							</tr>
						</tfoot>
					</table>
				</div>

				<!-- RIGHT: Liabilities & Equity -->
				<div class="bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 overflow-hidden">
					<div class="px-3 py-2 bg-orange-50/50 dark:bg-orange-900/20 font-semibold text-sm dark:text-gray-200 flex justify-between items-center">
						<span>{$i18n.t("Liabilities & Owners' Equity")}</span>
						<span class="text-xs font-medium px-2 py-0.5 rounded-lg {data.is_balanced ? 'bg-green-500/20 text-green-700 dark:text-green-300' : 'bg-red-500/20 text-red-700 dark:text-red-300'}">{data.is_balanced ? $i18n.t('Balanced') : $i18n.t('Unbalanced')}</span>
					</div>
					<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
						<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
							<tr>
								<th class="px-2 py-1.5"></th>
								<th class="px-2 py-1.5 w-8">#</th>
								<th class="px-2 py-1.5 text-right">{$i18n.t('Beg. Balance')}</th>
								<th class="px-2 py-1.5 text-right">{$i18n.t('End. Balance')}</th>
							</tr>
						</thead>
						<tbody>
							{#if liabGroups.current.length > 0}
								<tr class="bg-orange-50/20 dark:bg-orange-900/5">
									<td class="px-2 py-1 font-medium text-gray-600 dark:text-gray-400 italic" colspan="4">{$i18n.t('Current liabilities:')}</td>
								</tr>
								{#each liabGroups.current as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30">
										<td class="px-2 py-1" style="padding-left: {8 + item.level * 10}px">{item.account_name}</td>
										<td class="px-2 py-1 text-gray-400 text-center">{nextLine()}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.beginning_balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.balance)}</td>
									</tr>
								{/each}
								<tr class="font-medium border-t border-gray-200 dark:border-gray-700 text-[10px]">
									<td class="px-2 py-1 italic" colspan="2">{$i18n.t('Sub-total current liabilities')}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumBeg(liabGroups.current))}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumEnd(liabGroups.current))}</td>
								</tr>
							{/if}

							{#if liabGroups.nonCurrent.length > 0}
								<tr class="bg-orange-50/20 dark:bg-orange-900/5">
									<td class="px-2 py-1 font-medium text-gray-600 dark:text-gray-400 italic" colspan="4">{$i18n.t('Non-current liabilities:')}</td>
								</tr>
								{#each liabGroups.nonCurrent as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30">
										<td class="px-2 py-1" style="padding-left: {8 + item.level * 10}px">{item.account_name}</td>
										<td class="px-2 py-1 text-gray-400 text-center">{nextLine()}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.beginning_balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.balance)}</td>
									</tr>
								{/each}
								<tr class="font-medium border-t border-gray-200 dark:border-gray-700 text-[10px]">
									<td class="px-2 py-1 italic" colspan="2">{$i18n.t('Sub-total non-current liabilities')}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumBeg(liabGroups.nonCurrent))}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(sumEnd(liabGroups.nonCurrent))}</td>
								</tr>
							{/if}

							<tr class="font-semibold border-t border-gray-300 dark:border-gray-600">
								<td class="px-2 py-1.5" colspan="2">{$i18n.t('Total Liabilities')}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(sumBeg(data.liabilities))}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(data.total_liabilities)}</td>
							</tr>

							{#if data.equity.length > 0}
								<tr class="bg-purple-50/20 dark:bg-purple-900/5">
									<td class="px-2 py-1 font-medium text-gray-600 dark:text-gray-400 italic" colspan="4">{$i18n.t("Owner's equity:")}</td>
								</tr>
								{#each data.equity as item}
									<tr class="border-b border-gray-50 dark:border-gray-850/30">
										<td class="px-2 py-1" style="padding-left: {8 + item.level * 10}px">{item.account_name}</td>
										<td class="px-2 py-1 text-gray-400 text-center">{nextLine()}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.beginning_balance)}</td>
										<td class="px-2 py-1 text-right font-mono">{fmt(item.balance)}</td>
									</tr>
								{/each}
								<tr class="font-semibold border-t border-gray-300 dark:border-gray-600">
									<td class="px-2 py-1.5" colspan="2">{$i18n.t("Total Owner's Equity")}</td>
									<td class="px-2 py-1.5 text-right font-mono">{fmt(sumBeg(data.equity))}</td>
									<td class="px-2 py-1.5 text-right font-mono">{fmt(data.total_equity)}</td>
								</tr>
							{/if}
						</tbody>
						<tfoot class="font-bold bg-orange-50/30 dark:bg-orange-900/10">
							<tr class="border-t-2 border-orange-200 dark:border-orange-800">
								<td class="px-2 py-2" colspan="2">{$i18n.t("Total Liabilities & Equity")}</td>
								<td class="px-2 py-2 text-right font-mono">{fmt(sumBeg(data.liabilities) + sumBeg(data.equity))}</td>
								<td class="px-2 py-2 text-right font-mono">{fmt(parseFloat(data.total_liabilities) + parseFloat(data.total_equity))}</td>
							</tr>
						</tfoot>
					</table>
				</div>
			</div>
		{/if}
	{/if}
</div>
