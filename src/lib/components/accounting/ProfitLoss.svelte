<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { getProfitLoss, getPeriods, getCompany } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	export let companyId: number;

	let loading = false;
	let data: any = null;
	let companyCountry: string = '';
	let isFrench = false;

	let selectedMonth = '';
	let monthOptions: Array<{ value: string; label: string; from: string; to: string; fiscalStart: string }> = [];

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
				const from = `${y}-${String(m + 1).padStart(2, '0')}-01`;
				const lastDay = new Date(y, m + 1, 0).getDate();
				const to = `${y}-${String(m + 1).padStart(2, '0')}-${String(lastDay).padStart(2, '0')}`;
				const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
				options.push({ value: `${y}-${String(m + 1).padStart(2, '0')}`, label, from, to, fiscalStart });
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
			companyCountry = company?.country ?? '';
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
			data = await getProfitLoss({
				company_id: companyId,
				date_from: opt.from,
				date_to: opt.to,
				ytd_start: opt.fiscalStart
			});
		} catch (err) { toast.error(`${err}`); }
		loading = false;
	};

	const fmt = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '—';
		return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
	};

	const isCostAccount = (code: string, name: string): boolean => {
		const c = code.replace(/\./g, '');
		const lname = name.toLowerCase();
		return c.startsWith('54') || c.startsWith('59') || lname.includes('cost') || lname.includes('coût') || lname.includes('cogs');
	};

	const isTaxSurcharge = (code: string, name: string): boolean => {
		const lname = name.toLowerCase();
		return lname.includes('tax') && (lname.includes('surcharge') || lname.includes('附加'));
	};

	const isFinancialExpense = (code: string, name: string): boolean => {
		const c = code.replace(/\./g, '');
		const lname = name.toLowerCase();
		return c.startsWith('56') || c.startsWith('66') || lname.includes('financial') || lname.includes('financ');
	};

	// === French PCG classification helpers ===

	function pcgPrefix(code: string): number {
		const c = code.replace(/\./g, '');
		return parseInt(c.substring(0, 2)) || 0;
	}

	function isFrenchOperatingRevenue(code: string): boolean {
		const p = pcgPrefix(code);
		return p >= 70 && p <= 75;
	}

	function isFrenchOperatingExpense(code: string): boolean {
		const p = pcgPrefix(code);
		return p >= 60 && p <= 65;
	}

	function isFrenchFinancialRevenue(code: string): boolean {
		return pcgPrefix(code) === 76;
	}

	function isFrenchFinancialExpense(code: string): boolean {
		return pcgPrefix(code) === 66;
	}

	function isFrenchExceptionalRevenue(code: string): boolean {
		return pcgPrefix(code) === 77;
	}

	function isFrenchExceptionalExpense(code: string): boolean {
		return pcgPrefix(code) === 67;
	}

	function isFrenchIncomeTax(code: string): boolean {
		return pcgPrefix(code) === 69;
	}

	function buildFrenchStructured(d: any) {
		const allRevenue = d.revenue ?? [];
		const allExpenses = d.expenses ?? [];

		const sumAmt = (items: any[]) => items.reduce((s: number, e: any) => s + (parseFloat(e.amount) || 0), 0);
		const sumYtd = (items: any[]) => items.reduce((s: number, e: any) => s + (parseFloat(e.ytd_amount) || 0), 0);

		// Operating
		const operatingRevenue = allRevenue.filter((e: any) => isFrenchOperatingRevenue(e.account_code));
		const operatingExpenses = allExpenses.filter((e: any) => isFrenchOperatingExpense(e.account_code));
		const totalOpRevenue = sumAmt(operatingRevenue);
		const totalOpRevenueYtd = sumYtd(operatingRevenue);
		const totalOpExpenses = sumAmt(operatingExpenses);
		const totalOpExpensesYtd = sumYtd(operatingExpenses);
		const resultatExploitation = totalOpRevenue - totalOpExpenses;
		const resultatExploitationYtd = totalOpRevenueYtd - totalOpExpensesYtd;

		// Financial
		const financialRevenue = allRevenue.filter((e: any) => isFrenchFinancialRevenue(e.account_code));
		const financialExpenses = allExpenses.filter((e: any) => isFrenchFinancialExpense(e.account_code));
		const totalFinRevenue = sumAmt(financialRevenue);
		const totalFinRevenueYtd = sumYtd(financialRevenue);
		const totalFinExpenses = sumAmt(financialExpenses);
		const totalFinExpensesYtd = sumYtd(financialExpenses);
		const resultatFinancier = totalFinRevenue - totalFinExpenses;
		const resultatFinancierYtd = totalFinRevenueYtd - totalFinExpensesYtd;

		// Exceptional
		const exceptionalRevenue = allRevenue.filter((e: any) => isFrenchExceptionalRevenue(e.account_code));
		const exceptionalExpenses = allExpenses.filter((e: any) => isFrenchExceptionalExpense(e.account_code));
		const totalExcRevenue = sumAmt(exceptionalRevenue);
		const totalExcRevenueYtd = sumYtd(exceptionalRevenue);
		const totalExcExpenses = sumAmt(exceptionalExpenses);
		const totalExcExpensesYtd = sumYtd(exceptionalExpenses);
		const resultatExceptionnel = totalExcRevenue - totalExcExpenses;
		const resultatExceptionnelYtd = totalExcRevenueYtd - totalExcExpensesYtd;

		// Income Tax
		const incomeTax = allExpenses.filter((e: any) => isFrenchIncomeTax(e.account_code));
		const totalIncomeTax = sumAmt(incomeTax);
		const totalIncomeTaxYtd = sumYtd(incomeTax);

		// Uncategorized (safety net)
		const categorizedRevCodes = new Set([...operatingRevenue, ...financialRevenue, ...exceptionalRevenue].map((e: any) => e.account_code));
		const categorizedExpCodes = new Set([...operatingExpenses, ...financialExpenses, ...exceptionalExpenses, ...incomeTax].map((e: any) => e.account_code));
		const uncategorizedRevenue = allRevenue.filter((e: any) => !categorizedRevCodes.has(e.account_code));
		const uncategorizedExpenses = allExpenses.filter((e: any) => !categorizedExpCodes.has(e.account_code));
		const totalUncatRevenue = sumAmt(uncategorizedRevenue);
		const totalUncatRevenueYtd = sumYtd(uncategorizedRevenue);
		const totalUncatExpenses = sumAmt(uncategorizedExpenses);
		const totalUncatExpensesYtd = sumYtd(uncategorizedExpenses);

		// Résultat courant avant impôts
		const resultatCourant = resultatExploitation + resultatFinancier;
		const resultatCourantYtd = resultatExploitationYtd + resultatFinancierYtd;

		// Total des produits / charges
		const totalProduits = totalOpRevenue + totalFinRevenue + totalExcRevenue + totalUncatRevenue;
		const totalProduitsYtd = totalOpRevenueYtd + totalFinRevenueYtd + totalExcRevenueYtd + totalUncatRevenueYtd;
		const totalCharges = totalOpExpenses + totalFinExpenses + totalExcExpenses + totalIncomeTax + totalUncatExpenses;
		const totalChargesYtd = totalOpExpensesYtd + totalFinExpensesYtd + totalExcExpensesYtd + totalIncomeTaxYtd + totalUncatExpensesYtd;

		// Net Income
		const netIncome = totalProduits - totalCharges;
		const netIncomeYtd = totalProduitsYtd - totalChargesYtd;

		return {
			operatingRevenue, operatingExpenses,
			totalOpRevenue, totalOpRevenueYtd,
			totalOpExpenses, totalOpExpensesYtd,
			resultatExploitation, resultatExploitationYtd,
			financialRevenue, financialExpenses,
			totalFinRevenue, totalFinRevenueYtd,
			totalFinExpenses, totalFinExpensesYtd,
			resultatFinancier, resultatFinancierYtd,
			exceptionalRevenue, exceptionalExpenses,
			totalExcRevenue, totalExcRevenueYtd,
			totalExcExpenses, totalExcExpensesYtd,
			resultatExceptionnel, resultatExceptionnelYtd,
			resultatCourant, resultatCourantYtd,
			incomeTax, totalIncomeTax, totalIncomeTaxYtd,
			uncategorizedRevenue, uncategorizedExpenses,
			totalUncatRevenue, totalUncatRevenueYtd,
			totalUncatExpenses, totalUncatExpensesYtd,
			totalProduits, totalProduitsYtd,
			totalCharges, totalChargesYtd,
			netIncome, netIncomeYtd
		};
	}

	$: structured = data ? (isFrench ? buildFrenchStructured(data) : buildStructured(data)) : null;

	function buildStructured(d: any) {
		const revenue = d.revenue ?? [];
		const expenses = d.expenses ?? [];

		const costs = expenses.filter((e: any) => isCostAccount(e.account_code, e.account_name));
		const taxes = expenses.filter((e: any) => isTaxSurcharge(e.account_code, e.account_name));
		const financial = expenses.filter((e: any) => isFinancialExpense(e.account_code, e.account_name));
		const operating = expenses.filter((e: any) =>
			!isCostAccount(e.account_code, e.account_name) &&
			!isTaxSurcharge(e.account_code, e.account_name) &&
			!isFinancialExpense(e.account_code, e.account_name)
		);

		const sumAmt = (items: any[]) => items.reduce((s: number, e: any) => s + (parseFloat(e.amount) || 0), 0);
		const sumYtd = (items: any[]) => items.reduce((s: number, e: any) => s + (parseFloat(e.ytd_amount) || 0), 0);

		const totalRevenue = parseFloat(d.total_revenue) || 0;
		const totalRevenueYtd = sumYtd(revenue);
		const totalCosts = sumAmt(costs);
		const totalCostsYtd = sumYtd(costs);
		const totalTaxes = sumAmt(taxes);
		const totalTaxesYtd = sumYtd(taxes);
		const grossProfit = totalRevenue - totalCosts - totalTaxes;
		const grossProfitYtd = totalRevenueYtd - totalCostsYtd - totalTaxesYtd;
		const totalOperating = sumAmt(operating);
		const totalOperatingYtd = sumYtd(operating);
		const totalFinancial = sumAmt(financial);
		const totalFinancialYtd = sumYtd(financial);
		const operatingProfit = grossProfit - totalOperating - totalFinancial;
		const operatingProfitYtd = grossProfitYtd - totalOperatingYtd - totalFinancialYtd;
		const netIncome = parseFloat(d.net_income) || 0;
		const netIncomeYtd = totalRevenueYtd - sumYtd(expenses);

		return {
			revenue, costs, taxes, operating, financial,
			totalRevenue, totalRevenueYtd,
			totalCosts, totalCostsYtd,
			totalTaxes, totalTaxesYtd,
			grossProfit, grossProfitYtd,
			totalOperating, totalOperatingYtd,
			totalFinancial, totalFinancialYtd,
			operatingProfit, operatingProfitYtd,
			netIncome, netIncomeYtd
		};
	}

	$: selectedLabel = monthOptions.find(o => o.value === selectedMonth)?.label ?? '';
	$: selectedFiscalStart = monthOptions.find(o => o.value === selectedMonth)?.fiscalStart ?? '';

	let lineNo = 0;
	function nextLine(): number { return ++lineNo; }
	function resetLines() { lineNo = 0; }

	// === French format helpers ===
	let collapsedSections = new Set<string>();

	function toggleSection(key: string) {
		if (collapsedSections.has(key)) {
			collapsedSections.delete(key);
		} else {
			collapsedSections.add(key);
		}
		collapsedSections = collapsedSections; // trigger reactivity
	}

	const fmtFr = (v: any): string => {
		const n = typeof v === 'string' ? parseFloat(v) : (v ?? 0);
		if (n === 0) return '';
		return n.toLocaleString('fr-FR', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
	};
</script>

<div class="space-y-3">
	<div class="flex flex-wrap gap-3 items-end">
		<div>
			<label class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">{$i18n.t('Month')}</label>
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
	{:else if structured}
		{@const _ = resetLines()}

		{#if isFrench}
			<!-- ========== FRENCH FORMAT: Compte de Résultat (HACANTHE) ========== -->
			<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2">
								{$i18n.t('Compte de résultat')}
							</th>
							<th class="px-2 py-2 text-right w-36">
								<div>{$i18n.t('Current Period')}</div>
								<div class="font-normal normal-case text-gray-400">{selectedLabel}</div>
							</th>
							<th class="px-2 py-2 text-right w-36">
								<div>{$i18n.t('Year to Date')}</div>
								<div class="font-normal normal-case text-gray-400">{selectedFiscalStart} — {data.date_to}</div>
							</th>
						</tr>
					</thead>
					<tbody>
						<!-- ===== 1. Produits d'exploitation ===== -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('prodExpl')}
						>
							<td class="px-3 py-1.5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('prodExpl') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t("Produits d'exploitation")}
							</td>
							{#if collapsedSections.has('prodExpl')}
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpRevenue)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpRevenueYtd)}</td>
							{:else}
								<td class="px-2 py-1.5"></td>
								<td class="px-2 py-1.5"></td>
							{/if}
						</tr>
						{#if !collapsedSections.has('prodExpl')}
							{#each structured.operatingRevenue as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
							<!-- Subtotal -->
							<tr class="text-green-700 dark:text-green-400 font-bold italic">
								<td class="px-3 py-1.5">{$i18n.t("PRODUITS D'EXPLOITATION")}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpRevenue)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpRevenueYtd)}</td>
							</tr>
						{/if}

						<!-- ===== 2. Charges d'exploitation ===== -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('charExpl')}
						>
							<td class="px-3 py-1.5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('charExpl') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t("Charges d'exploitation")}
							</td>
							{#if collapsedSections.has('charExpl')}
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpExpenses)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpExpensesYtd)}</td>
							{:else}
								<td class="px-2 py-1.5"></td>
								<td class="px-2 py-1.5"></td>
							{/if}
						</tr>
						{#if !collapsedSections.has('charExpl')}
							{#each structured.operatingExpenses as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
							<!-- Subtotal -->
							<tr class="text-green-700 dark:text-green-400 font-bold italic">
								<td class="px-3 py-1.5">{$i18n.t("CHARGES D'EXPLOITATION")}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpExpenses)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalOpExpensesYtd)}</td>
							</tr>
						{/if}

						<!-- ===== 3. RÉSULTAT D'EXPLOITATION ===== -->
						<tr class="bg-green-50/50 dark:bg-green-900/15 font-bold text-green-700 dark:text-green-400">
							<td class="px-3 py-1.5">{$i18n.t("RÉSULTAT D'EXPLOITATION")}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatExploitation)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatExploitationYtd)}</td>
						</tr>

						<!-- ===== 4. Produits financiers ===== -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('prodFin')}
						>
							<td class="px-3 py-1.5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('prodFin') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Produits financiers')}
							</td>
							{#if collapsedSections.has('prodFin')}
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinRevenue)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinRevenueYtd)}</td>
							{:else}
								<td class="px-2 py-1.5"></td>
								<td class="px-2 py-1.5"></td>
							{/if}
						</tr>
						{#if !collapsedSections.has('prodFin')}
							{#each structured.financialRevenue as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
							<tr class="text-green-700 dark:text-green-400 font-bold italic">
								<td class="px-3 py-1.5">{$i18n.t('PRODUITS FINANCIERS')}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinRevenue)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinRevenueYtd)}</td>
							</tr>
						{/if}

						<!-- ===== 5. Charges financières ===== -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('charFin')}
						>
							<td class="px-3 py-1.5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('charFin') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Charges financières')}
							</td>
							{#if collapsedSections.has('charFin')}
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinExpenses)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinExpensesYtd)}</td>
							{:else}
								<td class="px-2 py-1.5"></td>
								<td class="px-2 py-1.5"></td>
							{/if}
						</tr>
						{#if !collapsedSections.has('charFin')}
							{#each structured.financialExpenses as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
							<tr class="text-green-700 dark:text-green-400 font-bold italic">
								<td class="px-3 py-1.5">{$i18n.t('CHARGES FINANCIÈRES')}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinExpenses)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalFinExpensesYtd)}</td>
							</tr>
						{/if}

						<!-- ===== 6. RÉSULTAT FINANCIER ===== -->
						<tr class="bg-green-50/50 dark:bg-green-900/15 font-bold text-green-700 dark:text-green-400">
							<td class="px-3 py-1.5">{$i18n.t('RÉSULTAT FINANCIER')}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatFinancier)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatFinancierYtd)}</td>
						</tr>

						<!-- ===== 7. RÉSULTAT COURANT AVANT IMPÔTS ===== -->
						<tr class="bg-green-50/50 dark:bg-green-900/15 font-bold text-green-700 dark:text-green-400">
							<td class="px-3 py-1.5">{$i18n.t('RÉSULTAT COURANT AVANT IMPÔTS')}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatCourant)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatCourantYtd)}</td>
						</tr>

						<!-- ===== 8. Produits exceptionnels ===== -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('prodExc')}
						>
							<td class="px-3 py-1.5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('prodExc') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Produits exceptionnels')}
							</td>
							{#if collapsedSections.has('prodExc')}
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcRevenue)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcRevenueYtd)}</td>
							{:else}
								<td class="px-2 py-1.5"></td>
								<td class="px-2 py-1.5"></td>
							{/if}
						</tr>
						{#if !collapsedSections.has('prodExc')}
							{#each structured.exceptionalRevenue as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
							<tr class="text-green-700 dark:text-green-400 font-bold italic">
								<td class="px-3 py-1.5">{$i18n.t('PRODUITS EXCEPTIONNELS')}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcRevenue)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcRevenueYtd)}</td>
							</tr>
						{/if}

						<!-- ===== 9. Charges exceptionnelles ===== -->
						<tr
							class="bg-green-600 text-white text-xs font-semibold cursor-pointer select-none"
							on:click={() => toggleSection('charExc')}
						>
							<td class="px-3 py-1.5">
								<span class="mr-1 inline-block text-[10px]">{collapsedSections.has('charExc') ? '\u25B6' : '\u25BC'}</span>
								{$i18n.t('Charges exceptionnelles')}
							</td>
							{#if collapsedSections.has('charExc')}
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcExpenses)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcExpensesYtd)}</td>
							{:else}
								<td class="px-2 py-1.5"></td>
								<td class="px-2 py-1.5"></td>
							{/if}
						</tr>
						{#if !collapsedSections.has('charExc')}
							{#each structured.exceptionalExpenses as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
							<tr class="text-green-700 dark:text-green-400 font-bold italic">
								<td class="px-3 py-1.5">{$i18n.t('CHARGES EXCEPTIONNELLES')}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcExpenses)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalExcExpensesYtd)}</td>
							</tr>
						{/if}

						<!-- ===== 10. RÉSULTAT EXCEPTIONNEL ===== -->
						<tr class="bg-green-50/50 dark:bg-green-900/15 font-bold text-green-700 dark:text-green-400">
							<td class="px-3 py-1.5">{$i18n.t('RÉSULTAT EXCEPTIONNEL')}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatExceptionnel)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.resultatExceptionnelYtd)}</td>
						</tr>

						<!-- ===== 11. Participation & Impôts ===== -->
						{#each structured.incomeTax as item}
							<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
								<td class="px-3 py-1">{item.account_name}</td>
								<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
								<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
							</tr>
						{/each}

						<!-- Uncategorized (safety net) -->
						{#if structured.uncategorizedRevenue.length > 0}
							{#each structured.uncategorizedRevenue as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30 text-gray-500">
									<td class="px-3 py-1">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
						{/if}
						{#if structured.uncategorizedExpenses.length > 0}
							{#each structured.uncategorizedExpenses as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30 text-gray-500">
									<td class="px-3 py-1">{item.account_name}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmtFr(item.ytd_amount)}</td>
								</tr>
							{/each}
						{/if}

						<!-- ===== 12. TOTAL DES PRODUITS ===== -->
						<tr class="border-t-2 border-gray-300 dark:border-gray-600 text-green-700 dark:text-green-400 font-bold italic">
							<td class="px-3 py-1.5">{$i18n.t('TOTAL DES PRODUITS')}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalProduits)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalProduitsYtd)}</td>
						</tr>

						<!-- ===== 13. TOTAL DES CHARGES ===== -->
						<tr class="text-green-700 dark:text-green-400 font-bold italic">
							<td class="px-3 py-1.5">{$i18n.t('TOTAL DES CHARGES')}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalCharges)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmtFr(structured.totalChargesYtd)}</td>
						</tr>
					</tbody>

					<!-- ===== 14. BÉNÉFICE OU PERTE ===== -->
					<tfoot>
						<tr class="border-t-2 border-gray-300 dark:border-gray-600 bg-green-600/20 font-bold text-green-700 dark:text-green-400 text-sm">
							<td class="px-3 py-2.5">{$i18n.t('BÉNÉFICE OU PERTE (Total des produits - Total des charges)')}</td>
							<td class="px-2 py-2.5 text-right font-mono">{fmtFr(structured.netIncome)}</td>
							<td class="px-2 py-2.5 text-right font-mono">{fmtFr(structured.netIncomeYtd)}</td>
						</tr>
					</tfoot>
				</table>
			</div>
		{:else}
			<!-- ========== DEFAULT FORMAT (US/other) ========== -->
			<div class="overflow-x-auto bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30">
				<table class="w-full text-xs text-left text-gray-700 dark:text-gray-300">
					<thead class="text-[10px] uppercase bg-gray-50 dark:bg-gray-850/50 text-gray-600 dark:text-gray-400">
						<tr>
							<th class="px-3 py-2" colspan="2">
								{$i18n.t('PROFIT & LOSS')}
							</th>
							<th class="px-2 py-2 text-right w-36">
								<div>{$i18n.t('Current Period')}</div>
								<div class="font-normal normal-case text-gray-400">{selectedLabel}</div>
							</th>
							<th class="px-2 py-2 text-right w-36">
								<div>{$i18n.t('Year to Date')}</div>
								<div class="font-normal normal-case text-gray-400">{selectedFiscalStart} — {data.date_to}</div>
							</th>
						</tr>
					</thead>
					<tbody>
						<!-- Revenue -->
						<tr class="font-semibold bg-green-50/30 dark:bg-green-900/10">
							<td class="px-3 py-1.5">{$i18n.t('Revenue')}</td>
							<td class="px-2 py-1.5 text-center w-10">{nextLine()}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalRevenue)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalRevenueYtd)}</td>
						</tr>
						{#each structured.revenue as item}
							<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
								<td class="px-3 py-1" style="padding-left: {20 + item.level * 16}px">{item.account_name}</td>
								<td class="px-2 py-1 text-center text-gray-400">{nextLine()}</td>
								<td class="px-2 py-1 text-right font-mono">{fmt(item.amount)}</td>
								<td class="px-2 py-1 text-right font-mono">{fmt(item.ytd_amount)}</td>
							</tr>
						{/each}

						<!-- Cost of sales -->
						{#if structured.costs.length > 0}
							<tr class="text-gray-600 dark:text-gray-400">
								<td class="px-3 py-1.5 italic">{$i18n.t('Less: Cost of sales')}</td>
								<td class="px-2 py-1.5 text-center">{nextLine()}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalCosts)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalCostsYtd)}</td>
							</tr>
							{#each structured.costs as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {28 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-center text-gray-400">{nextLine()}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(item.ytd_amount)}</td>
								</tr>
							{/each}
						{/if}

						<!-- Taxes and surcharges -->
						{#if structured.taxes.length > 0}
							<tr class="text-gray-600 dark:text-gray-400">
								<td class="px-3 py-1.5 italic">{$i18n.t('Less: Taxes and surcharges')}</td>
								<td class="px-2 py-1.5 text-center">{nextLine()}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalTaxes)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalTaxesYtd)}</td>
							</tr>
						{/if}

						<!-- Gross Profit -->
						<tr class="font-semibold border-t-2 border-gray-200 dark:border-gray-700 bg-blue-50/30 dark:bg-blue-900/10">
							<td class="px-3 py-1.5">{$i18n.t('Gross Profit')}</td>
							<td class="px-2 py-1.5 text-center">{nextLine()}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.grossProfit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.grossProfitYtd)}</td>
						</tr>

						<!-- Operating expenses -->
						{#if structured.operating.length > 0}
							<tr class="text-gray-600 dark:text-gray-400">
								<td class="px-3 py-1.5 italic">{$i18n.t('Less: Operating expenses')}</td>
								<td class="px-2 py-1.5 text-center">{nextLine()}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalOperating)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalOperatingYtd)}</td>
							</tr>
							{#each structured.operating as item}
								<tr class="border-b border-gray-50 dark:border-gray-850/30 hover:bg-gray-50/50 dark:hover:bg-gray-850/30">
									<td class="px-3 py-1" style="padding-left: {28 + item.level * 16}px">{item.account_name}</td>
									<td class="px-2 py-1 text-center text-gray-400">{nextLine()}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(item.amount)}</td>
									<td class="px-2 py-1 text-right font-mono">{fmt(item.ytd_amount)}</td>
								</tr>
							{/each}
						{/if}

						<!-- Financial expenses -->
						{#if structured.financial.length > 0}
							<tr class="text-gray-600 dark:text-gray-400">
								<td class="px-3 py-1.5 italic">{$i18n.t('Less: Financial expenses')}</td>
								<td class="px-2 py-1.5 text-center">{nextLine()}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalFinancial)}</td>
								<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.totalFinancialYtd)}</td>
							</tr>
						{/if}

						<!-- Operating Profit -->
						<tr class="font-semibold border-t-2 border-gray-200 dark:border-gray-700 bg-blue-50/30 dark:bg-blue-900/10">
							<td class="px-3 py-1.5">{$i18n.t('Operating Profit')}</td>
							<td class="px-2 py-1.5 text-center">{nextLine()}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.operatingProfit)}</td>
							<td class="px-2 py-1.5 text-right font-mono">{fmt(structured.operatingProfitYtd)}</td>
						</tr>
					</tbody>
					<tfoot class="font-bold text-sm {structured.netIncome >= 0 ? 'bg-green-50/50 dark:bg-green-900/15' : 'bg-red-50/50 dark:bg-red-900/15'}">
						<tr class="border-t-2 border-gray-300 dark:border-gray-600">
							<td class="px-3 py-2.5" colspan="2">{$i18n.t('Net Income')}</td>
							<td class="px-2 py-2.5 text-right font-mono {structured.netIncome >= 0 ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}">{fmt(structured.netIncome)}</td>
							<td class="px-2 py-2.5 text-right font-mono {structured.netIncomeYtd >= 0 ? 'text-green-700 dark:text-green-400' : 'text-red-700 dark:text-red-400'}">{fmt(structured.netIncomeYtd)}</td>
						</tr>
					</tfoot>
				</table>
			</div>
		{/if}
	{/if}
</div>
