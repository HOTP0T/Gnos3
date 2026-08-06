/**
 * Currency conversion utility for the accounting module.
 *
 * Used by components that read `displayCurrency` and `exchangeRates` from Svelte context.
 * Finds the nearest exchange rate by date for a given currency pair.
 */

export interface ExchangeRate {
	from_currency: string;
	to_currency: string;
	rate: number;
	effective_date: string;
}

export interface ConversionResult {
	converted: number;
	rate: number;
	rateDate: string;
	hasRate: boolean;
}

/**
 * Convert an amount from one currency to another using the nearest exchange rate.
 *
 * @param amount - The amount to convert
 * @param fromCurrency - Source currency code (e.g., "EUR")
 * @param toCurrency - Target currency code (e.g., "USD")
 * @param rates - Array of exchange rates loaded from the API
 * @param asOf - Date to find the nearest rate for (YYYY-MM-DD string or Date)
 * @returns ConversionResult with converted amount, rate used, and whether a rate was found
 */
export function convertAmount(
	amount: number,
	fromCurrency: string,
	toCurrency: string,
	rates: ExchangeRate[],
	asOf?: string | Date
): ConversionResult {
	if (!amount || fromCurrency === toCurrency) {
		return { converted: amount, rate: 1, rateDate: '', hasRate: true };
	}

	const targetDate = asOf
		? (typeof asOf === 'string' ? asOf : asOf.toISOString().slice(0, 10))
		: new Date().toISOString().slice(0, 10);

	// Find direct rate: from→to
	let bestRate: ExchangeRate | null = null;
	let bestDistance = Infinity;
	let inverse = false;

	for (const r of rates) {
		if (r.from_currency === fromCurrency && r.to_currency === toCurrency) {
			const dist = Math.abs(new Date(r.effective_date).getTime() - new Date(targetDate).getTime());
			if (dist < bestDistance) {
				bestDistance = dist;
				bestRate = r;
				inverse = false;
			}
		}
		// Also check inverse: to→from
		if (r.from_currency === toCurrency && r.to_currency === fromCurrency) {
			const dist = Math.abs(new Date(r.effective_date).getTime() - new Date(targetDate).getTime());
			if (dist < bestDistance) {
				bestDistance = dist;
				bestRate = r;
				inverse = true;
			}
		}
	}

	if (!bestRate) {
		return { converted: amount, rate: 0, rateDate: '', hasRate: false };
	}

	const effectiveRate = inverse ? (1 / bestRate.rate) : bestRate.rate;
	return {
		converted: Math.round(amount * effectiveRate * 100) / 100,
		rate: effectiveRate,
		rateDate: bestRate.effective_date,
		hasRate: true,
	};
}

/**
 * Format a monetary amount with currency code.
 */
export function formatMoney(amount: number, currency: string, locale: string = 'fr-FR'): string {
	if (amount === 0) return '';
	return `${amount.toLocaleString(locale, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${currency}`;
}

/**
 * Money field names across the accounting report payloads (GL account view,
 * Trial Balance, P&L, Balance Sheet, Cash Flow). A value found under one of
 * these keys — whether a JSON number (Cash Flow) or a Decimal-as-string (the
 * others) — is a monetary amount in the company base currency and gets scaled
 * by the display-currency factor. Everything else (account codes/names, dates,
 * exchange_rate, ids, counts, levels, boolean flags) is left untouched.
 */
const REPORT_MONEY_KEYS = new Set<string>([
	// balances
	'opening_balance', 'closing_balance', 'running_balance', 'beginning_balance', 'balance',
	// ledger debit/credit
	'debit', 'credit',
	// trial-balance columns
	'opening_debit', 'opening_credit', 'movement_debit', 'movement_credit',
	'accumulated_debit', 'accumulated_credit', 'ending_debit', 'ending_credit',
	// trial-balance grand totals
	'total_debit', 'total_credit',
	'total_opening_debit', 'total_opening_credit', 'total_movement_debit', 'total_movement_credit',
	'total_accumulated_debit', 'total_accumulated_credit',
	// P&L
	'amount', 'ytd_amount',
	'total_revenue', 'total_revenue_ytd', 'total_expenses', 'total_expenses_ytd',
	'net_income', 'net_income_ytd',
	// balance sheet
	'total_assets', 'total_liabilities', 'total_equity',
	// cash flow
	'depreciation', 'change',
	'cash_from_operations', 'cash_from_investing', 'cash_from_financing',
	'net_change_in_cash', 'opening_cash', 'closing_cash'
]);

/**
 * Deep-copy a report payload, multiplying every monetary leaf (identified by the
 * REPORT_MONEY_KEYS allow-list) by `factor`. Non-money values are returned
 * unchanged. Money strings are parsed and returned as numbers (all report
 * fmt()/parseFloat consumers accept numbers). Scaling the raw leaves + stored
 * totals keeps every client-derived subtotal correct because those are linear
 * sums, and boolean flags like `is_balanced` remain valid under uniform scaling.
 */
export function scaleReportMoney(
	node: any,
	factor: number,
	extraMoneyKeys?: Set<string>,
	key?: string
): any {
	const isMoney = (k?: string): boolean =>
		!!k && (REPORT_MONEY_KEYS.has(k) || (extraMoneyKeys?.has(k) ?? false));
	if (node == null) return node;
	if (typeof node === 'number') {
		return isMoney(key) ? node * factor : node;
	}
	if (typeof node === 'string') {
		if (isMoney(key) && node.trim() !== '') {
			const n = parseFloat(node);
			if (!isNaN(n)) return n * factor;
		}
		return node;
	}
	if (Array.isArray(node)) {
		return node.map((v) => scaleReportMoney(v, factor, extraMoneyKeys, key));
	}
	if (typeof node === 'object') {
		const out: Record<string, any> = {};
		for (const k of Object.keys(node)) {
			out[k] = scaleReportMoney(node[k], factor, extraMoneyKeys, k);
		}
		return out;
	}
	return node;
}
