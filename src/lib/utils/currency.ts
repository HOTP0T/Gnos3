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
