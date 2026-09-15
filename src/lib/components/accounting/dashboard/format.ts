import { convertAmount, convertRowAmount } from '$lib/utils/currency';

export function fmtNumber(v: number | string, opts?: Intl.NumberFormatOptions): string {
	const num = typeof v === 'string' ? parseFloat(v) : v ?? 0;
	return num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2, ...opts });
}

export function fmtNative(v: number | string, currency: string): string {
	return `${fmtNumber(v)} ${currency || ''}`.trim();
}

// Compact axis/label numbers: 1.2k, 3.4M.
export function fmtCompact(v: number): string {
	const abs = Math.abs(v);
	if (abs >= 1_000_000) return (v / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1) + 'M';
	if (abs >= 1_000) return (v / 1_000).toFixed(abs >= 10_000 ? 0 : 1) + 'k';
	return String(Math.round(v));
}

export interface MoneyView {
	converting: boolean;
	hasRate: boolean;
	display: string; // converted, no currency suffix
	original: string; // native, no currency suffix
}

// Mirror of AccountingDashboard's cvt(): converts a BASE-currency amount (a KPI,
// a report figure) to the display currency when one is selected and differs
// from the company's native currency. For a row that has its own currency (an
// entry, a payment) use `rowMoney` — never this.
export function money(
	amount: number | string | null | undefined,
	nativeCurrency: string,
	displayCurrency: string,
	rates: any[],
	date?: string
): MoneyView {
	const num = typeof amount === 'string' ? parseFloat(amount) : amount ?? 0;
	const converting = !!displayCurrency && displayCurrency !== nativeCurrency;
	if (!converting) {
		return { converting: false, hasRate: true, display: fmtNumber(num), original: fmtNumber(num) };
	}
	const r = convertAmount(num, nativeCurrency, displayCurrency, rates ?? [], date);
	return {
		converting: true,
		hasRate: r.hasRate,
		display: r.hasRate ? fmtNumber(r.converted) : '',
		original: fmtNumber(num)
	};
}

// A row in its own currency (an entry's total, a payment) shown in the display
// currency: the conversion starts from the row's currency, and an entry shown
// in the company currency is valued at the rate it was booked at.
export function rowMoney(
	amount: number | string | null | undefined,
	rowCurrency: string | null | undefined,
	nativeCurrency: string,
	displayCurrency: string,
	rates: any[],
	date?: string,
	bookedRate?: number | string | null
): MoneyView & { from: string; to: string } {
	const r = convertRowAmount(amount, rowCurrency, displayCurrency, nativeCurrency, rates ?? [], date, bookedRate);
	return { converting: r.converting, hasRate: r.hasRate, display: r.display, original: r.original, from: r.from, to: r.to };
}
