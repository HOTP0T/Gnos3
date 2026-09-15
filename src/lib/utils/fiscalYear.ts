/**
 * Fiscal-year helpers shared by the report and closing screens.
 *
 * A company's fiscal year starts in `fiscal_year_start_month` (1 = calendar
 * year, 4 = April – March as most Hong Kong companies use). Everything that
 * needs "the start of the year this month belongs to" — YTD columns, the
 * year-end close, the profits-tax period — reads it from here instead of
 * assuming the accounting period's start date or December.
 */

const pad = (n: number) => String(n).padStart(2, '0');

/** First day of the fiscal year containing `dateStr` (YYYY-MM-DD). */
export function fiscalYearStart(dateStr: string, startMonth = 1): string {
	const [y, m] = dateStr.split('-').map(Number);
	const sm = startMonth >= 1 && startMonth <= 12 ? startMonth : 1;
	const startYear = m >= sm ? y : y - 1;
	return `${startYear}-${pad(sm)}-01`;
}

/** Last day of the fiscal year containing `dateStr`. */
export function fiscalYearEnd(dateStr: string, startMonth = 1): string {
	const start = fiscalYearStart(dateStr, startMonth);
	const [y, m] = start.split('-').map(Number);
	const endYear = m === 1 ? y : y + 1;
	const endMonth = m === 1 ? 12 : m - 1;
	const lastDay = new Date(endYear, endMonth, 0).getDate();
	return `${endYear}-${pad(endMonth)}-${pad(lastDay)}`;
}

/** "2026" for a calendar year, "2026/27" for one straddling two. */
export function fiscalYearLabel(start: string, end: string): string {
	const sy = start.slice(0, 4);
	const ey = end.slice(0, 4);
	return sy === ey ? sy : `${sy}/${ey.slice(2)}`;
}

/** Whether the month `YYYY-MM` is the last month of the fiscal year. */
export function isFiscalYearEndMonth(ym: string, startMonth = 1): boolean {
	const m = Number(ym.slice(5, 7));
	const sm = startMonth >= 1 && startMonth <= 12 ? startMonth : 1;
	const endMonth = sm === 1 ? 12 : sm - 1;
	return m === endMonth;
}

export type MonthOption = {
	value: string; // YYYY-MM
	label: string;
	from: string; // first day of the month
	to: string; // last day of the month
	asOf: string; // alias of `to`
	fiscalStart: string;
	fiscalEnd: string;
	fiscalYear: string;
	isFiscalYearEnd: boolean;
};

/**
 * One option per month covered by the company's accounting periods, newest
 * first, each carrying the fiscal-year bounds of the month.
 */
export function buildMonthOptions(periods: any[], startMonth = 1): MonthOption[] {
	const options: MonthOption[] = [];
	for (const p of periods ?? []) {
		if (!p?.start_date || !p?.end_date) continue;
		const start = new Date(p.start_date);
		const end = new Date(p.end_date);
		let cursor = new Date(start.getFullYear(), start.getMonth(), 1);
		while (cursor <= end) {
			const y = cursor.getFullYear();
			const m = cursor.getMonth();
			const from = `${y}-${pad(m + 1)}-01`;
			const lastDay = new Date(y, m + 1, 0).getDate();
			const to = `${y}-${pad(m + 1)}-${pad(lastDay)}`;
			const label = cursor.toLocaleDateString(undefined, { year: 'numeric', month: 'long' });
			const fiscalStart = fiscalYearStart(from, startMonth);
			const fiscalEnd = fiscalYearEnd(from, startMonth);
			const value = `${y}-${pad(m + 1)}`;
			options.push({
				value,
				label,
				from,
				to,
				asOf: to,
				fiscalStart,
				fiscalEnd,
				fiscalYear: fiscalYearLabel(fiscalStart, fiscalEnd),
				isFiscalYearEnd: isFiscalYearEndMonth(value, startMonth)
			});
			cursor = new Date(y, m + 1, 1);
		}
	}
	const seen = new Map<string, MonthOption>();
	for (const o of options) seen.set(o.value, o);
	return Array.from(seen.values()).sort((a, b) => b.value.localeCompare(a.value));
}

export const MONTH_NAMES = [
	'January',
	'February',
	'March',
	'April',
	'May',
	'June',
	'July',
	'August',
	'September',
	'October',
	'November',
	'December'
];
