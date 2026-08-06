// Validated data-viz palette (dataviz skill reference instance).
// Categorical hues are assigned in fixed slot order — never cycled — which is
// the CVD-safety mechanism. Light/dark are selected sets, not an auto-flip.

export const CATEGORICAL = {
	light: ['#2a78d6', '#1baf7a', '#eda100', '#008300', '#4a3aa7', '#e34948', '#e87ba4', '#eb6834'],
	dark: ['#3987e5', '#199e70', '#c98500', '#008300', '#9085e9', '#e66767', '#d55181', '#d95926']
};

// Status palette (fixed — never themed, never used as a series color).
export const STATUS = {
	good: '#0ca30c',
	warning: '#fab219',
	serious: '#ec835a',
	critical: '#d03b3b'
};

// Chart chrome / ink, per mode.
export interface ChartTheme {
	isDark: boolean;
	surface: string;
	primary: string;
	secondary: string;
	muted: string;
	grid: string;
	axis: string;
	series: string[];
	// Semantic financial pair (categorical slots, well-separated): revenue vs expenses.
	revenue: string;
	expenses: string;
	net: string;
	good: string;
	critical: string;
	warning: string;
	serious: string;
}

export function chartTheme(isDark: boolean): ChartTheme {
	if (isDark) {
		return {
			isDark,
			surface: '#1a1a19',
			primary: '#ffffff',
			secondary: '#c3c2b7',
			muted: '#898781',
			grid: '#2c2c2a',
			axis: '#383835',
			series: CATEGORICAL.dark,
			revenue: '#199e70', // aqua slot
			expenses: '#d95926', // orange slot
			net: '#3987e5', // blue slot
			good: STATUS.good,
			critical: STATUS.critical,
			warning: STATUS.warning,
			serious: STATUS.serious
		};
	}
	return {
		isDark,
		surface: '#fcfcfb',
		primary: '#0b0b0b',
		secondary: '#52514e',
		muted: '#898781',
		grid: '#e1e0d9',
		axis: '#c3c2b7',
		series: CATEGORICAL.light,
		revenue: '#1baf7a',
		expenses: '#eb6834',
		net: '#2a78d6',
		good: STATUS.good,
		critical: STATUS.critical,
		warning: STATUS.warning,
		serious: STATUS.serious
	};
}

// Shared tooltip styling for ECharts, theme-aware.
export function tooltipStyle(t: ChartTheme) {
	return {
		backgroundColor: t.isDark ? 'rgba(17,24,39,0.95)' : 'rgba(255,255,255,0.97)',
		borderColor: t.isDark ? 'rgba(75,85,99,0.4)' : 'rgba(0,0,0,0.1)',
		borderWidth: 1,
		textStyle: { color: t.primary, fontSize: 12 },
		extraCssText: 'box-shadow: 0 4px 12px rgba(0,0,0,0.15); border-radius: 8px;'
	};
}
