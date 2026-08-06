// Widget catalog for the configurable accounting dashboard.
//
// Each entry describes one widget "type": what component renders it, its
// default/min grid size, which card chrome it wants, and any fixed options
// (e.g. which KPI metric or chart mode). The ConfigurableDashboard renders
// instances of these on a gridstack grid; the layout (which types are placed
// where/at what size) is what gets persisted per user+company.

import KpiWidget from './widgets/KpiWidget.svelte';
import TrendChartWidget from './widgets/TrendChartWidget.svelte';
import BreakdownChartWidget from './widgets/BreakdownChartWidget.svelte';
import AgingChartWidget from './widgets/AgingChartWidget.svelte';
import BalanceCompositionWidget from './widgets/BalanceCompositionWidget.svelte';
import CashFlowWidget from './widgets/CashFlowWidget.svelte';
import TopPartiesWidget from './widgets/TopPartiesWidget.svelte';
import TransactionsListWidget from './widgets/TransactionsListWidget.svelte';
import AuditListWidget from './widgets/AuditListWidget.svelte';
import UnmatchedListWidget from './widgets/UnmatchedListWidget.svelte';
import RecurringListWidget from './widgets/RecurringListWidget.svelte';
import PeriodSummaryWidget from './widgets/PeriodSummaryWidget.svelte';
import AlertsWidget from './widgets/AlertsWidget.svelte';
import QuickActionsWidget from './widgets/QuickActionsWidget.svelte';

export type WidgetCategory = 'KPI' | 'Charts' | 'Lists' | 'Panels';

export interface WidgetDef {
	type: string;
	title: string;
	category: WidgetCategory;
	description: string;
	component: any;
	options?: Record<string, any>;
	w: number;
	h: number;
	minW: number;
	minH: number;
	/** hide the card title bar (widget renders its own chrome) */
	bare?: boolean;
	/** card body has no padding (widget manages its own) */
	flush?: boolean;
	link?: (companyId: number) => string;
	linkLabel?: string;
}

const kpi = (
	type: string,
	title: string,
	description: string,
	metric: string
): WidgetDef => ({
	type,
	title,
	category: 'KPI',
	description,
	component: KpiWidget,
	options: { metric },
	w: 3,
	h: 2,
	minW: 2,
	minH: 2,
	bare: true
});

export const WIDGETS: WidgetDef[] = [
	// ── KPI tiles ──
	kpi('kpi_assets', 'Total Assets', 'Total assets from the balance sheet.', 'assets'),
	kpi('kpi_liabilities', 'Total Liabilities', 'Total liabilities from the balance sheet.', 'liabilities'),
	kpi('kpi_net_income', 'Net Income (MTD)', 'Net income for the current month.', 'net_income'),
	kpi('kpi_cash', 'Cash Position', 'Balance across all bank-linked accounts.', 'cash'),
	kpi('kpi_ar', 'Accounts Receivable', 'Outstanding money owed to you.', 'ar'),
	kpi('kpi_ap', 'Accounts Payable', 'Outstanding money you owe.', 'ap'),
	kpi('kpi_overdue', 'Overdue Invoices', 'Count and amount of past-due invoices.', 'overdue'),
	kpi('kpi_drafts', 'Draft Entries', 'Unposted journal entries awaiting review.', 'drafts'),
	kpi('kpi_unmatched', 'Unmatched Bank Lines', 'Bank statement lines not yet reconciled.', 'unmatched'),
	kpi('kpi_current_ratio', 'Assets / Liabilities', 'Ratio of total assets to total liabilities.', 'current_ratio'),

	// ── Charts ──
	{
		type: 'chart_revenue_expenses',
		title: 'Revenue vs Expenses',
		category: 'Charts',
		description: 'Monthly revenue and expenses over the last year.',
		component: TrendChartWidget,
		options: { mode: 'revenue_expenses', months: 12 },
		w: 8, h: 5, minW: 4, minH: 4
	},
	{
		type: 'chart_net_income',
		title: 'Net Income Trend',
		category: 'Charts',
		description: 'Monthly net income (revenue − expenses) over time.',
		component: TrendChartWidget,
		options: { mode: 'net_income', months: 12 },
		w: 6, h: 5, minW: 4, minH: 4
	},
	{
		type: 'chart_expense_breakdown',
		title: 'Expense Breakdown',
		category: 'Charts',
		description: 'Year-to-date expenses by account (donut).',
		component: BreakdownChartWidget,
		options: { kind: 'expenses', range: 'ytd' },
		w: 4, h: 5, minW: 3, minH: 4
	},
	{
		type: 'chart_revenue_breakdown',
		title: 'Revenue Breakdown',
		category: 'Charts',
		description: 'Year-to-date revenue by account (donut).',
		component: BreakdownChartWidget,
		options: { kind: 'revenue', range: 'ytd' },
		w: 4, h: 5, minW: 3, minH: 4
	},
	{
		type: 'chart_ar_aging',
		title: 'AR Aging',
		category: 'Charts',
		description: 'Receivables by age bucket (current → 90d+).',
		component: AgingChartWidget,
		options: { kind: 'ar' },
		w: 4, h: 5, minW: 3, minH: 4
	},
	{
		type: 'chart_ap_aging',
		title: 'AP Aging',
		category: 'Charts',
		description: 'Payables by age bucket (current → 90d+).',
		component: AgingChartWidget,
		options: { kind: 'ap' },
		w: 4, h: 5, minW: 3, minH: 4
	},
	{
		type: 'chart_balance_composition',
		title: 'Balance Sheet Composition',
		category: 'Charts',
		description: 'Assets vs Liabilities + Equity.',
		component: BalanceCompositionWidget,
		w: 4, h: 5, minW: 3, minH: 4
	},
	{
		type: 'chart_cash_flow',
		title: 'Cash Flow Waterfall',
		category: 'Charts',
		description: 'YTD operating / investing / financing cash movement.',
		component: CashFlowWidget,
		options: { range: 'ytd' },
		w: 6, h: 5, minW: 4, minH: 4
	},
	{
		type: 'chart_top_vendors',
		title: 'Top Vendors',
		category: 'Charts',
		description: 'Vendors ranked by total billed.',
		component: TopPartiesWidget,
		options: { direction: 'vendors', limit: 8 },
		w: 6, h: 5, minW: 3, minH: 4
	},
	{
		type: 'chart_top_customers',
		title: 'Top Customers',
		category: 'Charts',
		description: 'Customers ranked by total invoiced.',
		component: TopPartiesWidget,
		options: { direction: 'customers', limit: 8 },
		w: 6, h: 5, minW: 3, minH: 4
	},

	// ── Lists ──
	{
		type: 'list_transactions',
		title: 'Recent Transactions',
		category: 'Lists',
		description: 'The latest journal entries.',
		component: TransactionsListWidget,
		options: { status: 'all', limit: 25 },
		w: 6, h: 6, minW: 4, minH: 4,
		flush: true,
		link: (id) => `/accounting/company/${id}/entries`
	},
	{
		type: 'list_drafts',
		title: 'Draft Entries',
		category: 'Lists',
		description: 'Unposted entries awaiting review.',
		component: TransactionsListWidget,
		options: { status: 'draft', limit: 25 },
		w: 6, h: 6, minW: 4, minH: 4,
		flush: true,
		link: (id) => `/accounting/company/${id}/entries`
	},
	{
		type: 'list_activity',
		title: 'Recent Activity',
		category: 'Lists',
		description: 'Audit trail of posted / voided / edited entries.',
		component: AuditListWidget,
		options: { limit: 15 },
		w: 6, h: 6, minW: 4, minH: 4,
		flush: true
	},
	{
		type: 'list_unmatched',
		title: 'Unmatched Bank Lines',
		category: 'Lists',
		description: 'Bank lines awaiting reconciliation.',
		component: UnmatchedListWidget,
		options: { limit: 25 },
		w: 4, h: 6, minW: 3, minH: 4,
		flush: true,
		link: (id) => `/accounting/company/${id}/bank`,
		linkLabel: 'Reconcile'
	},
	{
		type: 'list_recurring',
		title: 'Upcoming Recurring',
		category: 'Lists',
		description: 'Recurring entries by next run date.',
		component: RecurringListWidget,
		options: { limit: 15 },
		w: 4, h: 6, minW: 3, minH: 4,
		flush: true,
		link: (id) => `/accounting/company/${id}/recurring`
	},

	// ── Panels ──
	{
		type: 'panel_alerts',
		title: 'Alerts',
		category: 'Panels',
		description: 'Actionable items: drafts, unmatched lines, overdue invoices.',
		component: AlertsWidget,
		w: 6, h: 3, minW: 3, minH: 2
	},
	{
		type: 'panel_quick_actions',
		title: 'Quick Actions',
		category: 'Panels',
		description: 'Shortcuts to create entries, payments, reports.',
		component: QuickActionsWidget,
		w: 6, h: 2, minW: 3, minH: 2
	},
	{
		type: 'panel_period',
		title: 'Current Period',
		category: 'Panels',
		description: 'The open accounting period at a glance.',
		component: PeriodSummaryWidget,
		w: 4, h: 2, minW: 3, minH: 2
	}
];

export const WIDGET_BY_TYPE: Record<string, WidgetDef> = Object.fromEntries(
	WIDGETS.map((w) => [w.type, w])
);

export interface LayoutItem {
	id: string;
	type: string;
	x: number;
	y: number;
	w: number;
	h: number;
	options?: Record<string, any>;
}

// Default layout — reproduces the classic dashboard (two KPI rows, alerts,
// quick actions, the revenue/expense trend, breakdown, and the two lists) so
// nothing is lost for users who never customize.
export const DEFAULT_LAYOUT: LayoutItem[] = [
	{ id: 'd1', type: 'kpi_assets', x: 0, y: 0, w: 3, h: 2 },
	{ id: 'd2', type: 'kpi_liabilities', x: 3, y: 0, w: 3, h: 2 },
	{ id: 'd3', type: 'kpi_net_income', x: 6, y: 0, w: 3, h: 2 },
	{ id: 'd4', type: 'kpi_drafts', x: 9, y: 0, w: 3, h: 2 },
	{ id: 'd5', type: 'kpi_cash', x: 0, y: 2, w: 3, h: 2 },
	{ id: 'd6', type: 'kpi_ar', x: 3, y: 2, w: 3, h: 2 },
	{ id: 'd7', type: 'kpi_ap', x: 6, y: 2, w: 3, h: 2 },
	{ id: 'd8', type: 'kpi_overdue', x: 9, y: 2, w: 3, h: 2 },
	{ id: 'd9', type: 'panel_alerts', x: 0, y: 4, w: 6, h: 3 },
	{ id: 'd10', type: 'panel_quick_actions', x: 6, y: 4, w: 6, h: 2 },
	{ id: 'd11', type: 'chart_revenue_expenses', x: 0, y: 7, w: 8, h: 5 },
	{ id: 'd12', type: 'chart_expense_breakdown', x: 8, y: 7, w: 4, h: 5 },
	{ id: 'd13', type: 'list_transactions', x: 0, y: 12, w: 6, h: 6 },
	{ id: 'd14', type: 'list_activity', x: 6, y: 12, w: 6, h: 6 }
];

export function defaultLayout(): LayoutItem[] {
	return DEFAULT_LAYOUT.map((d) => ({ ...d }));
}
