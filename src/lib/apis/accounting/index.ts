import { toast } from 'svelte-sonner';
import { INVOICE_API_BASE_URL } from '$lib/constants';

const BASE = `${INVOICE_API_BASE_URL}/api/accounting`;

// ─── Helpers ────────────────────────────────────────────────────────────────

function getAuthToken(): string | null {
	if (typeof localStorage === 'undefined') return null;
	return localStorage.getItem('token');
}

function authHeaders(): Record<string, string> {
	const token = getAuthToken();
	return {
		'Content-Type': 'application/json',
		...(token ? { Authorization: `Bearer ${token}` } : {})
	};
}

function uploadAuthHeaders(): Record<string, string> {
	// FormData sets its own multipart Content-Type — never override it here.
	const token = getAuthToken();
	return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseErrorResponse(res: Response): Promise<any> {
	try {
		return await res.json();
	} catch {
		const text = await res.text().catch(() => '');
		return { detail: text || `HTTP ${res.status}` };
	}
}

async function apiGet(path: string, params?: Record<string, string | number | boolean | undefined>) {
	const searchParams = new URLSearchParams();
	if (params) {
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null && v !== '') searchParams.set(k, String(v));
		}
	}
	const qs = searchParams.toString();
	const url = qs ? `${BASE}${path}?${qs}` : `${BASE}${path}`;
	const res = await fetch(url, { method: 'GET', headers: authHeaders() });
	if (!res.ok) throw await parseErrorResponse(res);
	return res.json();
}

async function apiPost(path: string, body?: any, params?: Record<string, string | number | boolean | undefined>) {
	const searchParams = new URLSearchParams();
	if (params) {
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null && v !== '') searchParams.set(k, String(v));
		}
	}
	const qs = searchParams.toString();
	const url = qs ? `${BASE}${path}?${qs}` : `${BASE}${path}`;
	const res = await fetch(url, {
		method: 'POST',
		headers: authHeaders(),
		body: body !== undefined ? JSON.stringify(body) : undefined
	});
	if (!res.ok) throw await parseErrorResponse(res);
	return res.json();
}

async function apiPatch(path: string, body: any) {
	const res = await fetch(`${BASE}${path}`, {
		method: 'PATCH',
		headers: authHeaders(),
		body: JSON.stringify(body)
	});
	if (!res.ok) throw await parseErrorResponse(res);
	return res.json();
}

async function apiPut(path: string, body: any) {
	const res = await fetch(`${BASE}${path}`, {
		method: 'PUT',
		headers: authHeaders(),
		body: JSON.stringify(body)
	});
	if (!res.ok) throw await parseErrorResponse(res);
	return res.json();
}

async function apiDelete(path: string) {
	const res = await fetch(`${BASE}${path}`, {
		method: 'DELETE',
		headers: authHeaders()
	});
	if (!res.ok) throw await parseErrorResponse(res);
	if (res.status === 204) return;
	return res.json();
}

async function apiUpload(path: string, formData: FormData, params?: Record<string, string | number | boolean | undefined>) {
	const searchParams = new URLSearchParams();
	if (params) {
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null && v !== '') searchParams.set(k, String(v));
		}
	}
	const qs = searchParams.toString();
	const url = qs ? `${BASE}${path}?${qs}` : `${BASE}${path}`;
	const res = await fetch(url, {
		method: 'POST',
		headers: uploadAuthHeaders(),
		body: formData
	});
	if (!res.ok) throw await res.json();
	return res.json();
}

// ─── Companies ──────────────────────────────────────────────────────────────

export const getCompanies = async (params?: { active?: boolean }) =>
	apiGet('/companies', params as any);

export const createCompany = async (data: Record<string, any>) =>
	apiPost('/companies', data);

export const getCompany = async (id: number) =>
	apiGet(`/companies/${id}`);

export const getCompanyStats = async (id: number) =>
	apiGet(`/companies/${id}/stats`);

export const updateCompany = async (id: number, data: Record<string, any>) =>
	apiPatch(`/companies/${id}`, data);

export const deleteCompany = async (id: number) =>
	apiDelete(`/companies/${id}`);

export const duplicateCompany = async (id: number, data: Record<string, any>) =>
	apiPost(`/companies/${id}/duplicate`, data);

// ─── Invoice Assignment ─────────────────────────────────────────────────────

export const assignInvoiceToCompany = async (companyId: number, invoiceId: number) =>
	apiPost(`/companies/${companyId}/assign-invoice/${invoiceId}`);

export const unassignInvoice = async (companyId: number, invoiceId: number) =>
	apiPost(`/companies/${companyId}/unassign-invoice/${invoiceId}`);

export const getCompanyInvoices = async (companyId: number, params?: Record<string, any>) =>
	apiGet(`/companies/${companyId}/invoices`, params);

export const getUnassignedInvoices = async (params?: Record<string, any>) =>
	apiGet('/invoices/unassigned', params);

export const bulkAssignInvoices = async (companyId: number, invoiceIds: number[]) =>
	apiPost(`/companies/${companyId}/assign-invoices`, { invoice_ids: invoiceIds });

export const bulkUnassignInvoices = async (companyId: number, invoiceIds: number[]) =>
	apiPost(`/companies/${companyId}/unassign-invoices`, { invoice_ids: invoiceIds });

// ─── Invoice Generation ────────────────────────────────────────────────────

export const createManualInvoice = async (companyId: number, data: Record<string, any>) =>
	apiPost(`/companies/${companyId}/invoices/create`, data);

export const getNextInvoiceNumber = async (companyId: number) =>
	apiGet(`/companies/${companyId}/next-invoice-number`);

export const downloadInvoicePdf = (invoiceId: number) =>
	downloadFile(`${BASE}/invoices/${invoiceId}/pdf`, `invoice-${invoiceId}.pdf`);

// ─── Chart Templates ────────────────────────────────────────────────────────

export const getChartTemplates = async () =>
	apiGet('/chart-templates');

export const createChartTemplate = async (data: Record<string, any>) =>
	apiPost('/chart-templates', data);

export const getChartTemplate = async (id: number) =>
	apiGet(`/chart-templates/${id}`);

export const deleteChartTemplate = async (id: number) =>
	apiDelete(`/chart-templates/${id}`);

export const importChartTemplateFromExcel = async (name: string, file: File, country?: string) => {
	const formData = new FormData();
	formData.append('file', file);
	const params: Record<string, string | undefined> = { name, country };
	return apiUpload('/chart-templates/import-excel', formData, params);
};

export const downloadChartImportTemplate = () =>
	downloadFile(`${BASE}/templates/chart-import-template`, 'chart_of_accounts_template.xlsx');

export const downloadPeriodImportTemplate = () =>
	downloadFile(`${BASE}/templates/period-import-template`, 'period_template.xlsx');

export const downloadBankStatementTemplate = () =>
	downloadFile(`${BASE}/templates/bank-statement-template`, 'bank_statement_template.xlsx');

export const downloadAssetImportTemplate = () =>
	downloadFile(`${BASE}/templates/asset-import-template`, 'fixed_assets_template.xlsx');

// ─── Report Exports ─────────────────────────────────────────────────────────

const qsOf = (params: Record<string, any>): string => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) { if (v !== undefined) qs.set(k, String(v)); }
	return qs.toString();
};

export const exportTrialBalance = (params: { company_id: number; as_of?: string; period_start?: string; ytd_start?: string }) =>
	downloadFile(`${BASE}/reports/trial-balance/export?${qsOf(params)}`, `trial_balance_${params.as_of ?? 'current'}.xlsx`);

export const exportProfitLoss = (params: { company_id: number; date_from: string; date_to: string; ytd_start?: string }) =>
	downloadFile(`${BASE}/reports/profit-loss/export?${qsOf(params)}`, `profit_loss_${params.date_from}_${params.date_to}.xlsx`);

export const exportBalanceSheet = (params: { company_id: number; as_of?: string; period_start?: string }) =>
	downloadFile(`${BASE}/reports/balance-sheet/export?${qsOf(params)}`, `balance_sheet_${params.as_of ?? 'current'}.xlsx`);

export const exportGeneralLedger = (params: { company_id: number; date_from?: string; date_to?: string }) =>
	downloadFile(`${BASE}/reports/general-ledger/export?${qsOf(params)}`, `general_ledger.xlsx`);

// ─── Period Templates ───────────────────────────────────────────────────────

export const getPeriodTemplates = async () =>
	apiGet('/period-templates');

export const createPeriodTemplate = async (data: Record<string, any>) =>
	apiPost('/period-templates', data);

export const getPeriodTemplate = async (id: number) =>
	apiGet(`/period-templates/${id}`);

export const deletePeriodTemplate = async (id: number) =>
	apiDelete(`/period-templates/${id}`);

export const importPeriodTemplateFromExcel = async (name: string, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	return apiUpload('/period-templates/import-excel', formData, { name });
};

// ─── Company Excel Import ───────────────────────────────────────────────────

export const importCompanyAccountsFromExcel = async (companyId: number, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	return apiUpload(`/companies/${companyId}/import-accounts`, formData);
};

export const importCompanyPeriodsFromExcel = async (companyId: number, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	return apiUpload(`/companies/${companyId}/import-periods`, formData);
};

export const importCompanyAssetsFromExcel = async (companyId: number, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	return apiUpload(`/companies/${companyId}/import-assets`, formData);
};

// ─── Opening Balances ───────────────────────────────────────────────────────

export const getOpeningBalances = async (companyId: number) =>
	apiGet(`/companies/${companyId}/opening-balances`);

export const updateOpeningBalances = async (companyId: number, entries: any[]) =>
	apiPatch(`/companies/${companyId}/opening-balances`, { entries });

// Per-party opening-balance sub-ledger detail (AR/AP)
export const getArApAccounts = async (companyId: number) =>
	apiGet(`/companies/${companyId}/ar-ap-accounts`);

export const getOpeningBalanceDetails = async (companyId: number, accountId?: number) =>
	apiGet(`/companies/${companyId}/opening-balance-details`, accountId ? { account_id: accountId } : undefined);

export const setOpeningBalanceDetails = async (companyId: number, accountId: number, details: any[]) =>
	apiPut(`/companies/${companyId}/opening-balance-details`, { account_id: accountId, details });

export const importOpeningBalanceDetails = async (companyId: number, file: File) => {
	const formData = new FormData();
	formData.append('file', file);
	return apiUpload(`/companies/${companyId}/import-opening-balance-details`, formData);
};

export const downloadOpeningBalanceDetailTemplate = () =>
	downloadFile(
		`${BASE}/templates/opening-balance-detail-template`,
		'opening_balance_detail_template.xlsx'
	);

// ─── Accounts ───────────────────────────────────────────────────────────────

export const getAccounts = async (params?: { company_id?: number; type?: string; active?: boolean }) =>
	apiGet('/accounts', params as any);

export const createAccount = async (data: Record<string, any>, company_id?: number) =>
	apiPost('/accounts', data, { company_id });

export const getAccount = async (id: number) =>
	apiGet(`/accounts/${id}`);

export const updateAccount = async (id: number, data: Record<string, any>) =>
	apiPatch(`/accounts/${id}`, data);

export const deleteAccount = async (id: number) =>
	apiDelete(`/accounts/${id}`);

export const getAccountBalance = async (id: number, params?: { as_of?: string }) =>
	apiGet(`/accounts/${id}/balance`, params);

export const seedAccounts = async (company_id?: number) =>
	apiPost('/accounts/seed', undefined, { company_id });

// ─── Transactions ───────────────────────────────────────────────────────────

export const getTransactions = async (params?: {
	company_id?: number;
	type?: string;
	status?: string;
	date_from?: string;
	date_to?: string;
	account_id?: number;
	invoice_id?: number;
	search?: string;
	limit?: number;
	offset?: number;
}) => apiGet('/transactions', params as any);

export const createTransaction = async (data: Record<string, any>, company_id?: number, auto_pay?: boolean, bank_statement_line_id?: number) =>
	apiPost('/transactions', data, { company_id, auto_pay, bank_statement_line_id });

export const getTransaction = async (id: number) =>
	apiGet(`/transactions/${id}`);

export const updateTransaction = async (id: number, data: Record<string, any>) =>
	apiPatch(`/transactions/${id}`, data);

export const deleteTransaction = async (id: number) =>
	apiDelete(`/transactions/${id}`);

export const postTransaction = async (id: number) =>
	apiPost(`/transactions/${id}/post`);

export const bulkPostTransactions = async (params: { company_id: number; period_start?: string; period_end?: string }) =>
	apiPost('/transactions/bulk-post', undefined, params as any);

export const voidTransaction = async (id: number) =>
	apiPost(`/transactions/${id}/void`);

// ─── Payments ───────────────────────────────────────────────────────────────

export const getPayments = async (params?: {
	company_id?: number;
	direction?: string;
	invoice_id?: number;
	date_from?: string;
	date_to?: string;
	method?: string;
	limit?: number;
	offset?: number;
}) => apiGet('/payments', params as any);

/** The two lines a payment would book (settled account + bank) and where each account comes from. */
export const getPaymentPreview = async (params: {
	company_id: number;
	direction: string;
	amount?: number;
	invoice_id?: number;
	bank_statement_line_id?: number;
	payee?: string;
	payer?: string;
	reference?: string;
	debit_account_id?: number;
	credit_account_id?: number;
}) => apiGet('/payments/preview', params as any);

export const createPayment = async (data: Record<string, any>, company_id?: number, bank_statement_line_id?: number) =>
	apiPost('/payments', data, { company_id, bank_statement_line_id });

export const getPayment = async (id: number) =>
	apiGet(`/payments/${id}`);

export const deletePayment = async (id: number) =>
	apiDelete(`/payments/${id}`);

// ─── Accounting Periods ─────────────────────────────────────────────────────

export const getPeriods = async (params?: { company_id?: number }) =>
	apiGet('/periods', params as any);

export const createPeriod = async (data: Record<string, any>, company_id?: number) =>
	apiPost('/periods', data, { company_id });

export const closePeriod = async (id: number) =>
	apiPost(`/periods/${id}/close`);

export const reopenPeriod = async (id: number) =>
	apiPost(`/periods/${id}/reopen`);

// ─── Reports ────────────────────────────────────────────────────────────────

export const getGeneralLedger = async (params?: {
	company_id?: number;
	account_id?: number;
	date_from?: string;
	date_to?: string;
	currency?: string;
	limit?: number;
	offset?: number;
}) => apiGet('/reports/general-ledger', params as any);

export const getFullGeneralLedger = async (params?: {
	company_id?: number;
	date_from?: string;
	date_to?: string;
	currency?: string;
	limit?: number;
	offset?: number;
}) => apiGet('/reports/general-ledger-full', params as any);

export const getTrialBalance = async (params?: { company_id?: number; as_of?: string; period_start?: string; ytd_start?: string; currency?: string }) =>
	apiGet('/reports/trial-balance', params as any);

export const getProfitLoss = async (params?: {
	company_id?: number;
	date_from?: string;
	date_to?: string;
	ytd_start?: string;
	currency?: string;
}) => apiGet('/reports/profit-loss', params as any);

export const getBalanceSheet = async (params?: { company_id?: number; as_of?: string; period_start?: string; currency?: string }) =>
	apiGet('/reports/balance-sheet', params as any);

export const getCashFlow = async (params: { company_id: number; date_from: string; date_to: string }) =>
	apiGet('/reports/cash-flow', params as any);

export const exportCashFlow = (params: { company_id: number; date_from: string; date_to: string }) =>
	downloadFile(`${BASE}/reports/cash-flow/export?${qsOf(params)}`, `cash_flow_${params.date_from}_${params.date_to}.xlsx`);

// ─── Statutory statements (fixed-line filings; the layout follows the company's country) ──

export type StatementLine = {
	key: string;
	line_no: number | null;
	label_zh: string;
	label_en: string;
	header: boolean;
	memo: boolean;
	is_total: boolean;
	indent: number;
	codes: string[];
	beginning?: string | null;
	ending?: string | null;
	months?: Record<string, string>;
	ytd?: string | null;
};

export type StatementLayout = {
	company_id: number;
	country: string | null;
	layout: string | null;
	label: string | null;
	presentation?: 'side_by_side' | 'vertical' | null;
	fiscal_year_start_month?: number;
	statements: string[];
};

export const getStatementLayout = async (companyId: number): Promise<StatementLayout> =>
	apiGet('/reports/statements/layout', { company_id: companyId } as any);

export const getStatutoryBalanceSheet = async (params: { company_id: number; as_of?: string }) =>
	apiGet('/reports/statements/balance-sheet', params as any);

export const getStatutoryProfitLoss = async (params: { company_id: number; as_of?: string }) =>
	apiGet('/reports/statements/profit-loss', params as any);

export const exportStatutoryBalanceSheet = (params: { company_id: number; as_of?: string }) =>
	downloadFile(
		`${BASE}/reports/statements/balance-sheet/export?${qsOf(params)}`,
		`balance_sheet_${params.as_of ?? 'current'}.xlsx`
	);

export const exportStatutoryProfitLoss = (params: { company_id: number; as_of?: string }) =>
	downloadFile(
		`${BASE}/reports/statements/profit-loss/export?${qsOf(params)}`,
		`profit_loss_${params.as_of ?? 'current'}.xlsx`
	);

// ─── Configurable dashboard ─────────────────────────────────────────────────

export interface DashboardLayoutWidget {
	id: string;
	type: string;
	x: number;
	y: number;
	w: number;
	h: number;
	options?: Record<string, any>;
}

export interface DashboardLayout {
	version: number;
	widgets: DashboardLayoutWidget[];
}

export const getDashboardLayout = async (
	companyId: number
): Promise<{ company_id: number; layout: DashboardLayout | Record<string, never>; updated_at: string | null }> =>
	apiGet(`/companies/${companyId}/dashboard-layout`);

export const saveDashboardLayout = async (companyId: number, layout: DashboardLayout) =>
	apiPut(`/companies/${companyId}/dashboard-layout`, { layout });

export interface MonthlySeriesPoint {
	period: string;
	revenue: number;
	expenses: number;
	net_income: number;
}

export const getMonthlySeries = async (params: {
	company_id: number;
	months?: number;
}): Promise<{ company_id: number; currency: string; points: MonthlySeriesPoint[] }> =>
	apiGet('/reports/monthly-series', params as any);

export interface TopPartyRow {
	name: string;
	total_amount: number;
	invoice_count: number;
}

export const getTopParties = async (params: {
	company_id: number;
	direction: 'vendors' | 'customers';
	limit?: number;
}): Promise<{ company_id: number; direction: string; currency: string; rows: TopPartyRow[] }> =>
	apiGet('/reports/top-parties', params as any);

// ─── Invoice Link ───────────────────────────────────────────────────────────

export const createInvoiceEntry = async (invoiceId: number, company_id?: number, data?: Record<string, any>) =>
	apiPost(`/invoices/${invoiceId}/create-entry`, data ?? {}, { company_id });

export const getInvoiceEntries = async (invoiceId: number) =>
	apiGet(`/invoices/${invoiceId}/entries`);

// ── Categorization Rules ──────────────────────────────────────────────

export const getCategorizationRules = async (companyId: number) =>
	apiGet('/categorization-rules', { company_id: companyId });

export const createCategorizationRule = async (companyId: number, data: Record<string, any>) =>
	apiPost('/categorization-rules', data, { company_id: companyId });

export const updateCategorizationRule = async (ruleId: number, data: Record<string, any>) =>
	apiPatch(`/categorization-rules/${ruleId}`, data);

export const approveCategorizationRule = async (ruleId: number) =>
	apiPost(`/categorization-rules/${ruleId}/approve`);

export const rejectCategorizationRule = async (ruleId: number) =>
	apiPost(`/categorization-rules/${ruleId}/reject`);

export const deleteCategorizationRule = async (ruleId: number) =>
	apiDelete(`/categorization-rules/${ruleId}`);

export const applyCategorizationRule = async (ruleId: number) =>
	apiPost(`/categorization-rules/${ruleId}/apply`);

/** Book an invoice: expense/revenue account (or a split), VAT and counterparty accounts. */
export const confirmInvoiceCategory = async (
	invoiceId: number,
	data: {
		account_code: string;
		counterparty_account_code?: string;
		counterparty_account_id?: number;
		tax_account_id?: number;
		main_lines?: Array<{ account_id?: number; account_code?: string; amount: number; description?: string }>;
	}
) => apiPost(`/invoices/${invoiceId}/confirm-category`, data);

/** The entry booking this invoice would create — each line with its account source, and the gaps. */
export const getBookingPreview = async (
	invoiceId: number,
	params?: { account_code?: string; counterparty_account_id?: number; tax_account_id?: number }
) => apiGet(`/invoices/${invoiceId}/booking-preview`, params as any);

// ── Audit Trail ───────────────────────────────────────────────────────

export const getAuditTrail = async (params: {
	company_id?: number;
	entity_type?: string;
	entity_id?: number;
	limit?: number;
	offset?: number;
}) => apiGet('/audit-trail', params as any);

// ─── Invoice List (for selectors) ──────────────────────────────────────────

// ── Aging Reports ─────────────────────────────────────────────────────

export interface AgingParams {
	company_id: number;
	as_of?: string;
	q?: string;
	bucket?: string;
	min_balance?: number;
	max_balance?: number;
	sort_by?: 'name' | 'balance' | 'days' | 'bucket';
	sort_dir?: 'asc' | 'desc';
}

export const getAPAging = async (params: AgingParams) =>
	apiGet('/reports/ap-aging', params as any);

export const getARAging = async (params: AgingParams) =>
	apiGet('/reports/ar-aging', params as any);

const agingExportUrl = (path: string, params: AgingParams): string => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) {
		if (v !== undefined && v !== null && v !== '') qs.set(k, String(v));
	}
	return `${BASE}${path}?${qs}`;
};

export const exportAPAging = (params: AgingParams) =>
	downloadFileAuth(agingExportUrl('/reports/ap-aging/export', params), `ap_aging_${params.as_of ?? 'current'}.xlsx`);

export const exportARAging = (params: AgingParams) =>
	downloadFileAuth(agingExportUrl('/reports/ar-aging/export', params), `ar_aging_${params.as_of ?? 'current'}.xlsx`);

// ── Transaction ↔ Invoice links (multi-invoice) ───────────────────────

export interface LinkedInvoiceRef {
	invoice_id: number;
	invoice_number?: string | null;
	invoice_date?: string | null;
	vendor_or_client?: string | null;
	total_amount?: number | null;
	k4mi_document_id?: number | null;
	allocated_amount?: number | null;
	is_primary?: boolean;
}

export const getUnmatchedBankLines = async (
	companyId: number,
	params?: { amount?: number; tolerance?: number; limit?: number }
) => apiGet(`/companies/${companyId}/unmatched-bank-lines`, params as any);

export const getTransactionInvoices = async (txnId: number): Promise<LinkedInvoiceRef[]> =>
	apiGet(`/transactions/${txnId}/invoices`);

export const setTransactionInvoices = async (
	txnId: number,
	invoiceIds: number[],
	allocations?: Record<number, number>
): Promise<LinkedInvoiceRef[]> =>
	apiPut(`/transactions/${txnId}/invoices`, { invoice_ids: invoiceIds, allocations: allocations ?? null });

// ─── Invoice List (for selectors) ──────────────────────────────────────────

// ── Journal Templates ────────────────────────────────────────────────

export const getJournalTemplates = async (companyId: number) =>
	apiGet('/journal-templates', { company_id: companyId });

export const createJournalTemplate = async (companyId: number, data: Record<string, any>) =>
	apiPost('/journal-templates', data, { company_id: companyId });

export const deleteJournalTemplate = async (id: number) =>
	apiDelete(`/journal-templates/${id}`);

export const createTemplateFromTransaction = async (transactionId: number, name: string) =>
	apiPost('/journal-templates/from-transaction', undefined, { transaction_id: transactionId, name });

// ── Bulk Operations ─────────────────────────────────────────────────

export const bulkDeleteDrafts = async (companyId: number, transactionIds: number[]) =>
	apiDelete(`/transactions/bulk-delete?company_id=${companyId}&transaction_ids=${transactionIds.join(',')}`);

// ── Attachments ─────────────────────────────────────────────────────

export const uploadAttachment = async (file: File): Promise<{ url: string; filename: string; size: number }> => {
	const formData = new FormData();
	formData.append('file', file);
	const res = await fetch(`${BASE}/attachments/upload`, {
		method: 'POST',
		headers: uploadAuthHeaders(),
		body: formData
	});
	if (!res.ok) throw await res.json();
	return res.json();
};

// ── Recurring Templates ───────────────────────────────────────────────

export const getRecurringTemplates = async (companyId: number) =>
	apiGet('/recurring', { company_id: companyId });

export const createRecurringTemplate = async (companyId: number, data: Record<string, any>) =>
	apiPost('/recurring', data, { company_id: companyId });

export const updateRecurringTemplate = async (templateId: number, data: Record<string, any>) =>
	apiPatch(`/recurring/${templateId}`, data);

export const deleteRecurringTemplate = async (templateId: number) =>
	apiDelete(`/recurring/${templateId}`);

export const generateRecurringNow = async (templateId: number) =>
	apiPost(`/recurring/${templateId}/generate-now`);

export const previewRecurring = async (templateId: number) =>
	apiGet(`/recurring/${templateId}/preview`);

// ── Bank Reconciliation ──────────────────────────────────────────────

export const getBankAccounts = async (companyId: number) =>
	apiGet('/bank-accounts', { company_id: companyId });

export const createBankAccount = async (companyId: number, data: Record<string, any>) =>
	apiPost('/bank-accounts', data, { company_id: companyId });

export const getBankStatements = async (bankAccountId: number, params?: Record<string, any>) =>
	apiGet(`/bank-accounts/${bankAccountId}/statements`, params);

export const importBankStatement = async (bankAccountId: number, file: File, currency?: string) => {
	const formData = new FormData();
	formData.append('file', file);
	return apiUpload(`/bank-accounts/${bankAccountId}/import`, formData, { currency });
};

export const matchBankStatement = async (lineId: number, transactionId: number) =>
	apiPost(`/bank-statements/${lineId}/match`, undefined, { transaction_id: transactionId });

export const unmatchBankStatement = async (lineId: number) =>
	apiPost(`/bank-statements/${lineId}/unmatch`);

export const excludeBankStatement = async (lineId: number) =>
	apiPost(`/bank-statements/${lineId}/exclude`);

export const includeBankStatement = async (lineId: number) =>
	apiPost(`/bank-statements/${lineId}/include`);

export const autoMatchBankStatements = async (bankAccountId: number) =>
	apiPost(`/bank-accounts/${bankAccountId}/auto-match`);

// ── Match Groups (M:N bank statement ↔ invoice matching) ────────────

export const createMatchGroup = async (
	companyId: number,
	data: {
		bsl_allocations: Array<{ bank_statement_line_id: number; allocated_amount: number }>;
		transaction_allocations: Array<{ transaction_id: number; allocated_amount: number }>;
		notes?: string;
	}
) => apiPost('/match-groups', data, { company_id: companyId });

export const getMatchGroups = async (params: {
	company_id: number;
	bank_statement_line_id?: number;
	invoice_id?: number;
	status?: string;
}) => apiGet('/match-groups', params as any);

export const voidMatchGroup = async (matchGroupId: number) =>
	apiPost(`/match-groups/${matchGroupId}/void`);

export const getInvoiceList = async (params?: {
	q?: string;
	limit?: number;
	offset?: number;
	needs_review?: boolean;
	sort_by?: string;
	sort_dir?: string;
}) => {
	const searchParams = new URLSearchParams();
	if (params) {
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null && v !== '') searchParams.set(k, String(v));
		}
	}
	const qs = searchParams.toString();
	const url = qs
		? `${INVOICE_API_BASE_URL}/api/invoices?${qs}`
		: `${INVOICE_API_BASE_URL}/api/invoices`;
	const res = await fetch(url, { method: 'GET', headers: authHeaders() });
	if (!res.ok) throw await res.json();
	return res.json();
};

// ─── Exchange Rates ──────────────────────────────────────────────────

export const getExchangeRates = async (params: { company_id: number; from_currency?: string; to_currency?: string }) =>
	apiGet('/exchange-rates', params as any);

export const createExchangeRate = async (companyId: number, data: Record<string, any>) =>
	apiPost('/exchange-rates', data, { company_id: companyId });

export const deleteExchangeRate = async (rateId: number) =>
	apiDelete(`/exchange-rates/${rateId}`);

export const bulkImportExchangeRates = async (companyId: number, rates: any[]) =>
	apiPost('/exchange-rates/bulk-import', { rates }, { company_id: companyId });

export const convertCurrency = async (params: { company_id: number; from_currency: string; to_currency: string; amount: number; as_of?: string }) =>
	apiGet('/exchange-rates/convert', params as any);

// ─── Global (platform) Exchange Rates ────────────────────────────────
// The shared "main settings" rate set every company reads unless it overrides.

// Rates land one row per publish day per pair, so the list is windowed:
// newest first, capped by `limit`, narrowable by exact day, by `month`
// (YYYY-MM), or by a start/end span.
export const getGlobalExchangeRates = async (params?: {
	from_currency?: string;
	to_currency?: string;
	effective_date?: string;
	month?: string;
	start?: string;
	end?: string;
	limit?: number;
}) => apiGet('/exchange-rates/global', (params ?? {}) as any);

export const createGlobalExchangeRate = async (data: Record<string, any>) =>
	apiPost('/exchange-rates/global', data);

export const deleteGlobalExchangeRate = async (rateId: number) =>
	apiDelete(`/exchange-rates/global/${rateId}`);

export const bulkImportGlobalExchangeRates = async (rates: any[]) =>
	apiPost('/exchange-rates/global/bulk-import', { rates });

export const fetchGlobalRatesNow = async (params?: { source?: string; effective_date?: string; force?: boolean }) =>
	apiPost('/exchange-rates/global/fetch', {}, (params ?? {}) as any);

// What the rate set covers per base currency, and where it does not: days on
// file, the newest rate's age, and `gaps` — stretches long enough that an entry
// landing in one would resolve to a stale rate rather than a neighbouring
// business day. Weekends and holidays are not gaps.
export const getGlobalRateCoverage = async (params?: { start?: string; end?: string }) =>
	apiGet('/exchange-rates/global/coverage', (params ?? {}) as any);

export const backfillGlobalRates = async (params?: {
	start?: string;
	end?: string;
	source?: string;
	force?: boolean;
}) => apiPost('/exchange-rates/global/backfill', {}, (params ?? {}) as any);

// ─── Tax Declaration (Country-Aware) ─────────────────────────────────

export const getTaxConfig = async (companyId: number) =>
	apiGet('/reports/tax-config', { company_id: companyId });

export const getTaxDeclaration = async (params: {
	company_id: number;
	period_start: string;
	period_end: string;
	opening_credit?: number;
}) =>
	apiGet('/reports/tax-declaration', params as any);

/** Create a DRAFT settlement entry and attach it to the period's filing. */
export const createTaxEntry = async (
	companyId: number,
	data: {
		tax_type: string;
		period_start: string;
		period_end: string;
		entry: any;
		tax_amount?: number;
		currency?: string;
		due_date?: string;
		details?: any;
	}
) => apiPost('/reports/tax-declaration/create-entry', data, { company_id: companyId });

// ─── Tax accounts (per-company role → GL account mapping) ────────────

export const getTaxAccounts = async (companyId: number) =>
	apiGet(`/companies/${companyId}/tax-accounts`);

/** `{ role: [account_id, …] }` — an empty list clears the role back to the country default. */
export const updateTaxAccounts = async (companyId: number, mappings: Record<string, number[]>) =>
	apiPut(`/companies/${companyId}/tax-accounts`, { mappings });

export const getTaxPaymentPreview = async (filingId: number, payableAccountId?: number, assessedAmount?: number) =>
	apiGet(`/tax-filings/${filingId}/payment-preview`, { payable_account_id: payableAccountId, assessed_amount: assessedAmount });

export type CitAdjustment = { label: string; amount: number; kind: 'add' | 'deduct' };

export const getCitDeclaration = async (params: {
	company_id: number;
	period_start: string;
	period_end: string;
	prior_year_losses?: number;
	cit_already_paid?: number;
	provisional_paid?: number;
}) => apiGet('/reports/cit-declaration', params as any);

/** The profits-tax / CIT form: the accountant's adjustments travel in the body. */
export const computeCitDeclaration = async (
	companyId: number,
	data: {
		period_start: string;
		period_end: string;
		prior_year_losses?: number;
		cit_already_paid?: number;
		provisional_paid?: number;
		adjustments?: CitAdjustment[];
		two_tier?: boolean;
	}
) => apiPost('/reports/cit-declaration', data, { company_id: companyId });

/** Provisional (prepaid) income tax demanded by the tax office: DR prepaid tax / CR bank. */
export const recordProvisionalTaxPayment = async (
	companyId: number,
	data: { amount: number; bank_account_id: number; paid_date?: string; reference?: string; period_end?: string }
) => apiPost(`/companies/${companyId}/provisional-tax-payment`, data);

export const getTaxFilings = async (params: { company_id: number; tax_type?: string }) =>
	apiGet('/tax-filings', params as any);

export const saveTaxFiling = async (companyId: number, data: Record<string, any>) =>
	apiPost('/tax-filings', data, { company_id: companyId });

export const markTaxFilingPaid = async (
	filingId: number,
	data: { bank_account_id: number; paid_date?: string; payable_account_id?: number; assessed_amount?: number }
) => apiPost(`/tax-filings/${filingId}/mark-paid`, data);

export const deleteTaxFiling = async (filingId: number) => apiDelete(`/tax-filings/${filingId}`);

export const exportTaxWorksheet = (params: {
	company_id: number;
	tax_type: string;
	period_start: string;
	period_end: string;
	prior_year_losses?: number;
	cit_already_paid?: number;
	opening_credit?: number;
	provisional_paid?: number;
	adjustments?: CitAdjustment[];
	two_tier?: boolean;
}) => {
	const { adjustments, ...rest } = params;
	const q: Record<string, any> = { ...rest };
	if (adjustments && adjustments.length) q.adjustments = JSON.stringify(adjustments);
	return downloadFile(
		`${BASE}/reports/tax-worksheet/export?${qsOf(q)}`,
		`${params.tax_type}_${params.period_end}.xlsx`
	);
};

// ─── Control-account postings (reclassify, per-line human choice) ────

export const getControlAccountPostings = async (companyId: number) =>
	apiGet('/control-account-postings', { company_id: companyId, include_voided: true });

export const applyReclassification = async (
	companyId: number,
	mappings: Array<{ line_id: number; target_code: string }>,
	dryRun: boolean
) => apiPost('/control-account-postings/apply', { mappings, dry_run: dryRun }, { company_id: companyId });

// ─── Closing ─────────────────────────────────────────────────────────

export const getClosingChecklist = async (params: { company_id: number; period_start: string; period_end: string }) =>
	apiGet(`/companies/${params.company_id}/closing-checklist`, { period_start: params.period_start, period_end: params.period_end });

export const yearEndClose = async (companyId: number, data: { fiscal_year_start: string; fiscal_year_end: string }) =>
	apiPost(`/companies/${companyId}/year-end-close`, data);

// ─── Fixed Assets ────────────────────────────────────────────────────

export const getFixedAssets = async (companyId: number, asOf?: string) =>
	apiGet('/fixed-assets', { company_id: companyId, as_of: asOf });

export const createFixedAsset = async (companyId: number, data: Record<string, any>) =>
	apiPost('/fixed-assets', data, { company_id: companyId });

export const deleteFixedAsset = async (assetId: number) =>
	apiDelete(`/fixed-assets/${assetId}`);

export const generateDepreciation = async (companyId: number, periodEnd: string) =>
	apiPost('/fixed-assets/generate-depreciation', undefined, { company_id: companyId, period_end: periodEnd });

// ─── Country Tax Configs (global, editable) ──────────────────────────

export const getCountryConfigs = async () => apiGet('/country-configs');

export const getCountryConfig = async (country: string) =>
	apiGet(`/country-configs/${encodeURIComponent(country)}`);

export const updateCountryConfig = async (country: string, data: Record<string, any>) =>
	apiPut(`/country-configs/${encodeURIComponent(country)}`, data);

// ─── Fiscal Year Carryforward ────────────────────────────────────────

export const carryForwardBalances = async (companyId: number, data: { closing_date: string; opening_date: string }) =>
	apiPost(`/companies/${companyId}/carry-forward-balances`, data);

// ─── Bank Account Update ─────────────────────────────────────────────

export const updateBankAccount = async (bankAccountId: number, data: Record<string, any>) =>
	apiPatch(`/bank-accounts/${bankAccountId}`, data);

// ─── Bank Statement Line Edit ────────────────────────────────────────

export const editBankStatementLine = async (lineId: number, data: Record<string, any>) =>
	apiPatch(`/bank-statements/${lineId}/edit`, data);

// ─── Exchange Rate Template Download ─────────────────────────────────

export const downloadExchangeRateTemplate = async () => {
	const url = `${INVOICE_API_BASE_URL}/api/accounting/templates/exchange-rate-template`;
	const res = await fetch(url, { headers: authHeaders() });
	if (!res.ok) throw new Error('Download failed');
	const blob = await res.blob();
	const a = document.createElement('a');
	a.href = URL.createObjectURL(blob);
	a.download = 'exchange_rate_template.csv';
	a.click();
	URL.revokeObjectURL(a.href);
};

// ── Accounting AI (CPA-Qwen3) ──────────────────────────────────

export const getAccountingAiStatus = async () => {
	const res = await fetch(`${INVOICE_API_BASE_URL}/api/accounting/ai/status`, {
		headers: authHeaders()
	});
	if (!res.ok) throw new Error('Failed to get AI status');
	return res.json();
};

export const aiCategorizeInvoice = async (invoiceId: number) => {
	const res = await fetch(
		`${INVOICE_API_BASE_URL}/api/accounting/invoices/${invoiceId}/ai-categorize`,
		{ method: 'POST', headers: authHeaders() }
	);
	if (!res.ok) throw new Error('AI categorization failed');
	return res.json();
};

export const aiCategorizeAll = async (companyId: number) => {
	const res = await fetch(
		`${INVOICE_API_BASE_URL}/api/accounting/companies/${companyId}/ai-categorize-all`,
		{ method: 'POST', headers: authHeaders() }
	);
	if (!res.ok) throw new Error('Bulk AI categorization failed');
	return res.json();
};

export const aiValidateTransaction = async (transactionId: number) => {
	const res = await fetch(
		`${INVOICE_API_BASE_URL}/api/accounting/transactions/${transactionId}/ai-validate`,
		{ method: 'POST', headers: authHeaders() }
	);
	if (!res.ok) throw new Error('AI validation failed');
	return res.json();
};

// ─── Employees ──────────────────────────────────────────────────────────────

export const getEmployees = async (
	companyId: number,
	params?: { active?: boolean; search?: string }
) => apiGet(`/companies/${companyId}/employees`, params as any);

export const createEmployee = async (companyId: number, data: Record<string, any>) =>
	apiPost(`/companies/${companyId}/employees`, data);

export const getEmployee = async (id: number) => apiGet(`/employees/${id}`);

export const updateEmployee = async (id: number, data: Record<string, any>) =>
	apiPatch(`/employees/${id}`, data);

export const deleteEmployee = async (id: number) => apiDelete(`/employees/${id}`);

export const restoreEmployee = async (id: number) =>
	apiPost(`/employees/${id}/restore`);

export const syncEmployeesToK4mi = async (companyId: number) =>
	apiPost(`/companies/${companyId}/employees/sync-k4mi`);

// Create (or re-link) a Gnos3 portal login for an existing employee.
// Returns { linked, created, user_id, temp_password? } — temp_password only
// when a new account was created.
export const provisionEmployeeLogin = async (employeeId: number) =>
	apiPost(`/employees/${employeeId}/provision-login`);

// ─── Expense Categories ─────────────────────────────────────────────────────

export const getExpenseCategories = async (companyId: number) =>
	apiGet(`/companies/${companyId}/expense-categories`);

export const createExpenseCategory = async (companyId: number, data: Record<string, any>) =>
	apiPost(`/companies/${companyId}/expense-categories`, data);

export const updateExpenseCategory = async (id: number, data: Record<string, any>) =>
	apiPatch(`/expense-categories/${id}`, data);

export const deleteExpenseCategory = async (id: number) =>
	apiDelete(`/expense-categories/${id}`);

export const instantiateDefaultExpenseCategories = async (companyId: number) =>
	apiPost(`/companies/${companyId}/expense-categories/instantiate-defaults`);

// ─── Expense Sheets ─────────────────────────────────────────────────────────

export const getExpenseSheets = async (
	companyId: number,
	params?: {
		status?: string;
		employee_id?: number;
		date_from?: string;
		date_to?: string;
		search?: string;
	}
) => apiGet(`/companies/${companyId}/expense-sheets`, params as any);

export const getExpenseSheetCandidates = async (
	companyId: number,
	params: { employee_id: number; period_start: string; period_end: string }
) => apiGet(`/companies/${companyId}/expense-sheets/candidates`, params as any);

export const getPendingReimbursableInvoices = async (companyId: number) =>
	apiGet(`/companies/${companyId}/expense-sheets/pending`);

export const createExpenseSheet = async (companyId: number, data: Record<string, any>) =>
	apiPost(`/companies/${companyId}/expense-sheets`, data);

export const generateExpenseSheet = async (companyId: number, data: Record<string, any>) =>
	apiPost(`/companies/${companyId}/expense-sheets/generate`, data);

// Bundle approved employee-submitted expense items into a draft reimbursement sheet.
export const createExpenseSheetFromItems = async (
	companyId: number,
	data: { employee_id: number; expense_item_ids: number[]; title?: string; notes?: string }
) => apiPost(`/companies/${companyId}/expense-sheets/from-items`, data);

export const getExpenseSheet = async (id: number) => apiGet(`/expense-sheets/${id}`);

export const updateExpenseSheet = async (id: number, data: Record<string, any>) =>
	apiPatch(`/expense-sheets/${id}`, data);

export const deleteExpenseSheet = async (id: number) => apiDelete(`/expense-sheets/${id}`);

export const transitionExpenseSheet = async (
	id: number,
	data: {
		target: 'submit' | 'approve' | 'reject' | 'mark_paid';
		note?: string;
		user_info?: string;
		rejection_reason?: string;
		payment_match_group_id?: number;
	}
) => apiPost(`/expense-sheets/${id}/transitions`, data);

// The module authenticates ONLY via the Authorization header, which a plain
// <a href>/window.open cannot send — so exports must be fetched with the Bearer
// token and streamed to the browser as a blob download.
async function downloadFileAuth(url: string, fallbackName: string): Promise<void> {
	const res = await fetch(url, { method: 'GET', headers: authHeaders() });
	if (!res.ok) throw await parseErrorResponse(res);
	const blob = await res.blob();
	let filename = fallbackName;
	const cd = res.headers.get('Content-Disposition');
	const m = cd && cd.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i);
	if (m && m[1]) {
		try {
			filename = decodeURIComponent(m[1]);
		} catch {
			filename = m[1];
		}
	}
	const objUrl = URL.createObjectURL(blob);
	const a = document.createElement('a');
	a.href = objUrl;
	a.download = filename;
	document.body.appendChild(a);
	a.click();
	a.remove();
	URL.revokeObjectURL(objUrl);
}

// Fire-and-forget authenticated download: never throws (surfaces errors as a toast),
// so `on:click={() => downloadX(...)}` handlers stay safe. Prefer this over
// window.open, which cannot send the Bearer token every module route requires.
async function downloadFile(url: string, fallbackName: string): Promise<void> {
	try {
		await downloadFileAuth(url, fallbackName);
	} catch (err: any) {
		toast.error(err?.detail ?? err?.message ?? String(err));
	}
}

export const downloadExpenseSheetPdf = (id: number) =>
	downloadFileAuth(`${BASE}/expense-sheets/${id}/export/pdf`, `expense-${id}.pdf`);

export const downloadExpenseSheetExcel = (id: number) =>
	downloadFileAuth(`${BASE}/expense-sheets/${id}/export/excel`, `expense-${id}.xlsx`);

// Filtered dossier: cover summary of all sheets matching the list filters, then
// each sheet's full detail. Same filter params as getExpenseSheets.
export const downloadExpenseSheetsPdf = (
	companyId: number,
	params?: {
		status?: string;
		employee_id?: number;
		date_from?: string;
		date_to?: string;
		search?: string;
	}
) => {
	const qs = new URLSearchParams();
	if (params) {
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null && v !== '') qs.set(k, String(v));
		}
	}
	const suffix = qs.toString() ? `?${qs}` : '';
	return downloadFileAuth(
		`${BASE}/companies/${companyId}/expense-sheets/export/pdf${suffix}`,
		`expenses-${companyId}.pdf`
	);
};
