import { INVOICE_API_BASE_URL } from '$lib/constants';

const BASE = `${INVOICE_API_BASE_URL}/api/accounting`;

// ─── Helpers ────────────────────────────────────────────────────────────────

async function apiGet(path: string, params?: Record<string, string | number | boolean | undefined>) {
	const searchParams = new URLSearchParams();
	if (params) {
		for (const [k, v] of Object.entries(params)) {
			if (v !== undefined && v !== null && v !== '') searchParams.set(k, String(v));
		}
	}
	const qs = searchParams.toString();
	const url = qs ? `${BASE}${path}?${qs}` : `${BASE}${path}`;
	const res = await fetch(url, { method: 'GET', headers: { 'Content-Type': 'application/json' } });
	if (!res.ok) throw await res.json();
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
		headers: { 'Content-Type': 'application/json' },
		body: body !== undefined ? JSON.stringify(body) : undefined
	});
	if (!res.ok) throw await res.json();
	return res.json();
}

async function apiPatch(path: string, body: any) {
	const res = await fetch(`${BASE}${path}`, {
		method: 'PATCH',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body)
	});
	if (!res.ok) throw await res.json();
	return res.json();
}

async function apiDelete(path: string) {
	const res = await fetch(`${BASE}${path}`, {
		method: 'DELETE',
		headers: { 'Content-Type': 'application/json' }
	});
	if (!res.ok) throw await res.json();
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

export const downloadChartImportTemplate = () => {
	window.open(`${BASE}/templates/chart-import-template`, '_blank');
};

export const downloadPeriodImportTemplate = () => {
	window.open(`${BASE}/templates/period-import-template`, '_blank');
};

export const downloadBankStatementTemplate = () => {
	window.open(`${BASE}/templates/bank-statement-template`, '_blank');
};

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

// ─── Opening Balances ───────────────────────────────────────────────────────

export const getOpeningBalances = async (companyId: number) =>
	apiGet(`/companies/${companyId}/opening-balances`);

export const updateOpeningBalances = async (companyId: number, entries: any[]) =>
	apiPatch(`/companies/${companyId}/opening-balances`, { entries });

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

export const createTransaction = async (data: Record<string, any>, company_id?: number) =>
	apiPost('/transactions', data, { company_id });

export const getTransaction = async (id: number) =>
	apiGet(`/transactions/${id}`);

export const updateTransaction = async (id: number, data: Record<string, any>) =>
	apiPatch(`/transactions/${id}`, data);

export const deleteTransaction = async (id: number) =>
	apiDelete(`/transactions/${id}`);

export const postTransaction = async (id: number) =>
	apiPost(`/transactions/${id}/post`);

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

export const createPayment = async (data: Record<string, any>, company_id?: number) =>
	apiPost('/payments', data, { company_id });

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

export const deleteCategorizationRule = async (ruleId: number) =>
	apiDelete(`/categorization-rules/${ruleId}`);

export const confirmInvoiceCategory = async (invoiceId: number, accountCode: string) =>
	apiPost(`/invoices/${invoiceId}/confirm-category`, { account_code: accountCode });

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

export const getAPAging = async (params: { company_id: number; as_of?: string }) =>
	apiGet('/reports/ap-aging', params as any);

export const getARAging = async (params: { company_id: number; as_of?: string }) =>
	apiGet('/reports/ar-aging', params as any);

// ─── Invoice List (for selectors) ──────────────────────────────────────────

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

export const autoMatchBankStatements = async (bankAccountId: number) =>
	apiPost(`/bank-accounts/${bankAccountId}/auto-match`);

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
	const res = await fetch(url, { method: 'GET', headers: { 'Content-Type': 'application/json' } });
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

// ─── Tax Declaration (Country-Aware) ─────────────────────────────────

export const getTaxConfig = async (companyId: number) =>
	apiGet('/reports/tax-config', { company_id: companyId });

export const getTaxDeclaration = async (params: { company_id: number; period_start: string; period_end: string }) =>
	apiGet('/reports/tax-declaration', params as any);

export const createTaxEntry = async (companyId: number, entry: any) =>
	apiPost('/reports/tax-declaration/create-entry', { entry }, { company_id: companyId });

// ─── Closing ─────────────────────────────────────────────────────────

export const getClosingChecklist = async (params: { company_id: number; period_start: string; period_end: string }) =>
	apiGet(`/companies/${params.company_id}/closing-checklist`, { period_start: params.period_start, period_end: params.period_end });

export const yearEndClose = async (companyId: number, data: { fiscal_year_start: string; fiscal_year_end: string }) =>
	apiPost(`/companies/${companyId}/year-end-close`, data);

// ─── Fixed Assets ────────────────────────────────────────────────────

export const getFixedAssets = async (companyId: number) =>
	apiGet('/fixed-assets', { company_id: companyId });

export const createFixedAsset = async (companyId: number, data: Record<string, any>) =>
	apiPost('/fixed-assets', data, { company_id: companyId });

export const deleteFixedAsset = async (assetId: number) =>
	apiDelete(`/fixed-assets/${assetId}`);

export const generateDepreciation = async (companyId: number, periodEnd: string) =>
	apiPost('/fixed-assets/generate-depreciation', undefined, { company_id: companyId, period_end: periodEnd });

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
	const res = await fetch(url);
	if (!res.ok) throw new Error('Download failed');
	const blob = await res.blob();
	const a = document.createElement('a');
	a.href = URL.createObjectURL(blob);
	a.download = 'exchange_rate_template.csv';
	a.click();
	URL.revokeObjectURL(a.href);
};
