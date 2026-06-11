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

export const downloadInvoicePdf = (invoiceId: number) => {
	window.open(`${BASE}/invoices/${invoiceId}/pdf`, '_blank');
};

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

// ─── Report Exports ─────────────────────────────────────────────────────────

export const exportTrialBalance = (params: { company_id: number; as_of?: string; period_start?: string; ytd_start?: string }) => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) { if (v !== undefined) qs.set(k, String(v)); }
	window.open(`${BASE}/reports/trial-balance/export?${qs}`, '_blank');
};

export const exportProfitLoss = (params: { company_id: number; date_from: string; date_to: string; ytd_start?: string }) => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) { if (v !== undefined) qs.set(k, String(v)); }
	window.open(`${BASE}/reports/profit-loss/export?${qs}`, '_blank');
};

export const exportBalanceSheet = (params: { company_id: number; as_of?: string; period_start?: string }) => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) { if (v !== undefined) qs.set(k, String(v)); }
	window.open(`${BASE}/reports/balance-sheet/export?${qs}`, '_blank');
};

export const exportGeneralLedger = (params: { company_id: number; date_from?: string; date_to?: string }) => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) { if (v !== undefined) qs.set(k, String(v)); }
	window.open(`${BASE}/reports/general-ledger/export?${qs}`, '_blank');
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

export const exportCashFlow = (params: { company_id: number; date_from: string; date_to: string }) => {
	const qs = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) { if (v !== undefined) qs.set(k, String(v)); }
	window.open(`${BASE}/reports/cash-flow/export?${qs}`, '_blank');
};

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

export const confirmInvoiceCategory = async (invoiceId: number, accountCode: string, counterpartyAccountCode?: string) =>
	apiPost(`/invoices/${invoiceId}/confirm-category`, {
		account_code: accountCode,
		...(counterpartyAccountCode ? { counterparty_account_code: counterpartyAccountCode } : {})
	});

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

export const expenseSheetPdfUrl = (id: number) =>
	`${INVOICE_API_BASE_URL}/api/accounting/expense-sheets/${id}/export/pdf`;

export const expenseSheetExcelUrl = (id: number) =>
	`${INVOICE_API_BASE_URL}/api/accounting/expense-sheets/${id}/export/excel`;
