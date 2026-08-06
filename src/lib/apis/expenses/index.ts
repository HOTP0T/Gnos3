import { INVOICE_API_BASE_URL } from '$lib/constants';

// Employee expense portal — self-service API (backed by invoice-processor).
const BASE = `${INVOICE_API_BASE_URL}/api/expenses`;

// ─── Types ────────────────────────────────────────────────────────────────

export type ExpenseStatus = 'draft' | 'submitted' | 'approved' | 'rejected' | 'reimbursed';
export type PaymentMethod = 'personal' | 'company_card';

export interface ExpenseProfile {
	employee_id: number;
	company_id: number;
	company_name: string;
	code: string;
	full_name: string;
	email?: string | null;
	department?: string | null;
	default_currency: string;
}

export interface ExpenseProfileList {
	profiles: ExpenseProfile[];
	approver_company_ids: number[];
}

export interface ExpenseCategory {
	id: number;
	key: string;
	label: string;
}

export interface ExpenseItem {
	id: number;
	company_id: number;
	employee_id: number;
	employee_name?: string | null;
	expense_date: string;
	merchant: string;
	description: string;
	report_title?: string | null;
	category_id?: number | null;
	category_label?: string | null;
	currency: string;
	amount: string;
	tax_amount?: string | null;
	payment_method: PaymentMethod;
	reimbursable: boolean;
	has_receipt: boolean;
	receipt_filename?: string | null;
	status: ExpenseStatus;
	submitted_at?: string | null;
	approved_at?: string | null;
	approved_by?: string | null;
	rejected_at?: string | null;
	rejected_by?: string | null;
	reimbursed_at?: string | null;
	rejection_reason?: string | null;
	expense_sheet_id?: number | null;
	created_at: string;
	updated_at: string;
}

export interface ExpenseItemList {
	items: ExpenseItem[];
	total: number;
}

export interface ExpenseItemCreate {
	employee_id: number;
	expense_date: string;
	merchant: string;
	description: string;
	category_id?: number | null;
	currency: string;
	amount: number | string;
	tax_amount?: number | string | null;
	payment_method: PaymentMethod;
	reimbursable: boolean;
}

// ─── Helpers ──────────────────────────────────────────────────────────────

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

async function parseError(res: Response): Promise<Error> {
	let detail = `HTTP ${res.status}`;
	try {
		const body = await res.json();
		if (body?.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail);
	} catch {
		/* ignore */
	}
	return new Error(detail);
}

function qs(params?: Record<string, string | number | boolean | undefined | null>): string {
	if (!params) return '';
	const sp = new URLSearchParams();
	for (const [k, v] of Object.entries(params)) {
		if (v !== undefined && v !== null && v !== '') sp.set(k, String(v));
	}
	const s = sp.toString();
	return s ? `?${s}` : '';
}

async function get<T>(path: string, params?: Record<string, any>): Promise<T> {
	const res = await fetch(`${BASE}${path}${qs(params)}`, { headers: authHeaders() });
	if (!res.ok) throw await parseError(res);
	return res.json();
}

async function post<T>(path: string, body?: any): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		method: 'POST',
		headers: authHeaders(),
		body: body === undefined ? undefined : JSON.stringify(body)
	});
	if (!res.ok) throw await parseError(res);
	return res.json();
}

async function patch<T>(path: string, body: any): Promise<T> {
	const res = await fetch(`${BASE}${path}`, {
		method: 'PATCH',
		headers: authHeaders(),
		body: JSON.stringify(body)
	});
	if (!res.ok) throw await parseError(res);
	return res.json();
}

// ─── Endpoints ────────────────────────────────────────────────────────────

export const getMyProfiles = () => get<ExpenseProfileList>('/me');

export const getCategories = (companyId: number) =>
	get<ExpenseCategory[]>('/categories', { company_id: companyId });

export const listMyExpenses = (params?: { status?: string; company_id?: number }) =>
	get<ExpenseItemList>('', params);

export const getExpense = (id: number) => get<ExpenseItem>(`/${id}`);

export const createExpense = (data: ExpenseItemCreate) => post<ExpenseItem>('', data);

export const updateExpense = (id: number, data: Partial<ExpenseItemCreate>) =>
	patch<ExpenseItem>(`/${id}`, data);

export async function deleteExpense(id: number): Promise<void> {
	const res = await fetch(`${BASE}/${id}`, { method: 'DELETE', headers: authHeaders() });
	if (!res.ok) throw await parseError(res);
}

export const submitExpense = (id: number) => post<ExpenseItem>(`/${id}/submit`);

export async function uploadReceipt(id: number, file: File): Promise<ExpenseItem> {
	const token = getAuthToken();
	const form = new FormData();
	form.append('file', file);
	const res = await fetch(`${BASE}/${id}/receipt`, {
		method: 'POST',
		headers: token ? { Authorization: `Bearer ${token}` } : {},
		body: form
	});
	if (!res.ok) throw await parseError(res);
	return res.json();
}

// Receipt bytes need the auth header, so fetch → blob URL (same pattern as
// the invoice/BC preview iframes).
export async function receiptBlobUrl(id: number): Promise<string> {
	const res = await fetch(`${BASE}/${id}/receipt`, { headers: authHeaders() });
	if (!res.ok) throw await parseError(res);
	const blob = await res.blob();
	return URL.createObjectURL(blob);
}

// Approver surface
export const getApprovalsInbox = (companyId?: number) =>
	get<ExpenseItemList>('/approvals/inbox', { company_id: companyId });

export const approveExpense = (id: number) => post<ExpenseItem>(`/${id}/approve`);

export const rejectExpense = (id: number, reason: string) =>
	post<ExpenseItem>(`/${id}/reject`, { reason });

export const markReimbursed = (id: number) => post<ExpenseItem>(`/${id}/mark-reimbursed`);

// Accounting-module review surface: all employee-submitted expenses in a
// company (approver-gated). Supports status/employee filters.
export const listCompanyExpenses = (
	companyId: number,
	params?: { status?: string; employee_id?: number }
) => get<ExpenseItemList>(`/company/${companyId}`, params);

// Count of submitted expenses awaiting review — for the approver badge.
export const getPendingCount = (companyId: number) =>
	get<{ count: number }>(`/company/${companyId}/pending-count`);

export interface ExpenseStatBucket {
	count: number;
	by_currency: Record<string, number>;
}
export interface ExpenseStatGroup {
	category_id?: number | null;
	employee_id?: number | null;
	label?: string;
	name?: string;
	count: number;
	by_currency: Record<string, number>;
}
export interface ExpenseStats {
	base_currency: string;
	by_status: Record<string, ExpenseStatBucket>;
	by_category: ExpenseStatGroup[];
	by_employee: ExpenseStatGroup[];
}
export const getExpenseStats = (companyId: number) =>
	get<ExpenseStats>(`/company/${companyId}/stats`);
