import { INVOICE_API_BASE_URL } from '$lib/constants';

// Per-company role assignments (Phase 3 RBAC). The invoice-processor module
// hosts the `company_membership` table and exposes admin CRUD at
// /api/accounting/company-memberships. Requires modules.accounting.admin.

const BASE = `${INVOICE_API_BASE_URL}/api/accounting/company-memberships`;

export interface CompanyMembership {
	user_id: string;
	company_id: number;
	role: 'viewer' | 'accountant' | 'admin' | 'none';
	granted_by: string | null;
	granted_at: string;
	updated_at: string;
}

function authHeaders(): Record<string, string> {
	const token = typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null;
	return {
		'Content-Type': 'application/json',
		...(token ? { Authorization: `Bearer ${token}` } : {})
	};
}

async function parseError(res: Response): Promise<any> {
	try {
		return await res.json();
	} catch {
		return { detail: `HTTP ${res.status}` };
	}
}

export const listCompanyMemberships = async (params?: {
	user_id?: string;
	company_id?: number;
}): Promise<CompanyMembership[]> => {
	const qs = new URLSearchParams();
	if (params?.user_id) qs.set('user_id', params.user_id);
	if (params?.company_id !== undefined) qs.set('company_id', String(params.company_id));
	const url = qs.toString() ? `${BASE}?${qs.toString()}` : BASE;
	const res = await fetch(url, { method: 'GET', headers: authHeaders() });
	if (!res.ok) throw await parseError(res);
	return res.json();
};

export const upsertCompanyMembership = async (
	user_id: string,
	company_id: number,
	role: CompanyMembership['role']
): Promise<CompanyMembership> => {
	const res = await fetch(BASE, {
		method: 'PUT',
		headers: authHeaders(),
		body: JSON.stringify({ user_id, company_id, role })
	});
	if (!res.ok) throw await parseError(res);
	return res.json();
};

export const deleteCompanyMembership = async (
	user_id: string,
	company_id: number
): Promise<void> => {
	const qs = new URLSearchParams({ user_id, company_id: String(company_id) });
	const res = await fetch(`${BASE}?${qs.toString()}`, {
		method: 'DELETE',
		headers: authHeaders()
	});
	if (!res.ok) throw await parseError(res);
};
