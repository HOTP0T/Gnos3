import { BUSINESS_CARDS_API_BASE_URL } from '$lib/constants';

export interface BusinessCard {
	id: number;
	source: string;
	k4mi_document_id: number | null;
	k4mi_document_url: string | null;
	full_name: string;
	job_title: string | null;
	company_name: string | null;
	email: string | null;
	email_secondary: string | null;
	phone: string | null;
	mobile: string | null;
	fax: string | null;
	website: string | null;
	address: string | null;
	linkedin: string | null;
	notes: string | null;
	k4mi_tags: string[] | null;
	k4mi_correspondent: string | null;
	k4mi_document_type: string | null;
	k4mi_notes: unknown[] | null;
	processing_status: string;
	confidence_score: number | null;
	needs_review: boolean;
	extraction_model: string | null;
	created_at: string | null;
	updated_at: string | null;
	user_corrected: boolean;
}

export interface BusinessCardListResponse {
	business_cards: BusinessCard[];
	total: number;
}

export interface BusinessCardStats {
	total: number;
	completed: number;
	processing: number;
	needs_review: number;
	failed: number;
	recent_added_30d: number;
	by_company: { company_name: string; count: number }[];
}

function authHeaders(): Record<string, string> {
	const token = typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null;
	return {
		'Content-Type': 'application/json',
		...(token ? { Authorization: `Bearer ${token}` } : {})
	};
}

async function asJson<T>(res: Response): Promise<T> {
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: res.statusText }));
		throw err;
	}
	return res.json();
}

export const getBusinessCards = async (
	_token: string,
	params?: {
		q?: string;
		company?: string;
		tag?: string;
		status?: string;
		needs_review?: boolean;
		sort_by?: string;
		sort_dir?: 'asc' | 'desc';
		limit?: number;
		offset?: number;
	}
): Promise<BusinessCardListResponse> => {
	const search = new URLSearchParams();
	if (params) {
		if (params.q) search.set('q', params.q);
		if (params.company) search.set('company', params.company);
		if (params.tag) search.set('tag', params.tag);
		if (params.status) search.set('status', params.status);
		if (params.needs_review !== undefined)
			search.set('needs_review', params.needs_review.toString());
		if (params.sort_by) search.set('sort_by', params.sort_by);
		if (params.sort_dir) search.set('sort_dir', params.sort_dir);
		if (params.limit !== undefined) search.set('limit', params.limit.toString());
		if (params.offset !== undefined) search.set('offset', params.offset.toString());
	}
	const res = await fetch(
		`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards?${search.toString()}`,
		{ method: 'GET', headers: authHeaders() }
	);
	return asJson(res);
};

export const getBusinessCard = async (_token: string, id: number): Promise<BusinessCard> => {
	const res = await fetch(`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/${id}`, {
		method: 'GET',
		headers: authHeaders()
	});
	return asJson(res);
};

export const updateBusinessCard = async (
	_token: string,
	id: number,
	data: Partial<BusinessCard>
): Promise<BusinessCard> => {
	const res = await fetch(`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/${id}`, {
		method: 'PATCH',
		headers: authHeaders(),
		body: JSON.stringify(data)
	});
	return asJson(res);
};

export const deleteBusinessCard = async (_token: string, id: number): Promise<void> => {
	const res = await fetch(`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/${id}`, {
		method: 'DELETE',
		headers: authHeaders()
	});
	if (!res.ok) {
		const err = await res.json().catch(() => ({ detail: res.statusText }));
		throw err;
	}
};

export const getBusinessCardStats = async (_token: string): Promise<BusinessCardStats> => {
	const res = await fetch(`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/stats`, {
		method: 'GET',
		headers: authHeaders()
	});
	return asJson(res);
};

export const getBusinessCardCompanies = async (_token: string): Promise<string[]> => {
	const res = await fetch(`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/companies`, {
		method: 'GET',
		headers: authHeaders()
	});
	return asJson(res);
};

export const getBusinessCardTags = async (_token: string): Promise<string[]> => {
	const res = await fetch(`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/tags`, {
		method: 'GET',
		headers: authHeaders()
	});
	return asJson(res);
};

export const reprocessBusinessCard = async (
	_token: string,
	id: number
): Promise<{ status: string; task_id: string; document_id: number }> => {
	const res = await fetch(
		`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/${id}/reprocess`,
		{ method: 'POST', headers: authHeaders() }
	);
	return asJson(res);
};

export const syncBusinessCardFromK4mi = async (
	_token: string,
	id: number
): Promise<{ status: string; task_id: string; document_id: number }> => {
	const res = await fetch(
		`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/${id}/sync`,
		{ method: 'POST', headers: authHeaders() }
	);
	return asJson(res);
};

export const getBusinessCardPreviewUrl = (id: number): string =>
	`${BUSINESS_CARDS_API_BASE_URL}/api/business-cards/${id}/preview`;
