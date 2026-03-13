import { INVOICE_API_BASE_URL } from '$lib/constants';

export const getInvoices = async (
	token: string,
	params?: {
		q?: string;
		vendor?: string;
		date_from?: string;
		date_to?: string;
		min_amount?: number;
		max_amount?: number;
		status?: string;
		needs_review?: boolean;
		tag?: string;
		notes_search?: string;
		sort_by?: string;
		sort_dir?: 'asc' | 'desc';
		limit?: number;
		offset?: number;
	}
) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (params) {
		if (params.q) searchParams.set('q', params.q);
		if (params.vendor) searchParams.set('vendor', params.vendor);
		if (params.date_from) searchParams.set('date_from', params.date_from);
		if (params.date_to) searchParams.set('date_to', params.date_to);
		if (params.min_amount !== undefined) searchParams.set('min_amount', params.min_amount.toString());
		if (params.max_amount !== undefined) searchParams.set('max_amount', params.max_amount.toString());
		if (params.status) searchParams.set('status', params.status);
		if (params.needs_review !== undefined) searchParams.set('needs_review', params.needs_review.toString());
		if (params.tag) searchParams.set('tag', params.tag);
		if (params.notes_search) searchParams.set('notes_search', params.notes_search);
		if (params.sort_by) searchParams.set('sort_by', params.sort_by);
		if (params.sort_dir) searchParams.set('sort_dir', params.sort_dir);
		if (params.limit !== undefined) searchParams.set('limit', params.limit.toString());
		if (params.offset !== undefined) searchParams.set('offset', params.offset.toString());
	}

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices?${searchParams.toString()}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((data) => {
			// Handle both old format (bare array) and new format ({invoices, total})
			if (Array.isArray(data)) {
				return { invoices: data, total: data.length };
			}
			return data;
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getInvoice = async (token: string, id: number) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/${id}`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const updateInvoice = async (token: string, id: number, data: Record<string, any>) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/${id}`, {
		method: 'PATCH',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify(data)
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const deleteInvoice = async (token: string, id: number) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/${id}`, {
		method: 'DELETE',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getVendors = async (token: string) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/vendors`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getTags = async (token: string): Promise<string[]> => {
	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/tags`, {
		method: 'GET',
		headers: { 'Content-Type': 'application/json' }
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			return [];
		});
	return res ?? [];
};

export const getSpendingSummary = async (
	token: string,
	params?: { period?: string; year?: number; vendor?: string }
) => {
	let error = null;

	const searchParams = new URLSearchParams();
	if (params) {
		if (params.period) searchParams.set('period', params.period);
		if (params.year !== undefined) searchParams.set('year', params.year.toString());
		if (params.vendor) searchParams.set('vendor', params.vendor);
	}

	const res = await fetch(
		`${INVOICE_API_BASE_URL}/api/invoices/spending-summary?${searchParams.toString()}`,
		{
			method: 'GET',
			headers: {
				'Content-Type': 'application/json'
			}
		}
	)
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getProcessingInvoices = async (token: string) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices?status=processing&limit=50`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.then((data) => {
			if (Array.isArray(data)) {
				return { invoices: data, total: data.length };
			}
			return data;
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getInvoiceStats = async (token: string) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/stats`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const reprocessInvoice = async (token: string, id: number) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/${id}/reprocess`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};

export const getNeedsReview = async (token: string) => {
	let error = null;

	const res = await fetch(`${INVOICE_API_BASE_URL}/api/invoices/needs-review`, {
		method: 'GET',
		headers: {
			'Content-Type': 'application/json'
		}
	})
		.then(async (res) => {
			if (!res.ok) throw await res.json();
			return res.json();
		})
		.catch((err) => {
			console.error(err);
			error = err.detail ?? err;
			return null;
		});

	if (error) {
		throw error;
	}

	return res;
};
