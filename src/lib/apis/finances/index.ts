// Finances module (ledger-sync) — frontend half of the SSO bridge.
//
// `getFinancesExchangeToken` fetches a 5-min HS256 JWT from Gnos3's
// `/api/v1/auths/finances/exchange-token` endpoint (caller must be
// authenticated to Gnos3 with `token`, and hold `modules.finances.read`).
// The embed component then loads `${FINANCES_BASE_URL}/sso?gnos3_token=<jwt>`
// inside the module iframe; the module verifies the token and starts a session.

import { WEBUI_API_BASE_URL } from '$lib/constants';

export interface FinancesExchangeTokenResponse {
	token: string;
	expires_at: number;
}

/** Mint a short-lived (5 min) Gnos3 JWT that the Finances module SSO bridge accepts.
 *
 * Requires an existing Gnos3 session token (`token`). Throws on non-2xx —
 * callers should wrap in try/catch and surface a friendly error.
 */
export const getFinancesExchangeToken = async (
	token: string
): Promise<FinancesExchangeTokenResponse> => {
	if (!token) {
		throw new Error('not authenticated to Gnos3');
	}
	const res = await fetch(`${WEBUI_API_BASE_URL}/auths/finances/exchange-token`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	});
	if (!res.ok) {
		const body = await res.text().catch(() => '');
		throw new Error(
			`Finances exchange-token failed: HTTP ${res.status}${body ? ` — ${body.slice(0, 200)}` : ''}`
		);
	}
	return (await res.json()) as FinancesExchangeTokenResponse;
};
