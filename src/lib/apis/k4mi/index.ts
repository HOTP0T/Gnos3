// Phase 4 SSO bridge — frontend half of the click-through flow.
//
// `getK4miExchangeToken` fetches a 5-min HS256 JWT from Gnos3's
// `/api/v1/auths/k4mi/exchange-token` endpoint (caller must be authenticated
// to Gnos3 with `token`).
//
// `openK4miDoc` is the high-level entry point — fetches a fresh exchange
// token at click time (tokens are too short-lived to bake into <a href>),
// builds the K4mi /sso/login URL, and navigates the browser there. K4mi
// validates the token, installs a Django session, and 302s to
// `/documents/<id>/details`.

import { K4MI_BASE_URL, WEBUI_API_BASE_URL } from '$lib/constants';

export interface K4miExchangeTokenResponse {
	token: string;
	expires_at: number;
}

let lastError: Error | null = null;

/** Mint a short-lived (5 min) JWT that K4mi's SSO bridge accepts.
 *
 * Requires an existing Gnos3 session token (`token`). Throws on non-2xx —
 * callers should wrap in try/catch and surface a friendly error.
 */
export const getK4miExchangeToken = async (
	token: string
): Promise<K4miExchangeTokenResponse> => {
	if (!token) {
		throw new Error('not authenticated to Gnos3');
	}
	const res = await fetch(`${WEBUI_API_BASE_URL}/auths/k4mi/exchange-token`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json',
			Authorization: `Bearer ${token}`
		}
	});
	if (!res.ok) {
		const body = await res.text().catch(() => '');
		throw new Error(
			`K4mi exchange-token failed: HTTP ${res.status}${body ? ` — ${body.slice(0, 200)}` : ''}`
		);
	}
	return (await res.json()) as K4miExchangeTokenResponse;
};

/** Build the K4mi SSO URL for a given exchange token + next path.
 *
 * Internal helper — most callers want `openK4miDoc` instead.
 */
export const buildK4miSSOUrl = (exchangeToken: string, nextPath: string): string => {
	const safeNext = nextPath.startsWith('/') ? nextPath : `/${nextPath}`;
	const params = new URLSearchParams({ token: exchangeToken, next: safeNext });
	return `${K4MI_BASE_URL}/sso/login?${params.toString()}`;
};

export interface OpenK4miDocOpts {
	/** Open in a new tab (window.open). Default true. */
	newTab?: boolean;
	/** Path on K4mi to land on. Defaults to `/documents/{docId}/details`. */
	nextPath?: string;
}

/** High-level click handler. Fetches a token, opens K4mi at the requested
 * path. Mirrors target="_blank" semantics by default.
 *
 * Returns nothing — navigation is the side effect. Surfaces the last error
 * via `getLastK4miError()` so UI can show a toast if needed.
 */
export const openK4miDoc = async (
	gnos3Token: string,
	docId: number | string,
	opts: OpenK4miDocOpts = {}
): Promise<void> => {
	const { newTab = true, nextPath } = opts;
	const next = nextPath ?? `/documents/${docId}/details`;
	try {
		lastError = null;
		const { token } = await getK4miExchangeToken(gnos3Token);
		const url = buildK4miSSOUrl(token, next);
		if (newTab) {
			// noopener stops the new K4mi tab from reaching back into the
			// Gnos3 window (window.opener); noreferrer hides the Gnos3 URL
			// from K4mi's referrer header (defense in depth — secret is
			// already in the URL fragment, no need to also expose where it
			// came from).
			window.open(url, '_blank', 'noopener,noreferrer');
		} else {
			window.location.assign(url);
		}
	} catch (err) {
		lastError = err instanceof Error ? err : new Error(String(err));
		console.error('openK4miDoc failed:', lastError);
		throw lastError;
	}
};

export const getLastK4miError = (): Error | null => lastError;
