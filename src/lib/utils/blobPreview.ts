// Fetch an auth-gated endpoint and return a blob URL the browser can use
// in <img src=…> or <iframe src=…>.
//
// Why: iframes / <img> tags cannot send an `Authorization: Bearer …` header.
// After Phase 1 RBAC, our module preview endpoints require Bearer auth.
// We fetch ourselves (where we control the headers), wrap the response as a
// Blob, and hand back a blob URL bound to the page's origin.
//
// Caller MUST revoke the URL when done (in `onDestroy` or before swapping to
// a new one) — blobs hang on to the response body otherwise.

export interface BlobPreviewResult {
	url: string;
	contentType: string;
}

export const fetchAsBlobUrl = async (
	url: string,
	token: string | null | undefined
): Promise<BlobPreviewResult | null> => {
	if (!url || !token) return null;
	try {
		const res = await fetch(url, {
			headers: { Authorization: `Bearer ${token}` }
		});
		if (!res.ok) return null;
		const blob = await res.blob();
		return {
			url: URL.createObjectURL(blob),
			contentType: res.headers.get('content-type') ?? blob.type ?? 'application/octet-stream'
		};
	} catch {
		return null;
	}
};

export const revokeBlobUrl = (url: string | null | undefined): void => {
	if (url) URL.revokeObjectURL(url);
};
