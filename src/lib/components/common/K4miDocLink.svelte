<script lang="ts">
	// Phase 4 SSO bridge — wraps a K4mi document link with the click-time
	// exchange-token flow. Pass `docId`, render whatever you want inside the
	// default slot.
	//
	// Renders nothing-but-the-slot when `docId` is null/undefined so callers
	// can `<K4miDocLink docId={inv.k4mi_document_id}>{inv.invoice_number}</K4miDocLink>`
	// without an extra `{#if}` outside.
	//
	// Right-click "open in new tab" lands on the bare K4mi doc URL — works
	// fine once the user has SSO'd in this tab once (Django session cookie
	// is HttpOnly, 21-day Max-Age).

	import { K4MI_BASE_URL } from '$lib/constants';
	import { openK4miDoc } from '$lib/apis/k4mi';

	export let docId: number | string | null | undefined = null;
	export let title = '';
	/** Extra classes appended to the link element. Defaults to the standard
	 * blue underline-on-hover used across invoice / BC tables. */
	export let extraClass = 'text-blue-600 dark:text-blue-400 hover:underline';
	/** When true, click handler also stops propagation — needed for table
	 * rows where the row itself binds an onclick (e.g., InvoiceAssignment). */
	export let stopPropagation = false;

	$: hasDoc = docId !== null && docId !== undefined && docId !== '';
	$: fallbackHref = hasDoc ? `${K4MI_BASE_URL}/documents/${docId}/details` : '#';

	const onClick = async (e: MouseEvent) => {
		if (!hasDoc) return;
		e.preventDefault();
		if (stopPropagation) e.stopPropagation();
		const token = typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null;
		if (!token) {
			// Not signed in to Gnos3 — fall back to opening the bare K4mi URL.
			// K4mi will redirect to its own login page; user can recover from there.
			window.open(fallbackHref, '_blank', 'noopener,noreferrer');
			return;
		}
		try {
			await openK4miDoc(token, docId as number | string);
		} catch {
			// openK4miDoc already logged; fall through to bare URL so the
			// user lands somewhere rather than nowhere.
			window.open(fallbackHref, '_blank', 'noopener,noreferrer');
		}
	};
</script>

{#if hasDoc}
	<a
		href={fallbackHref}
		target="_blank"
		rel="noopener noreferrer"
		class={extraClass}
		{title}
		on:click={onClick}
	>
		<slot />
	</a>
{:else}
	<slot />
{/if}
