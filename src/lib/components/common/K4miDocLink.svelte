<script lang="ts">
	// Phase 4 SSO bridge — a plain link into K4mi that arrives already signed in.
	//
	// This used to fetch an exchange token on click and then window.open() the
	// K4mi URL. That fails silently in two ways the frontend cannot detect: the
	// browser blocks the popup, or the page holds no localStorage bearer token
	// (Gnos3 signs users in with a session cookie and only sometimes mirrors one
	// there). Either way the hand-off never ran, no request reached the server,
	// and the user landed on K4mi's login page.
	//
	// The redirect now happens server-side at /auths/k4mi/open, so this is an
	// ordinary <a href>: no JS on the happy path, and right-click "open in new
	// tab" / middle-click work properly instead of falling back to an
	// unauthenticated URL.
	//
	// Renders nothing-but-the-slot when `docId` is null/undefined so callers can
	// `<K4miDocLink docId={inv.k4mi_document_id}>{inv.invoice_number}</K4miDocLink>`
	// without an extra `{#if}` outside.

	import { WEBUI_API_BASE_URL } from '$lib/constants';

	export let docId: number | string | null | undefined = null;
	export let title = '';
	/** Extra classes appended to the link element. Defaults to the standard
	 * blue underline-on-hover used across invoice / BC tables. */
	export let extraClass = 'text-blue-600 dark:text-blue-400 hover:underline';
	/** When true, the click also stops propagation — needed for table rows where
	 * the row itself binds an onclick (e.g., InvoiceAssignment). Navigation is
	 * NOT prevented; the link still follows its href. */
	export let stopPropagation = false;

	$: hasDoc = docId !== null && docId !== undefined && docId !== '';
	$: href = hasDoc
		? `${WEBUI_API_BASE_URL}/auths/k4mi/open?doc_id=${encodeURIComponent(String(docId))}`
		: '#';

	const onClick = (e: MouseEvent) => {
		if (stopPropagation) e.stopPropagation();
	};
</script>

{#if hasDoc}
	<a {href} target="_blank" rel="noopener noreferrer" class={extraClass} {title} on:click={onClick}>
		<slot />
	</a>
{:else}
	<slot />
{/if}
