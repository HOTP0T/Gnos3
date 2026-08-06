<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import FullHeightIframe from '$lib/components/common/FullHeightIframe.svelte';
	import { FINANCES_BASE_URL } from '$lib/constants';
	import { getFinancesExchangeToken } from '$lib/apis/finances';

	const i18n = getContext('i18n');

	let iframeSrc: string | null = null;
	let error: string | null = null;
	let loading = true;

	onMount(async () => {
		try {
			const token = localStorage.getItem('token');
			if (!token) {
				throw new Error('Not authenticated');
			}
			// Fetch a fresh 5-min exchange token at mount time (too short-lived to
			// pre-bake) and hand it to the module's /sso bootstrap route.
			const { token: exchangeToken } = await getFinancesExchangeToken(token);
			iframeSrc = `${FINANCES_BASE_URL}/sso?gnos3_token=${encodeURIComponent(exchangeToken)}`;
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			loading = false;
		}
	});
</script>

<div class="w-full h-full">
	{#if iframeSrc}
		<!-- The Finances module (ledger-sync) runs on its own origin (:8003).
		     allowSameOrigin lets the embedded SPA use its own localStorage/session;
		     it does NOT grant access to the Gnos3 origin (blocked by the same-origin
		     policy). -->
		<FullHeightIframe
			src={iframeSrc}
			title="Finances"
			allowScripts={true}
			allowForms={true}
			allowSameOrigin={true}
			allowPopups={true}
			allowDownloads={true}
			iframeClassName="w-full h-full rounded-2xl border-0"
		/>
	{:else if loading}
		<div class="flex items-center justify-center h-full text-gray-500 text-sm">
			{$i18n.t('Loading Finances…')}
		</div>
	{:else}
		<div class="flex flex-col items-center justify-center h-full gap-2 text-gray-500 text-sm">
			<div>{$i18n.t('Could not load Finances.')}</div>
			{#if error}<div class="text-xs text-gray-400">{error}</div>{/if}
		</div>
	{/if}
</div>
