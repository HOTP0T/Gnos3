<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { WEBUI_NAME, mobile, showSidebar, user, isAdmin } from '$lib/stores';
	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import MenuIcon from '$lib/components/icons/Sidebar.svelte';

	const i18n = getContext('i18n');
	let loaded = false;

	// RBAC: client-side guard. The server also enforces `modules.finances.read`
	// on the exchange-token endpoint — this is UX polish so non-permitted users
	// don't land on an empty page.
	onMount(async () => {
		const canAccess = $isAdmin || Boolean($user?.permissions?.modules?.finances?.read);
		if (!canAccess) {
			await goto('/');
			return;
		}
		loaded = true;
	});
</script>

<svelte:head>
	<title>{$i18n.t('Finances')} &bull; {$WEBUI_NAME}</title>
</svelte:head>

{#if loaded}
	<!-- Reserve space for the G3 sidebar so it never overlaps the embedded app
	     (mirrors the accounting layout); the iframe reflows as the area resizes. -->
	<div
		class="flex flex-col h-screen max-h-[100dvh] flex-1 transition-width duration-200 ease-in-out {$showSidebar
			? 'md:max-w-[calc(100%-var(--sidebar-width))]'
			: 'md:max-w-[calc(100%-49px)]'} w-full max-w-full"
	>
		{#if $mobile}
			<!-- Mobile-only bar: the G3 sidebar toggle. Finances has no header of its
			     own, and on phones the G3 sidebar is a drawer that needs a way to open. -->
			<nav class="px-2.5 pt-1.5">
				<div class="flex items-center gap-1">
					<div class="self-center flex flex-none items-center">
						<Tooltip
							content={$showSidebar ? $i18n.t('Close Sidebar') : $i18n.t('Open Sidebar')}
							interactive={true}
						>
							<button
								id="sidebar-toggle-button"
								class="cursor-pointer flex rounded-lg hover:bg-gray-100 dark:hover:bg-gray-850 transition"
								on:click={() => showSidebar.set(!$showSidebar)}
								aria-label={$i18n.t('Toggle Sidebar')}
							>
								<div class="self-center p-1.5"><MenuIcon /></div>
							</button>
						</Tooltip>
					</div>
					<div class="self-center text-sm font-medium">{$i18n.t('Finances')}</div>
				</div>
			</nav>
		{/if}

		<div
			class="flex-1 max-h-full overflow-hidden {$mobile ? 'px-1 pb-1' : 'p-1'}"
			id="finances-container"
		>
			<slot />
		</div>
	</div>
{/if}
