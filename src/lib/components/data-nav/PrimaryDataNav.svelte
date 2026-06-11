<script lang="ts">
	import { getContext, onMount } from 'svelte';
	import { page } from '$app/stores';
	import { enabledModules, ensureModulesLoaded } from '$lib/stores/modules';
	import { user, isAdmin} from '$lib/stores';

	const i18n = getContext('i18n');

	// Module-name → href + active matcher + perm key. Add new modules here as they ship.
	const TAB_FOR: Record<
		string,
		{ href: string; matches: (path: string) => boolean; permKey: string }
	> = {
		invoices: {
			href: '/invoices/dashboard',
			matches: (p) => p.startsWith('/invoices'),
			permKey: 'invoices',
		},
		business_cards: {
			href: '/business-cards/dashboard',
			matches: (p) => p.startsWith('/business-cards'),
			permKey: 'business_cards',
		},
	};

	onMount(() => {
		ensureModulesLoaded();
	});

	// RBAC: only show modules the user has read access to. Admins see everything.
	// Svelte 5: hoist store reads out of the .filter() callbacks (no $store
	// subscriptions allowed inside nested functions).
	$: _isAdminNow = $isAdmin;
	$: _userPermsNow = $user?.permissions;
	$: tabs = $enabledModules
		.filter((m) => TAB_FOR[m.name])
		.filter((m) => {
			if (_isAdminNow) return true;
			const key = TAB_FOR[m.name].permKey;
			return Boolean(_userPermsNow?.modules?.[key]?.read);
		})
		.map((m) => ({
			name: m.name,
			label: m.label || m.name,
			href: TAB_FOR[m.name].href,
			active: TAB_FOR[m.name].matches($page.url.pathname),
		}));
</script>

<div
	class="flex gap-1 scrollbar-none overflow-x-auto w-fit text-center text-sm font-medium rounded-full bg-transparent py-1"
>
	{#each tabs as tab (tab.name)}
		<a
			class="min-w-fit p-1.5 {tab.active
				? ''
				: 'text-gray-300 dark:text-gray-600 hover:text-gray-700 dark:hover:text-white'} transition"
			href={tab.href}>{$i18n.t(tab.label)}</a
		>
	{/each}
</div>
