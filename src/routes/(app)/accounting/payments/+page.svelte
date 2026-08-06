<script lang="ts">
	// Legacy pre-multi-company route — these views are now company-scoped under
	// /accounting/company/[id]/…. Redirect to the company list.
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	onMount(() => {
		goto('/accounting/companies');
	});
</script>
