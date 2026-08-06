<script lang="ts">
	import { page } from '$app/stores';
	import ConfigurableDashboard from '$lib/components/accounting/dashboard/ConfigurableDashboard.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

{#key companyId}
	<ConfigurableDashboard {companyId} />
{/key}
