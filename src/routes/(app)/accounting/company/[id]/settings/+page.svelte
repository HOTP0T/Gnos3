<script lang="ts">
	import { page } from '$app/stores';
	import CompanySettings from '$lib/components/accounting/CompanySettings.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<CompanySettings {companyId} />
