<script lang="ts">
	import { page } from '$app/stores';
	import AccountingDashboard from '$lib/components/accounting/AccountingDashboard.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<AccountingDashboard {companyId} />
