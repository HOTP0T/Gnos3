<script lang="ts">
	import { page } from '$app/stores';
	import TransactionTable from '$lib/components/accounting/TransactionTable.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<TransactionTable {companyId} />
