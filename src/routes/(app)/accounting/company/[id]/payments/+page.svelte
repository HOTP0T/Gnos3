<script lang="ts">
	import { page } from '$app/stores';
	import PaymentTable from '$lib/components/accounting/PaymentTable.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<PaymentTable {companyId} />
