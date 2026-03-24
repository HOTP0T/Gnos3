<script lang="ts">
	import { page } from '$app/stores';
	import InvoiceAssignment from '$lib/components/accounting/InvoiceAssignment.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<InvoiceAssignment {companyId} />
