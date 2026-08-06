<script lang="ts">
	import { page } from '$app/stores';
	import TaxView from '$lib/components/accounting/TaxView.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<TaxView {companyId} />
