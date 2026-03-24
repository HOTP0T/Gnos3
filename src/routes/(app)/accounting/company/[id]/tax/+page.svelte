<script lang="ts">
	import { page } from '$app/stores';
	import TaxDeclaration from '$lib/components/accounting/TaxDeclaration.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<TaxDeclaration {companyId} />
