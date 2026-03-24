<script lang="ts">
	import { page } from '$app/stores';
	import ClosingWizard from '$lib/components/accounting/ClosingWizard.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<ClosingWizard {companyId} />
