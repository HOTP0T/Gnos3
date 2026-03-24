<script lang="ts">
	import { page } from '$app/stores';
	import ReportsView from '$lib/components/accounting/ReportsView.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<ReportsView {companyId} />
