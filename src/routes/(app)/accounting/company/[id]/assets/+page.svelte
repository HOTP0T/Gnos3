<script lang="ts">
	import { page } from '$app/stores';
	import FixedAssets from '$lib/components/accounting/FixedAssets.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<FixedAssets {companyId} />
