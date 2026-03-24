<script lang="ts">
	import { page } from '$app/stores';
	import RecurringTemplates from '$lib/components/accounting/RecurringTemplates.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<div class="py-3">
	<RecurringTemplates {companyId} />
</div>
