<script lang="ts">
	import { page } from '$app/stores';
	import ExpenseSheetsList from '$lib/components/accounting/expenses/ExpenseSheetsList.svelte';

	$: companyId = parseInt($page.params.id, 10);
</script>

<ExpenseSheetsList {companyId} />
