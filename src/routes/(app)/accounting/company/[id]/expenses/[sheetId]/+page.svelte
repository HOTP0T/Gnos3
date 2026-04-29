<script lang="ts">
	import { page } from '$app/stores';
	import ExpenseSheetDetail from '$lib/components/accounting/expenses/ExpenseSheetDetail.svelte';

	$: companyId = parseInt($page.params.id, 10);
	$: sheetId = parseInt($page.params.sheetId, 10);
</script>

<ExpenseSheetDetail {companyId} {sheetId} />
