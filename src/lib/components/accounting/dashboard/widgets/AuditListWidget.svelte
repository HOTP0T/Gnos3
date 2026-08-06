<script lang="ts">
	// Recent activity — reuses the existing AuditTrail component.
	import AuditTrail from '$lib/components/accounting/AuditTrail.svelte';
	export let companyId: number;
	export let options: { limit?: number } = {};
	$: limit = options?.limit ?? 12;
</script>

<div class="h-full overflow-y-auto">
	<AuditTrail {companyId} {limit} />
</div>
