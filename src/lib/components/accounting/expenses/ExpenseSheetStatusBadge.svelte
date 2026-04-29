<script lang="ts">
	export let status: string;

	const CLASS_MAP: Record<string, string> = {
		draft: 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300',
		submitted: 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300',
		approved: 'bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300',
		paid: 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300',
		rejected: 'bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300'
	};
</script>

<span
	class="inline-block text-[10px] font-medium px-2 py-0.5 rounded uppercase tracking-wide {CLASS_MAP[
		status
	] ?? CLASS_MAP.draft}"
>
	{status}
</span>
