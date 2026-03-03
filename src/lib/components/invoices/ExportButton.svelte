<script lang="ts">
	import { getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	const i18n = getContext('i18n');

	export let invoices: any[] = [];

	let showMenu = false;

	const EXPORT_COLUMNS = [
		'id',
		'vendor_name',
		'invoice_number',
		'invoice_date',
		'due_date',
		'currency',
		'subtotal',
		'tax_amount',
		'total_amount',
		'amount_paid',
		'balance_due',
		'payment_terms',
		'po_number',
		'description',
		'processing_status',
		'confidence_score',
		'k4mi_document_id'
	];

	const exportCsv = () => {
		if (!invoices.length) {
			toast.error($i18n.t('No data to export'));
			return;
		}

		const header = EXPORT_COLUMNS.join(',');
		const rows = invoices.map((inv) =>
			EXPORT_COLUMNS.map((col) => {
				const val = inv[col];
				if (val === null || val === undefined) return '';
				const str = String(val);
				if (str.includes(',') || str.includes('"') || str.includes('\n')) {
					return `"${str.replace(/"/g, '""')}"`;
				}
				return str;
			}).join(',')
		);

		const csv = [header, ...rows].join('\n');
		downloadFile(csv, `invoices-export-${dayjs().format('YYYY-MM-DD')}.csv`, 'text/csv');
		showMenu = false;
		toast.success($i18n.t('Exported CSV'));
	};

	const exportXlsx = async () => {
		if (!invoices.length) {
			toast.error($i18n.t('No data to export'));
			return;
		}

		try {
			const XLSX = await import('xlsx');
			const wsData = [
				EXPORT_COLUMNS,
				...invoices.map((inv) => EXPORT_COLUMNS.map((col) => inv[col] ?? ''))
			];
			const ws = XLSX.utils.aoa_to_sheet(wsData);
			const wb = XLSX.utils.book_new();
			XLSX.utils.book_append_sheet(wb, ws, 'Invoices');
			XLSX.writeFile(wb, `invoices-export-${dayjs().format('YYYY-MM-DD')}.xlsx`);
			showMenu = false;
			toast.success($i18n.t('Exported Excel'));
		} catch (err) {
			toast.error(`Export failed: ${err}`);
		}
	};

	const downloadFile = (content: string, filename: string, type: string) => {
		const blob = new Blob([content], { type });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = filename;
		a.click();
		URL.revokeObjectURL(url);
	};
</script>

<div class="relative">
	<button
		class="flex text-xs items-center space-x-1 px-3 py-1.5 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-gray-200 transition"
		on:click={() => {
			showMenu = !showMenu;
		}}
	>
		<svg
			xmlns="http://www.w3.org/2000/svg"
			fill="none"
			viewBox="0 0 24 24"
			stroke-width="2"
			stroke="currentColor"
			class="size-3.5"
		>
			<path
				stroke-linecap="round"
				stroke-linejoin="round"
				d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 16.5m0 0L7.5 12m4.5 4.5V3"
			/>
		</svg>
		<span>{$i18n.t('Export')}</span>
	</button>

	{#if showMenu}
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="fixed inset-0 z-40"
			on:click={() => {
				showMenu = false;
			}}
		></div>
		<div
			class="absolute right-0 top-full mt-1 z-50 bg-white dark:bg-gray-900 rounded-xl border border-gray-100/30 dark:border-gray-850/30 shadow-lg py-1 min-w-[120px]"
		>
			<button
				class="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-50 dark:hover:bg-gray-850 transition dark:text-gray-200"
				on:click={exportCsv}
			>
				CSV
			</button>
			<button
				class="w-full text-left px-3 py-1.5 text-xs hover:bg-gray-50 dark:hover:bg-gray-850 transition dark:text-gray-200"
				on:click={exportXlsx}
			>
				Excel (XLSX)
			</button>
		</div>
	{/if}
</div>
