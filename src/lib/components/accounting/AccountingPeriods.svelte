<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';

	import { getPeriods, createPeriod, closePeriod, reopenPeriod } from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import ConfirmDialog from '$lib/components/common/ConfirmDialog.svelte';
	import Badge from '$lib/components/common/Badge.svelte';
	import Tooltip from '$lib/components/common/Tooltip.svelte';

	const i18n = getContext('i18n');

	export let companyId: number;

	let loading = true;
	let periods: any[] = [];

	// Create form
	let showCreateForm = false;
	let newName = '';
	let newStartDate = '';
	let newEndDate = '';
	let creating = false;

	// Confirm dialogs
	let showCloseConfirm = false;
	let closeTarget: any = null;
	let showReopenConfirm = false;
	let reopenTarget: any = null;

	const loadPeriods = async () => {
		loading = true;
		try {
			const res = await getPeriods({ company_id: companyId });
			periods = Array.isArray(res) ? res : res?.items ?? [];
		} catch (err) {
			toast.error(`${$i18n.t('Failed to load periods')}: ${err}`);
		}
		loading = false;
	};

	const handleCreate = async () => {
		if (!newName || !newStartDate || !newEndDate) {
			toast.error($i18n.t('Please fill in all fields'));
			return;
		}
		if (newEndDate < newStartDate) {
			toast.error($i18n.t('End date must be after start date'));
			return;
		}

		creating = true;
		try {
			await createPeriod({
				name: newName,
				start_date: newStartDate,
				end_date: newEndDate
			}, companyId);
			toast.success($i18n.t('Period created'));
			showCreateForm = false;
			newName = '';
			newStartDate = '';
			newEndDate = '';
			await loadPeriods();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to create period') + ': ' + msg);
		}
		creating = false;
	};

	const confirmClose = (period: any) => {
		closeTarget = period;
		showCloseConfirm = true;
	};

	const handleClose = async () => {
		if (!closeTarget) return;
		try {
			await closePeriod(closeTarget.id);
			toast.success($i18n.t('Period closed'));
			await loadPeriods();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to close period') + ': ' + msg);
		}
		closeTarget = null;
	};

	const confirmReopen = (period: any) => {
		reopenTarget = period;
		showReopenConfirm = true;
	};

	const handleReopen = async () => {
		if (!reopenTarget) return;
		try {
			await reopenPeriod(reopenTarget.id);
			toast.success($i18n.t('Period reopened'));
			await loadPeriods();
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Failed to reopen period') + ': ' + msg);
		}
		reopenTarget = null;
	};

	const suggestName = () => {
		if (newStartDate && !newName) {
			const d = dayjs(newStartDate);
			newName = d.format('MMMM YYYY');
		}
	};

	onMount(() => {
		loadPeriods();
	});
</script>

<ConfirmDialog
	bind:show={showCloseConfirm}
	on:confirm={handleClose}
	title={$i18n.t('Close Period')}
	message={$i18n.t(
		'Closing this period will prevent any new transactions from being posted within its date range. Continue?'
	)}
/>

<ConfirmDialog
	bind:show={showReopenConfirm}
	on:confirm={handleReopen}
	title={$i18n.t('Reopen Period')}
	message={$i18n.t(
		'Reopening this period will allow transactions to be posted within its date range again. Continue?'
	)}
/>

<div class="py-2">
	<!-- Header -->
	<div
		class="pt-0.5 pb-1 gap-1 flex flex-col md:flex-row justify-between sticky top-0 z-10 bg-white dark:bg-gray-900"
	>
		<div class="flex md:self-center text-lg font-medium px-0.5 gap-2">
			<div class="flex-shrink-0 dark:text-gray-200">{$i18n.t('Accounting Periods')}</div>
			<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
				{periods.length}
			</div>
		</div>

		<div class="flex gap-2">
			<button
				class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition"
				on:click={() => {
					showCreateForm = !showCreateForm;
				}}
			>
				{showCreateForm ? $i18n.t('Cancel') : $i18n.t('New Period')}
			</button>
		</div>
	</div>

	<!-- Create Form -->
	{#if showCreateForm}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-blue-200/50 dark:border-blue-800/30 mb-3"
		>
			<div class="text-sm font-medium dark:text-gray-200 mb-3">
				{$i18n.t('Create New Period')}
			</div>
			<div class="grid grid-cols-1 md:grid-cols-4 gap-3 items-end">
				<div>
					<label
						for="period-name"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Name')} *
					</label>
					<input
						id="period-name"
						type="text"
						bind:value={newName}
						placeholder={$i18n.t('e.g. March 2026')}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="period-start"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('Start Date')} *
					</label>
					<input
						id="period-start"
						type="date"
						bind:value={newStartDate}
						on:change={suggestName}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<div>
					<label
						for="period-end"
						class="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1"
					>
						{$i18n.t('End Date')} *
					</label>
					<input
						id="period-end"
						type="date"
						bind:value={newEndDate}
						class="w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					/>
				</div>
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition disabled:opacity-50"
					on:click={handleCreate}
					disabled={creating || !newName || !newStartDate || !newEndDate}
				>
					{creating ? $i18n.t('Creating...') : $i18n.t('Create')}
				</button>
			</div>
		</div>
	{/if}

	<!-- Periods Table -->
	{#if loading}
		<div class="flex justify-center my-10">
			<Spinner className="size-5" />
		</div>
	{:else if periods.length === 0}
		<div
			class="bg-white dark:bg-gray-900 rounded-xl p-8 border border-gray-100/30 dark:border-gray-850/30 text-center"
		>
			<div class="text-gray-400 dark:text-gray-500 text-sm mb-3">
				{$i18n.t('No accounting periods defined yet.')}
			</div>
			<div class="text-gray-400 dark:text-gray-500 text-xs">
				{$i18n.t(
					'Create periods to control when transactions can be posted. Closing a period prevents changes to transactions in that date range.'
				)}
			</div>
		</div>
	{:else}
		<div class="overflow-x-auto">
			<table class="w-full text-sm text-left text-gray-900 dark:text-gray-100">
				<thead
					class="text-xs text-gray-900 dark:text-gray-100 font-bold uppercase bg-gray-100 dark:bg-gray-800"
				>
					<tr class="border-b-[1.5px] border-gray-200 dark:border-gray-700">
						<th class="px-3 py-2">{$i18n.t('Name')}</th>
						<th class="px-3 py-2">{$i18n.t('Start Date')}</th>
						<th class="px-3 py-2">{$i18n.t('End Date')}</th>
						<th class="px-3 py-2">{$i18n.t('Status')}</th>
						<th class="px-3 py-2">{$i18n.t('Closed At')}</th>
						<th class="px-3 py-2 text-right">{$i18n.t('Actions')}</th>
					</tr>
				</thead>
				<tbody>
					{#each periods as period (period.id)}
						<tr
							class="bg-white dark:bg-gray-900 border-b border-gray-100 dark:border-gray-850 text-xs hover:bg-gray-50 dark:hover:bg-gray-850/50 transition"
						>
							<td class="px-3 py-2 font-medium dark:text-gray-200">
								{period.name}
							</td>
							<td class="px-3 py-2">
								{dayjs(period.start_date).format('YYYY-MM-DD')}
							</td>
							<td class="px-3 py-2">
								{dayjs(period.end_date).format('YYYY-MM-DD')}
							</td>
							<td class="px-3 py-2">
								{#if period.is_closed}
									<Badge type="error" content={$i18n.t('Closed')} />
								{:else}
									<Badge type="success" content={$i18n.t('Open')} />
								{/if}
							</td>
							<td class="px-3 py-2 text-gray-400">
								{period.closed_at
									? dayjs(period.closed_at).format('YYYY-MM-DD HH:mm')
									: '-'}
							</td>
							<td class="px-3 py-2 text-right">
								{#if period.is_closed}
									<Tooltip content={$i18n.t('Reopen this period')}>
										<button
											class="px-3 py-1 text-xs font-medium rounded-lg bg-yellow-50 text-yellow-700 hover:bg-yellow-100 dark:bg-yellow-900/20 dark:text-yellow-300 dark:hover:bg-yellow-900/40 transition"
											on:click={() => confirmReopen(period)}
										>
											{$i18n.t('Reopen')}
										</button>
									</Tooltip>
								{:else}
									<Tooltip content={$i18n.t('Close this period to prevent changes')}>
										<button
											class="px-3 py-1 text-xs font-medium rounded-lg bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/20 dark:text-red-300 dark:hover:bg-red-900/40 transition"
											on:click={() => confirmClose(period)}
										>
											{$i18n.t('Close')}
										</button>
									</Tooltip>
								{/if}
							</td>
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>
