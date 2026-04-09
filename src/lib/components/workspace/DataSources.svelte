<script lang="ts">
	import dayjs from 'dayjs';
	import relativeTime from 'dayjs/plugin/relativeTime';
	dayjs.extend(relativeTime);

	import { toast } from 'svelte-sonner';
	import { onMount, getContext } from 'svelte';
	const i18n = getContext('i18n');

	import { WEBUI_NAME, user } from '$lib/stores';
	import {
		getDataConnectors,
		getConnectorTypes,
		createDataConnector,
		updateDataConnector,
		deleteDataConnector,
		syncDataConnector
	} from '$lib/apis/datasources';

	import DeleteConfirmDialog from '../common/ConfirmDialog.svelte';
	import Spinner from '../common/Spinner.svelte';
	import Tooltip from '../common/Tooltip.svelte';
	import Plus from '../icons/Plus.svelte';

	let loaded = false;
	let items: any[] = [];
	let connectorTypes: any[] = [];
	let showDeleteConfirm = false;
	let selectedItem: any = null;

	// Create/edit modal state
	let showFormModal = false;
	let editingItem: any = null;
	let formData = {
		name: '',
		description: '',
		connector_type: '',
		config: {} as Record<string, any>,
		enabled: true,
		sync_interval: 3600
	};

	const init = async () => {
		try {
			items = (await getDataConnectors(localStorage.token)) || [];
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const loadTypes = async () => {
		try {
			connectorTypes = (await getConnectorTypes(localStorage.token)) || [];
		} catch (e) {
			console.error('Failed to load connector types:', e);
		}
	};

	const openCreateModal = () => {
		editingItem = null;
		formData = {
			name: '',
			description: '',
			connector_type: connectorTypes.length > 0 ? connectorTypes[0].type : '',
			config: {},
			enabled: true,
			sync_interval: 3600
		};
		showFormModal = true;
	};

	const openEditModal = (item: any) => {
		editingItem = item;
		formData = {
			name: item.name,
			description: item.description || '',
			connector_type: item.connector_type,
			config: item.config || {},
			enabled: item.enabled,
			sync_interval: item.sync_interval
		};
		showFormModal = true;
	};

	const saveConnector = async () => {
		try {
			if (editingItem) {
				await updateDataConnector(localStorage.token, editingItem.id, {
					name: formData.name,
					description: formData.description,
					config: formData.config,
					enabled: formData.enabled,
					sync_interval: formData.sync_interval
				});
				toast.success($i18n.t('Connector updated successfully'));
			} else {
				await createDataConnector(localStorage.token, formData);
				toast.success($i18n.t('Connector created successfully'));
			}
			showFormModal = false;
			await init();
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const deleteHandler = async (item: any) => {
		try {
			await deleteDataConnector(localStorage.token, item.id);
			toast.success($i18n.t('Connector deleted successfully'));
			await init();
		} catch (e) {
			toast.error(`${e}`);
		}
	};

	const syncHandler = async (item: any) => {
		try {
			toast.info($i18n.t('Sync started for') + ` ${item.name}...`);
			const result = await syncDataConnector(localStorage.token, item.id);
			if (result?.stats) {
				const s = result.stats;
				toast.success(
					`${item.name}: +${s.added} added, ${s.updated} updated, ${s.deleted} deleted`
				);
			} else {
				toast.success($i18n.t('Sync completed'));
			}
			await init();
		} catch (e) {
			toast.error(`Sync failed: ${e}`);
			await init();
		}
	};

	const getStatusColor = (status: string | null) => {
		switch (status) {
			case 'success':
				return 'text-green-500';
			case 'running':
				return 'text-blue-500';
			case 'error':
				return 'text-red-500';
			default:
				return 'text-gray-400';
		}
	};

	const getTypeLabel = (type: string) => {
		const ct = connectorTypes.find((t) => t.type === type);
		return ct?.label || type;
	};

	const getConfigSchema = (type: string) => {
		const ct = connectorTypes.find((t) => t.type === type);
		return ct?.config_schema?.properties || {};
	};

	onMount(async () => {
		await loadTypes();
		await init();
		loaded = true;
	});
</script>

<svelte:head>
	<title>
		{$i18n.t('Data Sources')} &bull; {$WEBUI_NAME}
	</title>
</svelte:head>

{#if loaded}
	<DeleteConfirmDialog
		bind:show={showDeleteConfirm}
		on:confirm={() => {
			deleteHandler(selectedItem);
		}}
	/>

	<!-- Create/Edit Modal -->
	{#if showFormModal}
		<!-- svelte-ignore a11y-click-events-have-key-events -->
		<!-- svelte-ignore a11y-no-static-element-interactions -->
		<div
			class="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
			on:click|self={() => (showFormModal = false)}
		>
			<div
				class="bg-white dark:bg-gray-900 rounded-2xl w-full max-w-lg max-h-[80vh] overflow-y-auto p-6"
			>
				<h2 class="text-lg font-semibold mb-4">
					{editingItem ? $i18n.t('Edit Connector') : $i18n.t('New Data Source')}
				</h2>

				<div class="flex flex-col gap-3">
					<div>
						<label class="text-sm font-medium text-gray-700 dark:text-gray-300"
							>{$i18n.t('Name')}</label
						>
						<input
							bind:value={formData.name}
							class="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
							placeholder="My K4mi Documents"
						/>
					</div>

					<div>
						<label class="text-sm font-medium text-gray-700 dark:text-gray-300"
							>{$i18n.t('Description')}</label
						>
						<input
							bind:value={formData.description}
							class="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
							placeholder="Documents from the company archive"
						/>
					</div>

					{#if !editingItem}
						<div>
							<label class="text-sm font-medium text-gray-700 dark:text-gray-300"
								>{$i18n.t('Type')}</label
							>
							<select
								bind:value={formData.connector_type}
								on:change={() => (formData.config = {})}
								class="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
							>
								{#each connectorTypes as ct}
									<option value={ct.type}>{ct.label}</option>
								{/each}
							</select>
						</div>
					{/if}

					<!-- Dynamic config fields based on connector type schema -->
					{#if formData.connector_type}
						{@const schema = getConfigSchema(formData.connector_type)}
						{#each Object.entries(schema) as [key, field]}
							<div>
								<label class="text-sm font-medium text-gray-700 dark:text-gray-300">
									{field.title || key}
								</label>
								{#if field.description}
									<p class="text-xs text-gray-500 mb-1">{field.description}</p>
								{/if}
								{#if field.type === 'boolean'}
									<label class="flex items-center gap-2 mt-1">
										<input
											type="checkbox"
											checked={formData.config[key] ?? field.default ?? false}
											on:change={(e) => (formData.config[key] = e.target.checked)}
											class="rounded"
										/>
										<span class="text-sm">{$i18n.t('Enabled')}</span>
									</label>
								{:else if field.type === 'array'}
									<input
										value={(formData.config[key] || []).join(', ')}
										on:change={(e) =>
											(formData.config[key] = e.target.value
												.split(',')
												.map((s) => s.trim())
												.filter(Boolean)
												.map((s) => (isNaN(Number(s)) ? s : Number(s))))}
										class="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
										placeholder="Comma-separated values"
									/>
								{:else}
									<input
										type={field.format === 'password' ? 'password' : 'text'}
										value={formData.config[key] || ''}
										on:input={(e) => (formData.config[key] = e.target.value)}
										class="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
										placeholder={field.default || ''}
									/>
								{/if}
							</div>
						{/each}
					{/if}

					<div class="flex gap-3">
						<div class="flex-1">
							<label class="text-sm font-medium text-gray-700 dark:text-gray-300"
								>{$i18n.t('Sync Interval')}</label
							>
							<select
								bind:value={formData.sync_interval}
								class="w-full mt-1 px-3 py-2 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg text-sm"
							>
								<option value={300}>5 minutes</option>
								<option value={900}>15 minutes</option>
								<option value={1800}>30 minutes</option>
								<option value={3600}>1 hour</option>
								<option value={21600}>6 hours</option>
								<option value={86400}>24 hours</option>
							</select>
						</div>
						<div class="flex items-end">
							<label class="flex items-center gap-2 pb-2">
								<input type="checkbox" bind:checked={formData.enabled} class="rounded" />
								<span class="text-sm">{$i18n.t('Enabled')}</span>
							</label>
						</div>
					</div>
				</div>

				<div class="flex justify-end gap-2 mt-6">
					<button
						class="px-4 py-2 text-sm rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-100 dark:hover:bg-gray-800 transition"
						on:click={() => (showFormModal = false)}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						class="px-4 py-2 text-sm rounded-lg bg-black text-white dark:bg-white dark:text-black hover:opacity-90 transition font-medium"
						disabled={!formData.name || !formData.connector_type}
						on:click={saveConnector}
					>
						{editingItem ? $i18n.t('Save') : $i18n.t('Create')}
					</button>
				</div>
			</div>
		</div>
	{/if}

	<div class="flex flex-col gap-1 px-1 mt-1.5 mb-3">
		<!-- Header -->
		<div class="flex justify-between items-center">
			<div class="flex items-center md:self-center text-xl font-medium px-0.5 gap-2 shrink-0">
				<div>{$i18n.t('Data Sources')}</div>
				<div class="text-lg font-medium text-gray-500 dark:text-gray-500">
					{items.length}
				</div>
			</div>

			<div class="flex w-full justify-end gap-1.5">
				<button
					class="px-2 py-1.5 rounded-xl bg-black text-white dark:bg-white dark:text-black transition font-medium text-sm flex items-center gap-1"
					on:click={openCreateModal}
				>
					<Plus className="size-3.5" />
					{$i18n.t('New Source')}
				</button>
			</div>
		</div>

		<!-- Connector Grid -->
		{#if items.length > 0}
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-2 mt-2">
				{#each items as item}
					<div
						class="flex flex-col justify-between rounded-xl px-4 py-3 border border-gray-100 dark:border-gray-800 hover:bg-gray-50 dark:hover:bg-gray-900/50 transition"
					>
						<div class="flex justify-between items-start">
							<div class="flex-1 min-w-0">
								<div class="flex items-center gap-2">
									<div class="font-medium text-sm truncate">{item.name}</div>
									<span
										class="text-xs px-1.5 py-0.5 rounded-full bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400"
									>
										{getTypeLabel(item.connector_type)}
									</span>
									{#if !item.enabled}
										<span
											class="text-xs px-1.5 py-0.5 rounded-full bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400"
										>
											{$i18n.t('Disabled')}
										</span>
									{/if}
								</div>
								{#if item.description}
									<div class="text-xs text-gray-500 mt-0.5 truncate">
										{item.description}
									</div>
								{/if}
							</div>
						</div>

						<!-- Stats row -->
						<div class="flex items-center gap-3 mt-2 text-xs text-gray-500">
							<div class="flex items-center gap-1">
								<span class={getStatusColor(item.last_sync_status)}>&#9679;</span>
								<span>
									{#if item.last_sync_status === 'running'}
										{$i18n.t('Syncing...')}
									{:else if item.last_sync_at}
										{$i18n.t('Synced')}
										{dayjs(item.last_sync_at * 1000).fromNow()}
									{:else}
										{$i18n.t('Never synced')}
									{/if}
								</span>
							</div>

							<div>{item.doc_count} {$i18n.t('docs')}</div>

							{#if item.last_sync_status === 'error' && item.last_sync_error}
								<Tooltip content={item.last_sync_error}>
									<span class="text-red-500 cursor-help">{$i18n.t('Error')}</span>
								</Tooltip>
							{/if}
						</div>

						<!-- Action buttons -->
						<div class="flex items-center gap-1 mt-2 pt-2 border-t border-gray-100 dark:border-gray-800">
							<button
								class="px-2.5 py-1 text-xs rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition"
								disabled={item.last_sync_status === 'running'}
								on:click={() => syncHandler(item)}
							>
								{#if item.last_sync_status === 'running'}
									<Spinner className="size-3" />
								{:else}
									{$i18n.t('Sync Now')}
								{/if}
							</button>
							<button
								class="px-2.5 py-1 text-xs rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition"
								on:click={() => openEditModal(item)}
							>
								{$i18n.t('Edit')}
							</button>
							<button
								class="px-2.5 py-1 text-xs rounded-lg text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 transition"
								on:click={() => {
									selectedItem = item;
									showDeleteConfirm = true;
								}}
							>
								{$i18n.t('Delete')}
							</button>
						</div>
					</div>
				{/each}
			</div>
		{:else}
			<div class="text-center text-gray-500 text-sm mt-8 mb-4">
				<div class="text-3xl mb-2">&#128268;</div>
				<div>{$i18n.t('No data sources connected yet')}</div>
				<div class="text-xs mt-1">
					{$i18n.t('Connect K4mi, your accounting database, or other sources to chat with your documents')}
				</div>
			</div>
		{/if}
	</div>
{:else}
	<div class="flex justify-center py-8">
		<Spinner />
	</div>
{/if}
