<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	import {
		getCompanies,
		createCompany,
		updateCompany,
		deleteCompany,
		duplicateCompany,
		getChartTemplates,
		getPeriodTemplates
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';
	import CompanyFormModal from '$lib/components/accounting/CompanyFormModal.svelte';

	const i18n = getContext('i18n');

	let loading = true;
	let companies: any[] = [];
	let query = '';
	let searchDebounceTimer: ReturnType<typeof setTimeout>;

	// Modal state
	let showFormModal = false;
	let editingCompany: any = null;

	// Duplicate modal state
	let showDuplicateModal = false;
	let duplicatingCompany: any = null;
	let duplicateName = '';
	let duplicating = false;

	// Deactivate confirm state
	let showDeactivateConfirm = false;
	let deactivatingCompany: any = null;
	let deactivating = false;

	$: filteredCompanies = companies.filter((c) =>
		c.name.toLowerCase().includes(query.toLowerCase())
	);

	const loadCompanies = async () => {
		loading = true;
		try {
			const res = await getCompanies();
			const list = Array.isArray(res) ? res : res?.companies ?? [];
			companies = list;
		} catch (err: any) {
			toast.error(err?.detail || `${err}`);
		}
		loading = false;
	};

	const handleAddCompany = () => {
		editingCompany = null;
		showFormModal = true;
	};

	const handleEditCompany = (company: any) => {
		editingCompany = company;
		showFormModal = true;
	};

	const handleDuplicateCompany = (company: any) => {
		duplicatingCompany = company;
		duplicateName = `${company.name} (Copy)`;
		showDuplicateModal = true;
	};

	const confirmDuplicate = async () => {
		if (!duplicateName.trim()) {
			toast.error($i18n.t('Name is required'));
			return;
		}
		duplicating = true;
		try {
			await duplicateCompany(duplicatingCompany.id, { name: duplicateName.trim() });
			toast.success($i18n.t('Company duplicated successfully'));
			showDuplicateModal = false;
			duplicatingCompany = null;
			duplicateName = '';
			await loadCompanies();
		} catch (err: any) {
			toast.error(err?.detail || `${err}`);
		}
		duplicating = false;
	};

	const handleDeactivate = (company: any) => {
		deactivatingCompany = company;
		showDeactivateConfirm = true;
	};

	const confirmDeactivate = async () => {
		deactivating = true;
		try {
			await updateCompany(deactivatingCompany.id, { is_active: false });
			toast.success($i18n.t('Company deactivated'));
			showDeactivateConfirm = false;
			deactivatingCompany = null;
			await loadCompanies();
		} catch (err: any) {
			toast.error(err?.detail || `${err}`);
		}
		deactivating = false;
	};

	const handleFormSave = async () => {
		showFormModal = false;
		editingCompany = null;
		await loadCompanies();
	};

	const handleCardClick = (company: any) => {
		goto(`/accounting/company/${company.id}/dashboard`);
	};

	const handleDuplicateKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			showDuplicateModal = false;
		}
	};

	const handleDeactivateKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			showDeactivateConfirm = false;
		}
	};

	$: if (showDuplicateModal) {
		window.addEventListener('keydown', handleDuplicateKeyDown);
	} else {
		window.removeEventListener('keydown', handleDuplicateKeyDown);
	}

	$: if (showDeactivateConfirm) {
		window.addEventListener('keydown', handleDeactivateKeyDown);
	} else {
		window.removeEventListener('keydown', handleDeactivateKeyDown);
	}

	onMount(() => {
		loadCompanies();
	});
</script>

<CompanyFormModal
	bind:show={showFormModal}
	company={editingCompany}
	on:save={handleFormSave}
/>

{#if loading}
	<div class="flex justify-center my-10">
		<Spinner className="size-5" />
	</div>
{:else}
	<div class="py-3 space-y-4">
		<!-- Header with search and add button -->
		<div
			class="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 sticky top-0 z-10 bg-white dark:bg-gray-900 pb-2"
		>
			<div class="flex items-center gap-3">
				<h2 class="text-lg font-medium dark:text-gray-200">
					{$i18n.t('Companies')}
				</h2>
				<span class="text-sm text-gray-500 dark:text-gray-400">
					{filteredCompanies.length}
				</span>
			</div>

			<div class="flex items-center gap-2 w-full md:w-auto">
				<!-- Search -->
				<div class="flex flex-1 md:flex-none items-center">
					<div class="self-center ml-1 mr-3">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							viewBox="0 0 20 20"
							fill="currentColor"
							class="w-4 h-4 text-gray-400"
						>
							<path
								fill-rule="evenodd"
								d="M9 3.5a5.5 5.5 0 100 11 5.5 5.5 0 000-11zM2 9a7 7 0 1112.452 4.391l3.328 3.329a.75.75 0 11-1.06 1.06l-3.329-3.328A7 7 0 012 9z"
								clip-rule="evenodd"
							/>
						</svg>
					</div>
					<input
						class="w-full md:w-64 text-sm pr-4 py-1 rounded-r-xl outline-hidden bg-transparent dark:text-gray-200"
						bind:value={query}
						aria-label={$i18n.t('Search')}
						placeholder={$i18n.t('Search companies...')}
					/>
				</div>

				<!-- Add Company Button -->
				<button
					class="px-4 py-2 text-sm font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition whitespace-nowrap"
					on:click={handleAddCompany}
				>
					{$i18n.t('Add Company')}
				</button>
			</div>
		</div>

		<!-- Company Cards Grid -->
		{#if filteredCompanies.length === 0}
			<div class="flex flex-col items-center justify-center py-16 text-gray-500 dark:text-gray-400">
				<svg
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
					stroke-width="1.5"
					stroke="currentColor"
					class="w-12 h-12 mb-3 text-gray-300 dark:text-gray-600"
				>
					<path
						stroke-linecap="round"
						stroke-linejoin="round"
						d="M2.25 21h19.5m-18-18v18m10.5-18v18m6-13.5V21M6.75 6.75h.75m-.75 3h.75m-.75 3h.75m3-6h.75m-.75 3h.75m-.75 3h.75M6.75 21v-3.375c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125V21M3 3h12m-.75 4.5H21m-3.75 3h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008zm0 3h.008v.008h-.008v-.008z"
					/>
				</svg>
				{#if query}
					<p class="text-sm">{$i18n.t('No companies match your search')}</p>
				{:else}
					<p class="text-sm">{$i18n.t('No companies yet')}</p>
					<p class="text-xs mt-1">{$i18n.t('Create your first company to get started')}</p>
				{/if}
			</div>
		{:else}
			<div class="grid grid-cols-1 md:grid-cols-2 gap-3">
				{#each filteredCompanies as company (company.id)}
					<!-- svelte-ignore a11y-click-events-have-key-events -->
					<!-- svelte-ignore a11y-no-static-element-interactions -->
					<div
						class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30 hover:border-blue-300 dark:hover:border-blue-700 cursor-pointer transition group {!company.is_active
							? 'opacity-60'
							: ''}"
						on:click={() => handleCardClick(company)}
					>
						<div class="flex justify-between items-start mb-3">
							<div class="flex-1 min-w-0">
								<div class="flex items-center gap-2">
									<h3
										class="text-sm font-medium dark:text-gray-200 truncate"
									>
										{company.name}
									</h3>
									{#if !company.is_active}
										<span
											class="text-[10px] font-medium px-1.5 py-0.5 rounded-full bg-gray-200 text-gray-600 dark:bg-gray-700 dark:text-gray-400"
										>
											{$i18n.t('Inactive')}
										</span>
									{/if}
								</div>
								{#if company.description}
									<p class="text-xs text-gray-500 dark:text-gray-400 mt-0.5 truncate">
										{company.description}
									</p>
								{/if}
							</div>

							<!-- Action buttons -->
							<div
								class="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition ml-2 flex-shrink-0"
							>
								<button
									class="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition"
									aria-label={$i18n.t('Edit')}
									title={$i18n.t('Edit')}
									on:click|stopPropagation={() => handleEditCompany(company)}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="w-3.5 h-3.5"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="m16.862 4.487 1.687-1.688a1.875 1.875 0 1 1 2.652 2.652L6.832 19.82a4.5 4.5 0 0 1-1.897 1.13l-2.685.8.8-2.685a4.5 4.5 0 0 1 1.13-1.897L16.863 4.487Zm0 0L19.5 7.125"
										/>
									</svg>
								</button>
								<button
									class="p-1.5 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition"
									aria-label={$i18n.t('Duplicate')}
									title={$i18n.t('Duplicate')}
									on:click|stopPropagation={() => handleDuplicateCompany(company)}
								>
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="w-3.5 h-3.5"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 0 1-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 0 1 1.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 0 0-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 0 1-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 0 0-3.375-3.375h-1.5a1.125 1.125 0 0 1-1.125-1.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H9.75"
										/>
									</svg>
								</button>
								{#if company.is_active}
									<button
										class="p-1.5 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition text-gray-500 hover:text-red-600 dark:hover:text-red-400"
										aria-label={$i18n.t('Deactivate')}
										title={$i18n.t('Deactivate')}
										on:click|stopPropagation={() => handleDeactivate(company)}
									>
										<svg
											xmlns="http://www.w3.org/2000/svg"
											fill="none"
											viewBox="0 0 24 24"
											stroke-width="1.5"
											stroke="currentColor"
											class="w-3.5 h-3.5"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												d="M18.364 18.364A9 9 0 0 0 5.636 5.636m12.728 12.728A9 9 0 0 1 5.636 5.636m12.728 12.728L5.636 5.636"
											/>
										</svg>
									</button>
								{/if}
							</div>
						</div>

						<!-- Card details -->
						<div class="flex flex-wrap gap-x-4 gap-y-1 text-xs text-gray-500 dark:text-gray-400">
							{#if company.country}
								<div class="flex items-center gap-1">
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="w-3 h-3"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M12 21a9.004 9.004 0 0 0 8.716-6.747M12 21a9.004 9.004 0 0 1-8.716-6.747M12 21c2.485 0 4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997 8.997 0 0 1 7.843 4.582M12 3a8.997 8.997 0 0 0-7.843 4.582m15.686 0A11.953 11.953 0 0 1 12 10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0 1 21 12c0 .778-.099 1.533-.284 2.253m0 0A17.919 17.919 0 0 1 12 16.5c-3.162 0-6.133-.815-8.716-2.247m0 0A9.015 9.015 0 0 1 3 12c0-1.605.42-3.113 1.157-4.418"
										/>
									</svg>
									<span>{company.country}</span>
								</div>
							{/if}
							<div class="flex items-center gap-1">
								<svg
									xmlns="http://www.w3.org/2000/svg"
									fill="none"
									viewBox="0 0 24 24"
									stroke-width="1.5"
									stroke="currentColor"
									class="w-3 h-3"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										d="M12 6v12m-3-2.818.879.659c1.171.879 3.07.879 4.242 0 1.172-.879 1.172-2.303 0-3.182C13.536 12.219 12.768 12 12 12c-.725 0-1.45-.22-2.003-.659-1.106-.879-1.106-2.303 0-3.182s2.9-.879 4.006 0l.415.33M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0z"
									/>
								</svg>
								<span>{company.currency}</span>
							</div>
							{#if company.chart_template_name}
								<div class="flex items-center gap-1">
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="w-3 h-3"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 0 1 0 3.75H5.625a1.875 1.875 0 0 1 0-3.75z"
										/>
									</svg>
									<span>{company.chart_template_name}</span>
								</div>
							{/if}
							{#if company.period_template_name}
								<div class="flex items-center gap-1">
									<svg
										xmlns="http://www.w3.org/2000/svg"
										fill="none"
										viewBox="0 0 24 24"
										stroke-width="1.5"
										stroke="currentColor"
										class="w-3 h-3"
									>
										<path
											stroke-linecap="round"
											stroke-linejoin="round"
											d="M6.75 3v2.25M17.25 3v2.25M3 18.75V7.5a2.25 2.25 0 0 1 2.25-2.25h13.5A2.25 2.25 0 0 1 21 7.5v11.25m-18 0A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75m-18 0v-7.5A2.25 2.25 0 0 1 5.25 9h13.5A2.25 2.25 0 0 1 21 11.25v7.5"
										/>
									</svg>
									<span>{company.period_template_name}</span>
								</div>
							{/if}
						</div>
					</div>
				{/each}
			</div>
		{/if}
	</div>
{/if}

<!-- Duplicate Company Modal -->
{#if showDuplicateModal}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-99999999 overflow-hidden overscroll-contain"
		on:mousedown={() => {
			showDuplicateModal = false;
		}}
	>
		<div
			class="m-auto max-w-full w-[28rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[100dvh] shadow-3xl border border-white dark:border-gray-900"
			on:mousedown|stopPropagation
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{$i18n.t('Duplicate Company')}
				</div>

				<form
					class="flex flex-col gap-3"
					on:submit|preventDefault={confirmDuplicate}
				>
					<div>
						<label
							for="duplicate-name"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('New Company Name')}
						</label>
						<input
							id="duplicate-name"
							type="text"
							bind:value={duplicateName}
							placeholder={$i18n.t('Enter company name')}
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
							required
						/>
					</div>

					<div class="text-xs text-gray-500 dark:text-gray-400">
						{$i18n.t('This will copy the chart of accounts and period configuration from')}
						<span class="font-medium">{duplicatingCompany?.name}</span>.
					</div>

					<div class="mt-3 flex justify-between gap-1.5">
						<button
							type="button"
							class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
							on:click={() => {
								showDuplicateModal = false;
							}}
						>
							{$i18n.t('Cancel')}
						</button>
						<button
							type="submit"
							class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
							disabled={duplicating}
						>
							{duplicating ? $i18n.t('Duplicating...') : $i18n.t('Duplicate')}
						</button>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}

<!-- Deactivate Confirmation Modal -->
{#if showDeactivateConfirm}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-99999999 overflow-hidden overscroll-contain"
		on:mousedown={() => {
			showDeactivateConfirm = false;
		}}
	>
		<div
			class="m-auto max-w-full w-[28rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[100dvh] shadow-3xl border border-white dark:border-gray-900"
			on:mousedown|stopPropagation
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-2">
					{$i18n.t('Deactivate Company')}
				</div>

				<p class="text-sm text-gray-600 dark:text-gray-400 mb-4">
					{$i18n.t('Are you sure you want to deactivate')}
					<span class="font-medium dark:text-gray-200">{deactivatingCompany?.name}</span>?
					{$i18n.t('This company will no longer appear in active views.')}
				</p>

				<div class="flex justify-between gap-1.5">
					<button
						type="button"
						class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
						on:click={() => {
							showDeactivateConfirm = false;
						}}
					>
						{$i18n.t('Cancel')}
					</button>
					<button
						type="button"
						class="text-sm bg-red-600 hover:bg-red-700 text-white font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
						disabled={deactivating}
						on:click={confirmDeactivate}
					>
						{deactivating ? $i18n.t('Deactivating...') : $i18n.t('Deactivate')}
					</button>
				</div>
			</div>
		</div>
	</div>
{/if}
