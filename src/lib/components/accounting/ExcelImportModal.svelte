<script lang="ts">
	import { onDestroy, getContext, createEventDispatcher } from 'svelte';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';
	import { toast } from 'svelte-sonner';

	import {
		importChartTemplateFromExcel,
		importPeriodTemplateFromExcel,
		importCompanyAccountsFromExcel,
		importCompanyPeriodsFromExcel
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let type: 'chart' | 'period' = 'chart';
	export let companyId: number | undefined = undefined;
	export let onImported: (() => void) | undefined = undefined;

	// Form state
	let name = '';
	let country = '';
	let file: File | null = null;
	let fileInput: HTMLInputElement;
	let submitting = false;

	// Result state
	let result: { imported_count?: number; skipped_count?: number; errors?: string[] } | null = null;

	let modalElement: HTMLElement | null = null;

	$: if (show) {
		resetForm();
	}

	const resetForm = () => {
		name = '';
		country = '';
		file = null;
		result = null;
		submitting = false;
		if (fileInput) {
			fileInput.value = '';
		}
	};

	const handleKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			show = false;
		}
	};

	const handleFileChange = (event: Event) => {
		const target = event.target as HTMLInputElement;
		if (target.files && target.files.length > 0) {
			file = target.files[0];
		}
	};

	const handleSubmit = async () => {
		if (!file) {
			toast.error($i18n.t('Please select a file'));
			return;
		}

		if (!companyId && !name.trim()) {
			toast.error($i18n.t('Please enter a template name'));
			return;
		}

		submitting = true;
		try {
			let res: any;

			if (companyId) {
				// Import directly to company
				if (type === 'chart') {
					res = await importCompanyAccountsFromExcel(companyId, file);
				} else {
					res = await importCompanyPeriodsFromExcel(companyId, file);
				}
			} else {
				// Import as template
				if (type === 'chart') {
					res = await importChartTemplateFromExcel(
						name.trim(),
						file,
						country.trim() || undefined
					);
				} else {
					res = await importPeriodTemplateFromExcel(name.trim(), file);
				}
			}

			result = {
				imported_count: res?.imported_count ?? res?.count ?? 0,
				skipped_count: res?.skipped_count ?? 0,
				errors: res?.errors ?? []
			};

			toast.success($i18n.t('Import completed'));
			dispatch('imported');
			if (onImported) {
				onImported();
			}
		} catch (err: any) {
			const msg = err?.detail ?? err?.message ?? String(err);
			toast.error($i18n.t('Import failed') + ': ' + msg);
		}
		submitting = false;
	};

	$: if (show && modalElement) {
		window.addEventListener('keydown', handleKeyDown);
		document.body.style.overflow = 'hidden';
	}

	$: if (!show) {
		window.removeEventListener('keydown', handleKeyDown);
		document.body.style.overflow = 'unset';
	}

	onDestroy(() => {
		window.removeEventListener('keydown', handleKeyDown);
		document.body.style.overflow = 'unset';
	});
</script>

{#if show}
	<!-- svelte-ignore a11y-click-events-have-key-events -->
	<!-- svelte-ignore a11y-no-static-element-interactions -->
	<div
		bind:this={modalElement}
		class="fixed top-0 right-0 left-0 bottom-0 bg-black/60 w-full h-screen max-h-[100dvh] flex justify-center z-99999999 overflow-hidden overscroll-contain"
		in:fade={{ duration: 10 }}
		on:mousedown={() => {
			show = false;
		}}
	>
		<div
			class="m-auto max-w-full w-[32rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[100dvh] shadow-3xl border border-white dark:border-gray-900"
			in:flyAndScale
			on:mousedown={(e) => {
				e.stopPropagation();
			}}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{#if companyId}
						{type === 'chart'
							? $i18n.t('Import Accounts from Excel')
							: $i18n.t('Import Periods from Excel')}
					{:else}
						{type === 'chart'
							? $i18n.t('Import Chart Template from Excel')
							: $i18n.t('Import Period Template from Excel')}
					{/if}
				</div>

				{#if result}
					<!-- Result View -->
					<div class="flex flex-col gap-3">
						<div class="bg-green-50 dark:bg-green-900/20 rounded-xl p-4 border border-green-200/50 dark:border-green-800/30">
							<div class="text-sm font-medium text-green-800 dark:text-green-200 mb-2">
								{$i18n.t('Import Complete')}
							</div>
							<div class="text-xs text-green-700 dark:text-green-300 space-y-1">
								<div>{$i18n.t('Imported')}: {result.imported_count ?? 0}</div>
								{#if result.skipped_count}
									<div>{$i18n.t('Skipped')}: {result.skipped_count}</div>
								{/if}
							</div>
						</div>

						{#if result.errors && result.errors.length > 0}
							<div class="bg-red-50 dark:bg-red-900/20 rounded-xl p-4 border border-red-200/50 dark:border-red-800/30">
								<div class="text-sm font-medium text-red-800 dark:text-red-200 mb-2">
									{$i18n.t('Errors')}
								</div>
								<ul class="text-xs text-red-700 dark:text-red-300 space-y-0.5 list-disc list-inside max-h-32 overflow-y-auto">
									{#each result.errors as error}
										<li>{error}</li>
									{/each}
								</ul>
							</div>
						{/if}

						<div class="mt-3 flex justify-end">
							<button
								type="button"
								class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium px-6 py-2 rounded-3xl transition"
								on:click={() => {
									show = false;
								}}
							>
								{$i18n.t('Done')}
							</button>
						</div>
					</div>
				{:else}
					<!-- Upload Form -->
					<form
						class="flex flex-col gap-3"
						on:submit|preventDefault={handleSubmit}
					>
						<!-- File Input -->
						<div>
							<label
								for="import-file"
								class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
							>
								{$i18n.t('Excel File')} *
							</label>
							<input
								id="import-file"
								type="file"
								accept=".xlsx,.xls"
								bind:this={fileInput}
								on:change={handleFileChange}
								class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition file:mr-3 file:rounded-lg file:border-0 file:bg-blue-50 file:px-3 file:py-1 file:text-xs file:font-medium file:text-blue-700 dark:file:bg-blue-900/20 dark:file:text-blue-300"
							/>
						</div>

						<!-- Name (only for template import, not company import) -->
						{#if !companyId}
							<div>
								<label
									for="import-name"
									class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
								>
									{$i18n.t('Template Name')} *
								</label>
								<input
									id="import-name"
									type="text"
									bind:value={name}
									placeholder={type === 'chart'
										? $i18n.t('e.g. French PCG')
										: $i18n.t('e.g. Calendar Year 2026')}
									class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
									required
								/>
							</div>
						{/if}

						<!-- Country (only for chart template import) -->
						{#if type === 'chart' && !companyId}
							<div>
								<label
									for="import-country"
									class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
								>
									{$i18n.t('Country')}
								</label>
								<input
									id="import-country"
									type="text"
									bind:value={country}
									placeholder={$i18n.t('e.g. FR (optional)')}
									class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
								/>
							</div>
						{/if}

						<!-- Actions -->
						<div class="mt-3 flex justify-between gap-1.5">
							<button
								type="button"
								class="text-sm bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white font-medium w-full py-2 rounded-3xl transition"
								on:click={() => {
									show = false;
								}}
							>
								{$i18n.t('Cancel')}
							</button>
							<button
								type="submit"
								class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50 flex items-center justify-center gap-2"
								disabled={submitting || !file || (!companyId && !name.trim())}
							>
								{#if submitting}
									<Spinner className="size-4" />
									{$i18n.t('Importing...')}
								{:else}
									{$i18n.t('Import')}
								{/if}
							</button>
						</div>
					</form>
				{/if}
			</div>
		</div>
	</div>
{/if}
