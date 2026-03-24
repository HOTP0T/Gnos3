<script lang="ts">
	import { onMount, onDestroy, getContext, createEventDispatcher } from 'svelte';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';
	import { toast } from 'svelte-sonner';

	import {
		createCompany,
		updateCompany,
		getChartTemplates,
		getPeriodTemplates
	} from '$lib/apis/accounting';

	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let company: any = null;
	export let onSave: (() => void) | null = null;

	// Form state
	let name = '';
	let country = '';
	let currency = 'USD';
	let description = '';
	let chart_template_id: number | null = null;
	let period_template_id: number | null = null;

	let submitting = false;
	let loadingTemplates = false;

	// Template options
	let chartTemplates: any[] = [];
	let periodTemplates: any[] = [];

	const COUNTRY_OPTIONS = [
		{ value: 'France', currency: 'EUR', label: 'France' },
		{ value: 'United States', currency: 'USD', label: 'United States' },
		{ value: 'United Kingdom', currency: 'GBP', label: 'United Kingdom' },
		{ value: 'Germany', currency: 'EUR', label: 'Germany' },
		{ value: 'China', currency: 'CNY', label: 'China' },
		{ value: 'Japan', currency: 'JPY', label: 'Japan' },
		{ value: 'Canada', currency: 'CAD', label: 'Canada' },
		{ value: 'Australia', currency: 'AUD', label: 'Australia' },
		{ value: 'Switzerland', currency: 'CHF', label: 'Switzerland' },
		{ value: 'Spain', currency: 'EUR', label: 'Spain' },
		{ value: 'Italy', currency: 'EUR', label: 'Italy' },
		{ value: 'Belgium', currency: 'EUR', label: 'Belgium' },
		{ value: 'Netherlands', currency: 'EUR', label: 'Netherlands' },
		{ value: 'Brazil', currency: 'BRL', label: 'Brazil' },
		{ value: 'India', currency: 'INR', label: 'India' },
		{ value: 'South Korea', currency: 'KRW', label: 'South Korea' },
		{ value: 'Mexico', currency: 'MXN', label: 'Mexico' },
		{ value: 'Singapore', currency: 'SGD', label: 'Singapore' },
		{ value: 'Hong Kong', currency: 'HKD', label: 'Hong Kong' },
		{ value: 'Morocco', currency: 'MAD', label: 'Morocco' }
	];

	const CURRENCY_OPTIONS = ['EUR', 'USD', 'GBP', 'CNY', 'JPY', 'CAD', 'AUD', 'CHF', 'BRL', 'INR', 'KRW', 'MXN', 'SGD', 'HKD', 'MAD'];

	// Custom input state for "Other" selections
	let countrySelectValue = '';
	let customCountry = '';
	let currencySelectValue = '';
	let customCurrency = '';

	$: isCustomCountry = countrySelectValue === '__other__';
	$: isCustomCurrency = currencySelectValue === '__other__';

	// Sync country from select/custom input
	$: country = isCustomCountry ? customCountry : countrySelectValue;
	// Sync currency from select/custom input
	$: currency = isCustomCurrency ? customCurrency : currencySelectValue;

	let modalElement: HTMLElement | null = null;

	$: isEditMode = company !== null && company !== undefined;

	// When country changes from dropdown, auto-set currency and suggest chart template
	const handleCountryChange = () => {
		if (countrySelectValue && countrySelectValue !== '__other__') {
			const matched = COUNTRY_OPTIONS.find((c) => c.value === countrySelectValue);
			if (matched) {
				currencySelectValue = matched.currency;
				customCurrency = '';
			}
			// Auto-select matching chart template (user can override to "None")
			const matchingTemplate = chartTemplates.find(
				(t) => t.country && t.country.toLowerCase() === countrySelectValue.toLowerCase()
			);
			chart_template_id = matchingTemplate ? matchingTemplate.id : null;
		}
	};

	$: if (show) {
		initForm();
		loadTemplates();
	}

	const initForm = () => {
		if (company) {
			name = company.name || '';
			description = company.description || '';
			chart_template_id = company.chart_template_id || null;
			period_template_id = company.period_template_id || null;

			// Country: check if it's in the known list
			const companyCountry = company.country || '';
			if (COUNTRY_OPTIONS.some((c) => c.value === companyCountry)) {
				countrySelectValue = companyCountry;
				customCountry = '';
			} else if (companyCountry) {
				countrySelectValue = '__other__';
				customCountry = companyCountry;
			} else {
				countrySelectValue = '';
				customCountry = '';
			}

			// Currency: check if it's in the known list
			const companyCurrency = company.currency || 'USD';
			if (CURRENCY_OPTIONS.includes(companyCurrency)) {
				currencySelectValue = companyCurrency;
				customCurrency = '';
			} else if (companyCurrency) {
				currencySelectValue = '__other__';
				customCurrency = companyCurrency;
			} else {
				currencySelectValue = 'USD';
				customCurrency = '';
			}
		} else {
			name = '';
			countrySelectValue = '';
			customCountry = '';
			currencySelectValue = 'USD';
			customCurrency = '';
			description = '';
			chart_template_id = null;
			period_template_id = null;
		}
	};

	const loadTemplates = async () => {
		loadingTemplates = true;
		try {
			const [chartRes, periodRes] = await Promise.all([
				getChartTemplates(),
				getPeriodTemplates()
			]);

			const chartList = Array.isArray(chartRes) ? chartRes : chartRes?.templates ?? [];
			chartTemplates = chartList;

			const periodList = Array.isArray(periodRes) ? periodRes : periodRes?.templates ?? [];
			periodTemplates = periodList;
		} catch (err: any) {
			console.error('Failed to load templates:', err);
		}
		loadingTemplates = false;
	};

	const handleKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			show = false;
		}
	};

	const handleSubmit = async () => {
		if (!name.trim()) {
			toast.error($i18n.t('Name is required'));
			return;
		}

		submitting = true;
		try {
			const payload: Record<string, any> = {
				name: name.trim(),
				country: country.trim() || null,
				currency: currency.trim() || 'USD',
				description: description.trim() || null
			};

			// Always include template IDs (even null for clearing)
			payload.chart_template_id = chart_template_id || null;
			payload.period_template_id = period_template_id || null;

			let result;
			if (isEditMode) {
				result = await updateCompany(company.id, payload);
				toast.success($i18n.t('Company updated'));
			} else {
				result = await createCompany(payload);
				toast.success($i18n.t('Company created'));
			}
			dispatch('save', result);
			if (onSave) {
				onSave();
			}
			show = false;
		} catch (err: any) {
			toast.error(err?.detail || `${err}`);
		} finally {
			submitting = false;
		}
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
			class="m-auto max-w-full w-[36rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[90dvh] overflow-y-auto shadow-3xl border border-white dark:border-gray-900"
			in:flyAndScale
			on:mousedown={(e) => {
				e.stopPropagation();
			}}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{isEditMode ? $i18n.t('Edit Company') : $i18n.t('Create Company')}
				</div>

				<form
					class="flex flex-col gap-3"
					on:submit|preventDefault={handleSubmit}
				>
					<!-- Name -->
					<div>
						<label
							for="company-name"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Name')} <span class="text-red-500">*</span>
						</label>
						<input
							id="company-name"
							type="text"
							bind:value={name}
							placeholder={$i18n.t('e.g. Acme Corp')}
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
							required
						/>
					</div>

					<!-- Country & Currency Row -->
					<div class="grid grid-cols-2 gap-3">
						<div>
							<label
								for="company-country"
								class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
							>
								{$i18n.t('Country')}
							</label>
							<select
								id="company-country"
								bind:value={countrySelectValue}
								on:change={handleCountryChange}
								class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
							>
								<option value="">{$i18n.t('Select country...')}</option>
								{#each COUNTRY_OPTIONS as opt}
									<option value={opt.value}>{opt.label}</option>
								{/each}
								<option value="__other__">{$i18n.t('Other')}</option>
							</select>
							{#if isCustomCountry}
								<input
									type="text"
									bind:value={customCountry}
									placeholder={$i18n.t('Enter country name')}
									class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition mt-1.5"
								/>
							{/if}
						</div>

						<div>
							<label
								for="company-currency"
								class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
							>
								{$i18n.t('Currency')}
							</label>
							<select
								id="company-currency"
								bind:value={currencySelectValue}
								class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
							>
								{#each CURRENCY_OPTIONS as cur}
									<option value={cur}>{cur}</option>
								{/each}
								<option value="__other__">{$i18n.t('Other')}</option>
							</select>
							{#if isCustomCurrency}
								<input
									type="text"
									bind:value={customCurrency}
									placeholder={$i18n.t('e.g. XOF, TND')}
									class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition mt-1.5"
								/>
							{/if}
						</div>
					</div>

					<!-- Description -->
					<div>
						<label
							for="company-description"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Description')}
						</label>
						<textarea
							id="company-description"
							bind:value={description}
							placeholder={$i18n.t('Optional description')}
							rows="2"
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition resize-none"
						></textarea>
					</div>

					<!-- Templates -->
					<hr class="border-gray-100/30 dark:border-gray-850/30 my-1" />

						<div class="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
							{$i18n.t('Templates')}
						</div>

						{#if loadingTemplates}
							<div class="flex items-center gap-2 text-sm text-gray-500">
								<Spinner className="size-3" />
								{$i18n.t('Loading templates...')}
							</div>
						{:else}
							<!-- Chart Template -->
							<div>
								<label
									for="chart-template"
									class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
								>
									{$i18n.t('Chart of Accounts Template')}
								</label>
								<select
									id="chart-template"
									bind:value={chart_template_id}
									class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
								>
									<option value={null}>{$i18n.t('None (start from scratch)')}</option>
									{#each chartTemplates as template}
										<option value={template.id}>
											{template.name}
											{#if template.country}({template.country}){/if}
											{#if template.is_builtin} - {$i18n.t('Built-in')}{/if}
										</option>
									{/each}
								</select>
								{#if chartTemplates.length === 0}
									<p class="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
										{$i18n.t('No chart templates available')}
									</p>
								{/if}
							</div>

							<!-- Period Template -->
							<div>
								<label
									for="period-template"
									class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
								>
									{$i18n.t('Accounting Period Template')}
								</label>
								<select
									id="period-template"
									bind:value={period_template_id}
									class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
								>
									<option value={null}>{$i18n.t('None (configure later)')}</option>
									{#each periodTemplates as template}
										<option value={template.id}>
											{template.name}
											{#if template.description} - {template.description}{/if}
										</option>
									{/each}
								</select>
								{#if periodTemplates.length === 0}
									<p class="text-xs text-gray-400 dark:text-gray-500 mt-0.5">
										{$i18n.t('No period templates available')}
									</p>
								{/if}
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
							class="text-sm bg-gray-900 hover:bg-gray-850 text-gray-100 dark:bg-gray-100 dark:hover:bg-white dark:text-gray-800 font-medium w-full py-2 rounded-3xl transition disabled:opacity-50"
							disabled={submitting}
						>
							{#if submitting}
								<div class="flex items-center justify-center gap-2">
									<Spinner className="size-3" />
									{$i18n.t('Saving...')}
								</div>
							{:else}
								{isEditMode ? $i18n.t('Update') : $i18n.t('Create')}
							{/if}
						</button>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}
