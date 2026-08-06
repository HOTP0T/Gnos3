<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';

	import { getCountryConfigs, updateCountryConfig } from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');

	let loading = true;
	let saving = false;
	let configs: any[] = [];
	let selected = '';
	let form: any = null;
	let newCountryName = '';

	const FREQ = ['monthly', 'quarterly', 'yearly'];

	// Rates are stored as decimals (0.20) but shown/edited as percentages (20).
	const pct = (v: any) =>
		v === null || v === undefined || v === '' ? '' : +(Number(v) * 100).toFixed(6);
	const dec = (v: any) => (v === '' || v === null || v === undefined ? null : Number(v) / 100);

	function toForm(c: any) {
		return {
			country: c.country,
			tax_name: c.tax_name ?? '',
			tax_name_full: c.tax_name_full ?? '',
			currency: c.currency ?? '',
			filing_frequency: c.filing_frequency ?? 'monthly',
			vat_standard_rate: pct(c.vat_standard_rate),
			vat_reduced_rates: (c.vat_reduced_rates ?? [])
				.map((r: number) => +(Number(r) * 100).toFixed(6))
				.join(', '),
			collected_pattern: c.collected_pattern ?? '',
			deductible_pattern: c.deductible_pattern ?? '',
			settlement_payable_code: c.settlement_payable_code ?? '',
			settlement_receivable_code: c.settlement_receivable_code ?? '',
			collected_label: c.collected_label ?? '',
			deductible_label: c.deductible_label ?? '',
			payable_label: c.payable_label ?? '',
			receivable_label: c.receivable_label ?? '',
			declaration_label: c.declaration_label ?? '',
			surcharge_urban_rate: pct(c.surcharge_urban_rate),
			surcharge_education_rate: pct(c.surcharge_education_rate),
			surcharge_local_education_rate: pct(c.surcharge_local_education_rate),
			cit_rate: pct(c.cit_rate),
			cit_preferential_rate: pct(c.cit_preferential_rate),
			ir_rate: pct(c.ir_rate)
		};
	}

	function selectCountry() {
		const c = configs.find((x) => x.country === selected);
		form = c ? toForm(c) : null;
	}

	const load = async () => {
		loading = true;
		try {
			const res = await getCountryConfigs();
			configs = Array.isArray(res) ? res : [];
			if (configs.length && !configs.some((c) => c.country === selected)) {
				selected = configs[0].country;
			}
			selectCountry();
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to load country settings')}: ${err?.detail ?? err}`);
		}
		loading = false;
	};

	const addCountry = () => {
		const name = newCountryName.trim();
		if (!name) return;
		const existing = configs.find((c) => c.country.toLowerCase() === name.toLowerCase());
		if (existing) {
			selected = existing.country;
		} else {
			const blank = {
				country: name,
				tax_name: 'VAT',
				currency: 'USD',
				filing_frequency: 'monthly',
				vat_reduced_rates: []
			};
			configs = [...configs, blank].sort((a, b) => a.country.localeCompare(b.country));
			selected = name;
		}
		newCountryName = '';
		selectCountry();
	};

	const save = async () => {
		if (!form) return;
		saving = true;
		try {
			const reduced = String(form.vat_reduced_rates || '')
				.split(',')
				.map((s) => s.trim())
				.filter(Boolean)
				.map((s) => Number(s) / 100)
				.filter((n) => !isNaN(n));
			const payload = {
				tax_name: form.tax_name || form.country,
				tax_name_full: form.tax_name_full || null,
				currency: form.currency || 'USD',
				filing_frequency: form.filing_frequency || 'monthly',
				vat_standard_rate: dec(form.vat_standard_rate) ?? 0,
				vat_reduced_rates: reduced,
				collected_pattern: form.collected_pattern || null,
				deductible_pattern: form.deductible_pattern || null,
				settlement_payable_code: form.settlement_payable_code || null,
				settlement_receivable_code: form.settlement_receivable_code || null,
				collected_label: form.collected_label || null,
				deductible_label: form.deductible_label || null,
				payable_label: form.payable_label || null,
				receivable_label: form.receivable_label || null,
				declaration_label: form.declaration_label || null,
				surcharge_urban_rate: dec(form.surcharge_urban_rate) ?? 0,
				surcharge_education_rate: dec(form.surcharge_education_rate) ?? 0,
				surcharge_local_education_rate: dec(form.surcharge_local_education_rate) ?? 0,
				cit_rate: dec(form.cit_rate),
				cit_preferential_rate: dec(form.cit_preferential_rate),
				ir_rate: dec(form.ir_rate)
			};
			const updated = await updateCountryConfig(form.country, payload);
			const idx = configs.findIndex((c) => c.country === form.country);
			if (idx >= 0) configs[idx] = updated;
			else configs = [...configs, updated];
			configs = configs;
			toast.success($i18n.t('Country settings saved'));
		} catch (err: any) {
			toast.error(`${$i18n.t('Failed to save country settings')}: ${err?.detail ?? err}`);
		}
		saving = false;
	};

	onMount(load);

	const inputCls =
		'w-full text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition';
	const labelCls = 'block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1';
</script>

<div class="py-2">
	<div class="text-xs text-gray-400 dark:text-gray-500 px-0.5 mb-3">
		{$i18n.t(
			'Per-country tax parameters. These drive the tax module (VAT/IS/IR) and override the built-in defaults. Rates are entered as percentages.'
		)}
	</div>

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{:else}
		<!-- Country picker + add + save -->
		<div class="flex flex-col md:flex-row md:items-end gap-2 mb-4">
			<div class="flex-1">
				<label for="cc-country" class={labelCls}>{$i18n.t('Country')}</label>
				<select id="cc-country" bind:value={selected} on:change={selectCountry} class={inputCls}>
					{#each configs as c}
						<option value={c.country}>{c.country}</option>
					{/each}
				</select>
			</div>
			<div class="flex items-end gap-2">
				<input
					type="text"
					bind:value={newCountryName}
					placeholder={$i18n.t('Add country…')}
					class="text-sm rounded-lg px-3 py-2 bg-gray-50 dark:bg-gray-850 dark:text-gray-200 border border-gray-200 dark:border-gray-800 outline-hidden focus:border-blue-500 transition"
					on:keydown={(e) => e.key === 'Enter' && addCountry()}
				/>
				<button
					class="px-3 py-2 text-sm font-medium rounded-lg bg-gray-100 hover:bg-gray-200 text-gray-800 dark:bg-gray-850 dark:hover:bg-gray-800 dark:text-white transition"
					on:click={addCountry}
				>
					{$i18n.t('Add')}
				</button>
			</div>
		</div>

		{#if form}
			<div class="space-y-3">
				<!-- General -->
				<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30">
					<div class="text-sm font-medium dark:text-gray-200 mb-3">{$i18n.t('General')}</div>
					<div class="grid grid-cols-1 md:grid-cols-4 gap-3">
						<div>
							<label class={labelCls} for="cc-taxname">{$i18n.t('Tax name')}</label>
							<input id="cc-taxname" class={inputCls} bind:value={form.tax_name} placeholder="TVA / VAT / 增值税" />
						</div>
						<div class="md:col-span-2">
							<label class={labelCls} for="cc-taxnamefull">{$i18n.t('Full name')}</label>
							<input id="cc-taxnamefull" class={inputCls} bind:value={form.tax_name_full} />
						</div>
						<div>
							<label class={labelCls} for="cc-currency">{$i18n.t('Currency')}</label>
							<input id="cc-currency" class={inputCls} bind:value={form.currency} maxlength="3" placeholder="EUR" />
						</div>
						<div>
							<label class={labelCls} for="cc-freq">{$i18n.t('Filing frequency')}</label>
							<select id="cc-freq" class={inputCls} bind:value={form.filing_frequency}>
								{#each FREQ as f}<option value={f}>{$i18n.t(f)}</option>{/each}
							</select>
						</div>
					</div>
				</div>

				<!-- VAT -->
				<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30">
					<div class="text-sm font-medium dark:text-gray-200 mb-3">{$i18n.t('VAT / Sales tax')}</div>
					<div class="grid grid-cols-1 md:grid-cols-4 gap-3">
						<div>
							<label class={labelCls} for="cc-vat">{$i18n.t('Standard rate')} (%)</label>
							<input id="cc-vat" type="number" step="0.01" class={inputCls} bind:value={form.vat_standard_rate} />
						</div>
						<div class="md:col-span-3">
							<label class={labelCls} for="cc-reduced">{$i18n.t('Reduced rates')} (%, {$i18n.t('comma-separated')})</label>
							<input id="cc-reduced" class={inputCls} bind:value={form.vat_reduced_rates} placeholder="10, 5.5, 2.1" />
						</div>
						<div>
							<label class={labelCls} for="cc-colpat">{$i18n.t('Collected acct pattern')}</label>
							<input id="cc-colpat" class={inputCls} bind:value={form.collected_pattern} placeholder="4457%" />
						</div>
						<div>
							<label class={labelCls} for="cc-dedpat">{$i18n.t('Deductible acct pattern')}</label>
							<input id="cc-dedpat" class={inputCls} bind:value={form.deductible_pattern} placeholder="4456%" />
						</div>
						<div>
							<label class={labelCls} for="cc-payc">{$i18n.t('Settlement payable code')}</label>
							<input id="cc-payc" class={inputCls} bind:value={form.settlement_payable_code} placeholder="4455" />
						</div>
						<div>
							<label class={labelCls} for="cc-recc">{$i18n.t('Settlement receivable code')}</label>
							<input id="cc-recc" class={inputCls} bind:value={form.settlement_receivable_code} />
						</div>
						<div>
							<label class={labelCls} for="cc-lcol">{$i18n.t('Collected label')}</label>
							<input id="cc-lcol" class={inputCls} bind:value={form.collected_label} />
						</div>
						<div>
							<label class={labelCls} for="cc-lded">{$i18n.t('Deductible label')}</label>
							<input id="cc-lded" class={inputCls} bind:value={form.deductible_label} />
						</div>
						<div>
							<label class={labelCls} for="cc-lpay">{$i18n.t('Payable label')}</label>
							<input id="cc-lpay" class={inputCls} bind:value={form.payable_label} />
						</div>
						<div>
							<label class={labelCls} for="cc-lrec">{$i18n.t('Receivable label')}</label>
							<input id="cc-lrec" class={inputCls} bind:value={form.receivable_label} />
						</div>
						<div>
							<label class={labelCls} for="cc-ldec">{$i18n.t('Declaration label')}</label>
							<input id="cc-ldec" class={inputCls} bind:value={form.declaration_label} />
						</div>
					</div>
				</div>

				<!-- Surcharges -->
				<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30">
					<div class="text-sm font-medium dark:text-gray-200 mb-1">{$i18n.t('VAT surcharges')}</div>
					<div class="text-xs text-gray-400 dark:text-gray-500 mb-3">
						{$i18n.t('Levied on VAT payable (e.g. China urban-construction / education / local-education).')}
					</div>
					<div class="grid grid-cols-1 md:grid-cols-3 gap-3">
						<div>
							<label class={labelCls} for="cc-su">{$i18n.t('Urban construction')} (%)</label>
							<input id="cc-su" type="number" step="0.01" class={inputCls} bind:value={form.surcharge_urban_rate} />
						</div>
						<div>
							<label class={labelCls} for="cc-se">{$i18n.t('Education surcharge')} (%)</label>
							<input id="cc-se" type="number" step="0.01" class={inputCls} bind:value={form.surcharge_education_rate} />
						</div>
						<div>
							<label class={labelCls} for="cc-sl">{$i18n.t('Local education surcharge')} (%)</label>
							<input id="cc-sl" type="number" step="0.01" class={inputCls} bind:value={form.surcharge_local_education_rate} />
						</div>
					</div>
				</div>

				<!-- Income taxes -->
				<div class="bg-white dark:bg-gray-900 rounded-xl p-4 border border-gray-100/30 dark:border-gray-850/30">
					<div class="text-sm font-medium dark:text-gray-200 mb-3">{$i18n.t('Income taxes (IS / IR)')}</div>
					<div class="grid grid-cols-1 md:grid-cols-3 gap-3">
						<div>
							<label class={labelCls} for="cc-cit">{$i18n.t('Corporate (IS/CIT) rate')} (%)</label>
							<input id="cc-cit" type="number" step="0.01" class={inputCls} bind:value={form.cit_rate} />
						</div>
						<div>
							<label class={labelCls} for="cc-citp">{$i18n.t('CIT preferential rate')} (%)</label>
							<input id="cc-citp" type="number" step="0.01" class={inputCls} bind:value={form.cit_preferential_rate} />
						</div>
						<div>
							<label class={labelCls} for="cc-ir">{$i18n.t('Individual (IR) rate')} (%)</label>
							<input id="cc-ir" type="number" step="0.01" class={inputCls} bind:value={form.ir_rate} />
						</div>
					</div>
				</div>

				<div class="flex justify-end">
					<button
						class="px-4 py-2 text-sm font-medium rounded-lg bg-gray-900 text-white hover:bg-gray-800 dark:bg-gray-100 dark:text-gray-800 dark:hover:bg-white transition disabled:opacity-50"
						on:click={save}
						disabled={saving}
					>
						{saving ? $i18n.t('Saving...') : $i18n.t('Save')}
					</button>
				</div>
			</div>
		{:else}
			<div class="text-sm text-gray-400 dark:text-gray-500 py-6 text-center">
				{$i18n.t('No country selected.')}
			</div>
		{/if}
	{/if}
</div>
