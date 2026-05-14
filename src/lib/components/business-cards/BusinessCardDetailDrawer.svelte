<script lang="ts">
	import { createEventDispatcher, getContext } from 'svelte';
	import { toast } from 'svelte-sonner';
	import { fly, fade } from 'svelte/transition';

	import {
		updateBusinessCard,
		reprocessBusinessCard,
		syncBusinessCardFromK4mi,
		getBusinessCardPreviewUrl,
		type BusinessCard
	} from '$lib/apis/business-cards';
	import { K4MI_BASE_URL } from '$lib/constants';

	import Tooltip from '$lib/components/common/Tooltip.svelte';
	import Spinner from '$lib/components/common/Spinner.svelte';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher<{
		close: void;
		updated: BusinessCard;
		deleteRequest: BusinessCard;
	}>();

	export let card: BusinessCard;

	let draft: Partial<BusinessCard> = {};
	let saving = false;
	let syncing = false;
	let reprocessing = false;

	$: dirty = Object.keys(draft).length > 0;
	$: k4miHref = card.k4mi_document_id
		? `${K4MI_BASE_URL}/documents/${card.k4mi_document_id}/details`
		: '';

	type Field = {
		key: keyof BusinessCard;
		label: string;
		type?: 'text' | 'textarea';
		full?: boolean;
	};

	const SECTIONS: Array<{ title: string; fields: Field[] }> = [
		{
			title: 'Identity',
			fields: [
				{ key: 'full_name', label: 'Full name' },
				{ key: 'job_title', label: 'Job title' },
				{ key: 'company_name', label: 'Company' }
			]
		},
		{
			title: 'Contact',
			fields: [
				{ key: 'email', label: 'Email' },
				{ key: 'email_secondary', label: 'Email (secondary)' },
				{ key: 'phone', label: 'Phone' },
				{ key: 'mobile', label: 'Mobile' },
				{ key: 'fax', label: 'Fax' }
			]
		},
		{
			title: 'Online & address',
			fields: [
				{ key: 'website', label: 'Website' },
				{ key: 'linkedin', label: 'LinkedIn' },
				{ key: 'address', label: 'Address', type: 'textarea', full: true }
			]
		},
		{
			title: 'Notes',
			fields: [{ key: 'notes', label: 'Card notes', type: 'textarea', full: true }]
		}
	];

	const fieldValue = (key: keyof BusinessCard): string => {
		const v = key in draft ? (draft as any)[key] : (card as any)[key];
		return v == null ? '' : String(v);
	};

	const onChange = (key: keyof BusinessCard, value: string) => {
		const normalized = value.trim() === '' ? null : value;
		const original = (card as any)[key] ?? null;
		const isSame =
			normalized === original ||
			(normalized == null && (original == null || original === ''));
		if (isSame) {
			delete (draft as any)[key];
			draft = { ...draft };
		} else {
			draft = { ...draft, [key]: normalized };
		}
	};

	const handleSave = async () => {
		if (!dirty) return;
		saving = true;
		try {
			const updated = await updateBusinessCard(localStorage.token, card.id, draft);
			toast.success($i18n.t('Saved'));
			draft = {};
			dispatch('updated', updated);
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			saving = false;
		}
	};

	const handleSync = async () => {
		if (!card.k4mi_document_id) return;
		syncing = true;
		try {
			await syncBusinessCardFromK4mi(localStorage.token, card.id);
			toast.success($i18n.t('Syncing from K4mi…'));
			setTimeout(() => dispatch('close'), 600);
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			syncing = false;
		}
	};

	const handleReprocess = async () => {
		if (!card.k4mi_document_id) return;
		reprocessing = true;
		try {
			await reprocessBusinessCard(localStorage.token, card.id);
			toast.success($i18n.t('Re-extraction queued'));
			setTimeout(() => dispatch('close'), 600);
		} catch (err) {
			toast.error(`${err}`);
		} finally {
			reprocessing = false;
		}
	};

	const formatK4miNotes = (
		notes: unknown
	): Array<{ text: string; created: string }> => {
		if (!Array.isArray(notes)) return [];
		return notes
			.map((n) => {
				if (typeof n === 'string') return { text: n, created: '' };
				if (n && typeof n === 'object') {
					const obj = n as { text?: unknown; created?: unknown };
					return { text: String(obj.text ?? ''), created: String(obj.created ?? '') };
				}
				return { text: '', created: '' };
			})
			.filter((n) => n.text);
	};

	$: k4miNotes = formatK4miNotes(card.k4mi_notes);
	$: confidencePct =
		card.confidence_score != null ? Math.round(Number(card.confidence_score) * 100) : null;

	$: previewUrl = card.k4mi_document_id ? getBusinessCardPreviewUrl(card.id) : null;
	let previewMode: 'image' | 'iframe' = 'image';
	let previewError = false;
	let previewExpanded = false;
	$: card, ((previewMode = 'image'), (previewError = false), (previewExpanded = false));

	const handleClose = () => {
		if (dirty && !confirm($i18n.t('Discard unsaved changes?'))) return;
		dispatch('close');
	};

	const handleKeydown = (e: KeyboardEvent) => {
		if (e.key === 'Escape') handleClose();
		if ((e.metaKey || e.ctrlKey) && e.key === 's') {
			e.preventDefault();
			if (dirty) handleSave();
		}
	};
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="fixed inset-0 z-50 flex" transition:fade={{ duration: 150 }}>
	<button
		class="flex-1 bg-black/30 dark:bg-black/50 cursor-default"
		on:click={handleClose}
		aria-label={$i18n.t('Close')}
	></button>

	<aside
		class="w-full max-w-3xl bg-white dark:bg-gray-900 shadow-2xl flex flex-col h-full overflow-hidden"
		transition:fly={{ x: 600, duration: 250 }}
	>
		<header class="px-5 py-4 border-b border-gray-200 dark:border-gray-800 flex-shrink-0">
			<div class="flex items-start justify-between gap-3">
				<div class="min-w-0 flex-1">
					<h2 class="text-lg font-semibold text-gray-900 dark:text-gray-50 truncate">
						{card.full_name || $i18n.t('Untitled card')}
					</h2>
					{#if card.job_title || card.company_name}
						<p class="text-sm text-gray-500 dark:text-gray-400 truncate mt-0.5">
							{[card.job_title, card.company_name].filter(Boolean).join(' · ')}
						</p>
					{/if}

					<div class="flex flex-wrap items-center gap-2 mt-2.5">
						{#if card.processing_status === 'processing' || card.processing_status === 'pending'}
							<span
								class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
							>
								<Spinner className="size-3" />
								{$i18n.t('Processing')}
							</span>
						{:else if card.processing_status === 'failed'}
							<span
								class="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700 dark:bg-red-500/10 dark:text-red-300"
							>
								{$i18n.t('Failed')}
							</span>
						{:else if card.needs_review}
							<span
								class="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300"
							>
								{$i18n.t('Needs review')}
							</span>
						{:else}
							<span
								class="inline-flex px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300"
							>
								{$i18n.t('OK')}
							</span>
						{/if}

						{#if confidencePct != null}
							<span
								class="text-xs text-gray-500 dark:text-gray-400"
								title={$i18n.t('Extraction confidence')}
							>
								{confidencePct}% {$i18n.t('confidence')}
							</span>
						{/if}

						{#if card.user_corrected}
							<span
								class="text-xs text-blue-700 dark:text-blue-300"
								title={$i18n.t('Manually edited — DB is authoritative')}
							>
								{$i18n.t('User-edited')}
							</span>
						{/if}
					</div>
				</div>

				<button
					class="p-1 -m-1 text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition"
					on:click={handleClose}
					aria-label={$i18n.t('Close')}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="size-5"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg
					>
				</button>
			</div>

			<div class="flex flex-wrap items-center gap-2 mt-3">
				{#if k4miHref}
					<a
						class="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition"
						href={k4miHref}
						target="_blank"
						rel="noopener noreferrer"
					>
						<svg
							xmlns="http://www.w3.org/2000/svg"
							class="size-3.5"
							viewBox="0 0 24 24"
							fill="none"
							stroke="currentColor"
							stroke-width="2"
							stroke-linecap="round"
							stroke-linejoin="round"
							><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" /><polyline
								points="15 3 21 3 21 9"
							/><line x1="10" y1="14" x2="21" y2="3" /></svg
						>
						{$i18n.t('Open in K4mi')}
					</a>
				{/if}
				{#if card.k4mi_document_id}
					<button
						class="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition disabled:opacity-40 disabled:cursor-not-allowed"
						on:click={handleSync}
						disabled={syncing}
					>
						{#if syncing}
							<Spinner className="size-3.5" />
						{:else}
							<svg
								xmlns="http://www.w3.org/2000/svg"
								class="size-3.5"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path
									d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"
								/></svg
							>
						{/if}
						{$i18n.t('Sync from K4mi')}
					</button>
					<button
						class="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition disabled:opacity-40 disabled:cursor-not-allowed"
						on:click={handleReprocess}
						disabled={reprocessing}
					>
						{#if reprocessing}
							<Spinner className="size-3.5" />
						{:else}
							<svg
								xmlns="http://www.w3.org/2000/svg"
								class="size-3.5"
								viewBox="0 0 24 24"
								fill="none"
								stroke="currentColor"
								stroke-width="2"
								stroke-linecap="round"
								stroke-linejoin="round"
								><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" /></svg
							>
						{/if}
						{$i18n.t('Reprocess')}
					</button>
				{/if}
				<button
					class="ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-md text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-500/10 transition"
					on:click={() => dispatch('deleteRequest', card)}
				>
					<svg
						xmlns="http://www.w3.org/2000/svg"
						class="size-3.5"
						viewBox="0 0 24 24"
						fill="none"
						stroke="currentColor"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						><polyline points="3 6 5 6 21 6" /><path
							d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"
						/></svg
					>
					{$i18n.t('Delete')}
				</button>
			</div>
		</header>

		<div class="flex-1 overflow-y-auto px-5 py-4 space-y-6">
			{#if previewUrl}
				<section>
					<div class="flex items-center justify-between mb-2">
						<h3
							class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400"
						>
							{$i18n.t('Original document')}
						</h3>
						<div class="flex items-center gap-2">
							<button
								class="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition inline-flex items-center gap-1"
								on:click={() => (previewExpanded = !previewExpanded)}
								aria-label={previewExpanded ? $i18n.t('Collapse') : $i18n.t('Expand')}
							>
								{#if previewExpanded}
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="size-3.5"
										viewBox="0 0 24 24"
										fill="none"
										stroke="currentColor"
										stroke-width="2"
										stroke-linecap="round"
										stroke-linejoin="round"
										><polyline points="4 14 10 14 10 20" /><polyline
											points="20 10 14 10 14 4"
										/><line x1="14" y1="10" x2="21" y2="3" /><line
											x1="3"
											y1="21"
											x2="10"
											y2="14"
										/></svg
									>
									{$i18n.t('Collapse')}
								{:else}
									<svg
										xmlns="http://www.w3.org/2000/svg"
										class="size-3.5"
										viewBox="0 0 24 24"
										fill="none"
										stroke="currentColor"
										stroke-width="2"
										stroke-linecap="round"
										stroke-linejoin="round"
										><polyline points="15 3 21 3 21 9" /><polyline points="9 21 3 21 3 15" /><line
											x1="21"
											y1="3"
											x2="14"
											y2="10"
										/><line x1="3" y1="21" x2="10" y2="14" /></svg
									>
									{$i18n.t('Expand')}
								{/if}
							</button>
							<a
								class="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition"
								href={previewUrl}
								target="_blank"
								rel="noopener noreferrer"
							>
								{$i18n.t('Open in new tab')}
							</a>
						</div>
					</div>

					<div
						class="rounded-md border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-950 overflow-hidden flex items-center justify-center {previewExpanded
							? 'min-h-[60vh]'
							: 'h-[320px]'}"
					>
						{#if previewError}
							<div class="text-center text-sm text-gray-500 dark:text-gray-400 p-6">
								<p>{$i18n.t('Could not load the original document.')}</p>
								<p class="text-xs mt-1">
									{$i18n.t('Check that bc-api can reach K4mi and that the document still exists.')}
								</p>
								{#if k4miHref}
									<a
										class="inline-block mt-2 text-xs text-blue-600 dark:text-blue-400 hover:underline"
										href={k4miHref}
										target="_blank"
										rel="noopener noreferrer"
									>
										{$i18n.t('Open in K4mi instead →')}
									</a>
								{/if}
							</div>
						{:else if previewMode === 'image'}
							<img
								src={previewUrl}
								alt={card.full_name || 'Business card'}
								class="max-w-full max-h-full object-contain"
								on:error={() => {
									// Probably a PDF — swap to iframe.
									previewMode = 'iframe';
								}}
							/>
						{:else}
							<iframe
								src={previewUrl}
								class="w-full h-full"
								title={card.full_name || 'Business card preview'}
								on:error={() => (previewError = true)}
							></iframe>
						{/if}
					</div>
				</section>
			{/if}

			{#each SECTIONS as section}
				<section>
					<h3
						class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2"
					>
						{$i18n.t(section.title)}
					</h3>
					<div class="grid grid-cols-2 gap-x-3 gap-y-2.5">
						{#each section.fields as f}
							<label class={f.full ? 'col-span-2 block' : 'block'}>
								<span class="block text-xs text-gray-500 dark:text-gray-400 mb-0.5">
									{$i18n.t(f.label)}
								</span>
								{#if f.type === 'textarea'}
									<textarea
										class="w-full px-2.5 py-1.5 text-sm rounded-md bg-gray-50 dark:bg-gray-800 border border-transparent focus:border-blue-400 dark:focus:border-blue-500 focus:bg-white dark:focus:bg-gray-900 outline-none transition resize-y min-h-[3rem]"
										rows="2"
										value={fieldValue(f.key)}
										on:input={(e) => onChange(f.key, e.currentTarget.value)}
									></textarea>
								{:else}
									<input
										class="w-full px-2.5 py-1.5 text-sm rounded-md bg-gray-50 dark:bg-gray-800 border border-transparent focus:border-blue-400 dark:focus:border-blue-500 focus:bg-white dark:focus:bg-gray-900 outline-none transition"
										type="text"
										value={fieldValue(f.key)}
										on:input={(e) => onChange(f.key, e.currentTarget.value)}
									/>
								{/if}
							</label>
						{/each}
					</div>
				</section>
			{/each}

			<section>
				<h3
					class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2"
				>
					{$i18n.t('K4mi document notes')}
				</h3>
				{#if k4miNotes.length === 0}
					<p class="text-xs text-gray-400 dark:text-gray-600">
						{$i18n.t('No K4mi notes on this document yet. Add notes in K4mi and click "Sync from K4mi" above to pull them in.')}
					</p>
				{:else}
					<ul class="space-y-2">
						{#each k4miNotes as n}
							<li
								class="text-sm bg-gray-50 dark:bg-gray-800/60 rounded-md px-3 py-2 border border-gray-100 dark:border-gray-800"
							>
								<p class="text-gray-700 dark:text-gray-200 whitespace-pre-wrap">{n.text}</p>
								{#if n.created}
									<p class="text-xs text-gray-400 dark:text-gray-500 mt-1">
										{new Date(n.created).toLocaleString()}
									</p>
								{/if}
							</li>
						{/each}
					</ul>
				{/if}
			</section>

			{#if Array.isArray(card.k4mi_tags) && card.k4mi_tags.length > 0}
				<section>
					<h3
						class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2"
					>
						{$i18n.t('K4mi tags')}
					</h3>
					<div class="flex flex-wrap gap-1.5">
						{#each card.k4mi_tags as tag}
							<span
								class="inline-flex px-2 py-0.5 text-xs rounded-md bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300"
							>
								{tag}
							</span>
						{/each}
					</div>
				</section>
			{/if}

			{#if card.extraction_model}
				<section>
					<h3
						class="text-xs font-semibold uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2"
					>
						{$i18n.t('Extraction metadata')}
					</h3>
					<dl class="grid grid-cols-3 gap-x-3 gap-y-1.5 text-xs">
						<dt class="col-span-1 text-gray-500 dark:text-gray-400">{$i18n.t('Model')}</dt>
						<dd class="col-span-2 text-gray-700 dark:text-gray-300 break-all">
							{card.extraction_model}
						</dd>
						{#if card.created_at}
							<dt class="col-span-1 text-gray-500 dark:text-gray-400">{$i18n.t('Created')}</dt>
							<dd class="col-span-2 text-gray-700 dark:text-gray-300">
								{new Date(card.created_at).toLocaleString()}
							</dd>
						{/if}
						{#if card.updated_at}
							<dt class="col-span-1 text-gray-500 dark:text-gray-400">{$i18n.t('Updated')}</dt>
							<dd class="col-span-2 text-gray-700 dark:text-gray-300">
								{new Date(card.updated_at).toLocaleString()}
							</dd>
						{/if}
					</dl>
				</section>
			{/if}
		</div>

		<footer
			class="px-5 py-3 border-t border-gray-200 dark:border-gray-800 flex items-center justify-between gap-3 flex-shrink-0 bg-gray-50/60 dark:bg-gray-900/40"
		>
			<div class="text-xs text-gray-500 dark:text-gray-400">
				{#if dirty}
					{$i18n.t('Unsaved changes')}
				{:else}
					{$i18n.t('All changes saved')}
				{/if}
			</div>
			<div class="flex items-center gap-2">
				<button
					class="px-3 py-1.5 text-sm rounded-md text-gray-700 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-800 transition"
					on:click={handleClose}
				>
					{$i18n.t('Close')}
				</button>
				<Tooltip content={$i18n.t('Cmd/Ctrl + S')}>
					<button
						class="px-3 py-1.5 text-sm rounded-md bg-blue-600 hover:bg-blue-700 text-white transition disabled:opacity-40 disabled:cursor-not-allowed"
						on:click={handleSave}
						disabled={!dirty || saving}
					>
						{saving ? $i18n.t('Saving…') : $i18n.t('Save')}
					</button>
				</Tooltip>
			</div>
		</footer>
	</aside>
</div>
