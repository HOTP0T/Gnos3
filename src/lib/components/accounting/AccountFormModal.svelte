<script lang="ts">
	import { onMount, onDestroy, getContext, createEventDispatcher, tick } from 'svelte';
	import { fade } from 'svelte/transition';
	import { flyAndScale } from '$lib/utils/transitions';
	import { toast } from 'svelte-sonner';

	import { createAccount, updateAccount } from '$lib/apis/accounting';

	const i18n = getContext('i18n');
	const dispatch = createEventDispatcher();

	export let show = false;
	export let account: any = null;
	export let accounts: any[] = [];
	export let companyId: number;

	const ACCOUNT_TYPES = ['asset', 'liability', 'equity', 'revenue', 'expense'];

	// Form state
	let code = '';
	let name = '';
	let account_type = 'asset';
	let parent_id: number | null = null;
	let description = '';
	let submitting = false;

	let modalElement: HTMLElement | null = null;

	$: if (show) {
		initForm();
	}

	$: filteredParents = accounts.filter(
		(a) => a.account_type === account_type && a.is_active && (!account || a.id !== account.id)
	);

	const initForm = () => {
		if (account) {
			code = account.code || '';
			name = account.name || '';
			account_type = account.account_type || 'asset';
			parent_id = account.parent_id || null;
			description = account.description || '';
		} else {
			code = '';
			name = '';
			account_type = 'asset';
			parent_id = null;
			description = '';
		}
	};

	const handleKeyDown = (event: KeyboardEvent) => {
		if (event.key === 'Escape') {
			show = false;
		}
	};

	const handleSubmit = async () => {
		if (!code.trim() || !name.trim()) {
			toast.error($i18n.t('Code and Name are required'));
			return;
		}

		submitting = true;
		try {
			const payload: Record<string, any> = {
				code: code.trim(),
				name: name.trim(),
				account_type,
				description: description.trim() || null
			};
			if (parent_id) {
				payload.parent_id = parent_id;
			}

			let result;
			if (account) {
				result = await updateAccount(account.id, payload);
				toast.success($i18n.t('Account updated'));
			} else {
				result = await createAccount(payload, companyId);
				toast.success($i18n.t('Account created'));
			}
			dispatch('save', result);
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
			class="m-auto max-w-full w-[32rem] mx-2 bg-white/95 dark:bg-gray-950/95 backdrop-blur-sm rounded-4xl max-h-[100dvh] shadow-3xl border border-white dark:border-gray-900"
			in:flyAndScale
			on:mousedown={(e) => {
				e.stopPropagation();
			}}
		>
			<div class="px-[1.75rem] py-6 flex flex-col">
				<div class="text-lg font-medium dark:text-gray-200 mb-4">
					{account ? $i18n.t('Edit Account') : $i18n.t('Add Account')}
				</div>

				<form
					class="flex flex-col gap-3"
					on:submit|preventDefault={handleSubmit}
				>
					<!-- Code -->
					<div>
						<label
							for="account-code"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Code')}
						</label>
						<input
							id="account-code"
							type="text"
							bind:value={code}
							placeholder={$i18n.t('e.g. 1000')}
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
							required
						/>
					</div>

					<!-- Name -->
					<div>
						<label
							for="account-name"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Name')}
						</label>
						<input
							id="account-name"
							type="text"
							bind:value={name}
							placeholder={$i18n.t('e.g. Cash')}
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
							required
						/>
					</div>

					<!-- Account Type -->
					<div>
						<label
							for="account-type"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Account Type')}
						</label>
						<select
							id="account-type"
							bind:value={account_type}
							on:change={() => {
								parent_id = null;
							}}
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition capitalize"
						>
							{#each ACCOUNT_TYPES as t}
								<option value={t}>{$i18n.t(t.charAt(0).toUpperCase() + t.slice(1))}</option>
							{/each}
						</select>
					</div>

					<!-- Parent Account -->
					<div>
						<label
							for="account-parent"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Parent Account')}
						</label>
						<select
							id="account-parent"
							bind:value={parent_id}
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition"
						>
							<option value={null}>{$i18n.t('None (top-level)')}</option>
							{#each filteredParents as p}
								<option value={p.id}>{p.code} - {p.name}</option>
							{/each}
						</select>
					</div>

					<!-- Description -->
					<div>
						<label
							for="account-description"
							class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
						>
							{$i18n.t('Description')}
						</label>
						<textarea
							id="account-description"
							bind:value={description}
							placeholder={$i18n.t('Optional description')}
							rows="3"
							class="w-full rounded-lg px-4 py-2 text-sm dark:text-gray-300 dark:bg-gray-900 bg-gray-50 outline-hidden border border-gray-200 dark:border-gray-800 focus:border-blue-500 transition resize-none"
						></textarea>
					</div>

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
							{submitting
								? $i18n.t('Saving...')
								: account
									? $i18n.t('Update')
									: $i18n.t('Create')}
						</button>
					</div>
				</form>
			</div>
		</div>
	</div>
{/if}
