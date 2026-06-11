<script lang="ts">
	import { onMount, getContext } from 'svelte';
	import { goto } from '$app/navigation';
	import { toast } from 'svelte-sonner';

	import { user, WEBUI_NAME, isAdmin} from '$lib/stores';
	import { getUsers } from '$lib/apis/users';
	import { getCompanies } from '$lib/apis/accounting';
	import {
		listCompanyMemberships,
		upsertCompanyMembership,
		deleteCompanyMembership,
		type CompanyMembership
	} from '$lib/apis/permissions';

	const i18n: any = getContext('i18n');

	type Role = 'viewer' | 'accountant' | 'admin' | 'none';
	const ROLES: { value: Role | ''; label: string }[] = [
		{ value: '', label: '(default — from group permissions)' },
		{ value: 'viewer', label: 'viewer' },
		{ value: 'accountant', label: 'accountant' },
		{ value: 'admin', label: 'admin' },
		{ value: 'none', label: 'none (explicit deny)' }
	];

	let loaded = false;
	let users: { id: string; name: string; email: string; role: string }[] = [];
	let companies: { id: number; name: string }[] = [];
	let memberships: CompanyMembership[] = [];

	let selectedUserId: string = '';
	let saving: Record<string, boolean> = {}; // keyed by `${userId}:${companyId}`

	const reloadMemberships = async () => {
		try {
			memberships = await listCompanyMemberships();
		} catch (err: any) {
			toast.error(err?.detail ?? 'Failed to load memberships');
			memberships = [];
		}
	};

	const currentRoleFor = (userId: string, companyId: number): Role | '' => {
		const row = memberships.find(
			(m) => m.user_id === userId && m.company_id === companyId
		);
		return row ? row.role : '';
	};

	const handleRoleChange = async (
		userId: string,
		companyId: number,
		newValue: Role | ''
	) => {
		const key = `${userId}:${companyId}`;
		saving[key] = true;
		saving = saving;
		try {
			if (newValue === '') {
				await deleteCompanyMembership(userId, companyId).catch((err) => {
					if (err?.detail !== 'Membership not found') throw err;
				});
				toast.success('Override removed (falls back to group permissions)');
			} else {
				await upsertCompanyMembership(userId, companyId, newValue);
				toast.success(`Set ${newValue} on company ${companyId}`);
			}
			await reloadMemberships();
		} catch (err: any) {
			toast.error(err?.detail ?? 'Save failed');
		} finally {
			saving[key] = false;
			saving = saving;
		}
	};

	onMount(async () => {
		if (!$isAdmin) {
			await goto('/');
			return;
		}

		try {
			const usersResp = await getUsers(localStorage.token);
			users = (usersResp?.users ?? usersResp ?? []).map((u: any) => ({
				id: u.id,
				name: u.name || u.email,
				email: u.email,
				role: u.role
			}));
		} catch (err: any) {
			toast.error('Failed to load users: ' + (err?.detail ?? err));
		}

		try {
			const companiesResp = await getCompanies({ active: true });
			companies = (companiesResp?.companies ?? []).map((c: any) => ({
				id: c.id,
				name: c.name
			}));
		} catch (err: any) {
			toast.error('Failed to load companies: ' + (err?.detail ?? err));
		}

		await reloadMemberships();

		// Default to the first non-admin user (most likely target of perm edits)
		const firstNonAdmin = users.find((u) => u.role !== 'admin');
		selectedUserId = firstNonAdmin?.id ?? users[0]?.id ?? '';

		loaded = true;
	});
</script>

<svelte:head>
	<title>{$i18n.t('Company Permissions')} • {$WEBUI_NAME}</title>
</svelte:head>

{#if loaded}
	<div class="px-4 md:px-6 py-4 max-w-5xl mx-auto">
		<h1 class="text-xl font-medium mb-1">{$i18n.t('Company Permissions')}</h1>
		<p class="text-sm text-gray-500 dark:text-gray-400 mb-4">
			Per-company role overrides on top of group permissions. Resolution order: this
			table → group's <code>companies.&lt;id&gt;</code> → group's <code>companies.*</code> →
			deny. Admins bypass entirely.
		</p>

		<div class="mb-4">
			<label class="block text-sm font-medium mb-1" for="user-select">User</label>
			<select
				id="user-select"
				bind:value={selectedUserId}
				class="w-full md:w-96 px-3 py-2 rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-sm"
			>
				{#each users as u (u.id)}
					<option value={u.id}>
						{u.name} &lt;{u.email}&gt; · {u.role}
					</option>
				{/each}
			</select>
		</div>

		{#if selectedUserId}
			{@const selectedUser = users.find((u) => u.id === selectedUserId)}
			{#if selectedUser?.role === 'admin' || selectedUser?.role === 'superadmin'}
				<div
					class="mb-4 px-3 py-2 rounded-lg bg-amber-50 dark:bg-amber-950 text-sm text-amber-800 dark:text-amber-200"
				>
					This user has the global <strong>admin</strong> role. They bypass per-company
					checks regardless of overrides set here. Changes still save but will not
					affect their access.
				</div>
			{/if}

			{#if companies.length === 0}
				<p class="text-sm text-gray-500 italic">
					No companies exist yet — create one under Accounting → Companies first.
				</p>
			{:else}
				<table class="w-full text-sm border-collapse">
					<thead>
						<tr class="border-b border-gray-200 dark:border-gray-800">
							<th class="text-left py-2 px-3 font-medium">Company</th>
							<th class="text-left py-2 px-3 font-medium w-72">Role override</th>
						</tr>
					</thead>
					<tbody>
						{#each companies as c (c.id)}
							{@const role = currentRoleFor(selectedUserId, c.id)}
							{@const key = `${selectedUserId}:${c.id}`}
							<tr class="border-b border-gray-100 dark:border-gray-900">
								<td class="py-2 px-3">
									{c.name}
									<span class="text-xs text-gray-400 ml-2">#{c.id}</span>
								</td>
								<td class="py-2 px-3">
									<select
										value={role}
										on:change={(e) =>
											handleRoleChange(
												selectedUserId,
												c.id,
												(e.currentTarget as HTMLSelectElement).value as Role | ''
											)}
										disabled={!!saving[key]}
										class="w-full px-2 py-1.5 rounded-md bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-sm disabled:opacity-50"
									>
										{#each ROLES as opt (opt.value)}
											<option value={opt.value}>{opt.label}</option>
										{/each}
									</select>
								</td>
							</tr>
						{/each}
					</tbody>
				</table>
			{/if}
		{/if}
	</div>
{/if}
