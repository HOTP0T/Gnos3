import { DISPATCHER_API_BASE_URL } from '$lib/constants';

export interface ModuleInfo {
	name: string;
	label: string;
	base_url: string;
}

export interface ModulesResponse {
	modules: ModuleInfo[];
}

/** Returns the modules enabled on this deployment. Empty list if the
 * dispatcher is unreachable (treat as "no modules available").
 *
 * Sends Authorization header when a token is available — the dispatcher
 * currently ignores it (Phase 1), but Phase 2 will filter modules by the
 * caller's permissions, so the wiring needs to be in place. */
export const getModules = async (): Promise<ModuleInfo[]> => {
	const token = typeof localStorage !== 'undefined' ? localStorage.getItem('token') : null;
	try {
		const res = await fetch(`${DISPATCHER_API_BASE_URL}/api/modules`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				...(token ? { Authorization: `Bearer ${token}` } : {})
			}
		});
		if (!res.ok) return [];
		const data: ModulesResponse = await res.json();
		return data.modules ?? [];
	} catch {
		return [];
	}
};
