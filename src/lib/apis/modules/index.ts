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
 * dispatcher is unreachable (treat as "no modules available"). */
export const getModules = async (): Promise<ModuleInfo[]> => {
	try {
		const res = await fetch(`${DISPATCHER_API_BASE_URL}/api/modules`, {
			method: 'GET',
			headers: { 'Content-Type': 'application/json' }
		});
		if (!res.ok) return [];
		const data: ModulesResponse = await res.json();
		return data.modules ?? [];
	} catch {
		return [];
	}
};
