import { writable } from 'svelte/store';
import { getModules, type ModuleInfo } from '$lib/apis/modules';

/** Modules enabled on this deployment, populated from the dispatcher's
 * /api/modules endpoint at app boot. Empty until the first fetch resolves. */
export const enabledModules = writable<ModuleInfo[]>([]);

let initialized = false;

export const refreshEnabledModules = async () => {
	const mods = await getModules();
	enabledModules.set(mods);
	return mods;
};

export const ensureModulesLoaded = async () => {
	if (initialized) return;
	initialized = true;
	await refreshEnabledModules();
};
