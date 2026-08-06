import { writable, type Writable } from 'svelte/store';

// Common data batch-loaded once by ConfigurableDashboard and shared with every
// KPI tile via context, so 10 KPI widgets don't each refetch the same stats.
export interface CommonData {
	loading: boolean;
	stats: any; // getCompanyStats
	balanceSheet: any; // getBalanceSheet
	profitLoss: any; // getProfitLoss (current month)
}

export type CommonDataStore = Writable<CommonData>;

export function createCommonData(): CommonDataStore {
	return writable<CommonData>({ loading: true, stats: {}, balanceSheet: null, profitLoss: null });
}

export const COMMON_DATA_CTX = 'dashboardCommonData';
