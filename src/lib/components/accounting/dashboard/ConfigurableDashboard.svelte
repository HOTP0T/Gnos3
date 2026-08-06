<script lang="ts">
	import { onMount, onDestroy, setContext, getContext, tick } from 'svelte';
	import { toast } from 'svelte-sonner';
	import dayjs from 'dayjs';
	import 'gridstack/dist/gridstack.min.css';

	import {
		getBalanceSheet,
		getProfitLoss,
		getCompanyStats,
		getDashboardLayout,
		saveDashboardLayout,
		type DashboardLayout
	} from '$lib/apis/accounting';
	import Spinner from '$lib/components/common/Spinner.svelte';
	import WidgetCard from './WidgetCard.svelte';
	import { createCommonData, COMMON_DATA_CTX } from './store';
	import { WIDGETS, WIDGET_BY_TYPE, defaultLayout, type LayoutItem, type WidgetCategory } from './registry';

	export let companyId: number;

	const i18n: any = getContext('i18n');
	const T = (k: string) => (i18n?.t ? i18n.t(k) : k);

	// ── Shared common-data store (KPI tiles + alerts read from this) ──
	const common = createCommonData();
	setContext(COMMON_DATA_CTX, common);

	let loading = true;
	let editing = false;
	let dirty = false;
	let saving = false;
	let showCatalog = false;

	let layout: LayoutItem[] = [];
	let els: Record<string, HTMLElement> = {};
	let gridEl: HTMLDivElement;
	let grid: any = null;
	let GridStackCls: any = null;
	let idSeq = 1;

	const categories: WidgetCategory[] = ['KPI', 'Charts', 'Lists', 'Panels'];
	$: catalogByCat = categories.map((c) => ({ cat: c, items: WIDGETS.filter((w) => w.category === c) }));

	function newId() {
		return `w${Date.now().toString(36)}${idSeq++}`;
	}

	// Svelte action: set gridstack's hyphenated attributes ONCE, on mount, and
	// never again. This is critical — if these attributes were bound reactively
	// (e.g. an interpolated attr or a spread), Svelte would re-write gs-w/gs-h on
	// every array change and fight gridstack, resizing untouched widgets. gridstack
	// owns positioning after init; Svelte must not touch it.
	function gsItem(node: HTMLElement, params: { item: LayoutItem; def: any }) {
		const { item, def } = params;
		node.setAttribute('gs-id', String(item.id));
		node.setAttribute('gs-x', String(item.x));
		node.setAttribute('gs-y', String(item.y));
		node.setAttribute('gs-w', String(item.w));
		node.setAttribute('gs-h', String(item.h));
		node.setAttribute('gs-min-w', String(def?.minW ?? 2));
		node.setAttribute('gs-min-h', String(def?.minH ?? 2));
		// No `update` handler on purpose: attributes are never re-applied.
		return {};
	}

	// ── Load common data + saved layout ──
	onMount(async () => {
		await Promise.all([loadCommon(), loadLayout()]);
		loading = false;
		await tick();
		await initGrid();
	});

	async function loadCommon() {
		try {
			const now = new Date();
			const monthStart = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-01`;
			const monthEnd = dayjs(now).format('YYYY-MM-DD');
			const [stats, balanceSheet, profitLoss] = await Promise.all([
				getCompanyStats(companyId).catch(() => ({})),
				getBalanceSheet({ company_id: companyId }).catch(() => null),
				getProfitLoss({ date_from: monthStart, date_to: monthEnd, company_id: companyId }).catch(() => null)
			]);
			common.set({ loading: false, stats, balanceSheet, profitLoss });
		} catch {
			common.set({ loading: false, stats: {}, balanceSheet: null, profitLoss: null });
		}
	}

	async function loadLayout() {
		try {
			const res = await getDashboardLayout(companyId);
			const saved = res?.layout as DashboardLayout | undefined;
			if (saved && Array.isArray((saved as any).widgets) && (saved as any).widgets.length) {
				layout = (saved as any).widgets
					.filter((w: any) => WIDGET_BY_TYPE[w.type])
					.map((w: any) => ({
						id: w.id || newId(),
						type: w.type,
						x: w.x ?? 0,
						y: w.y ?? 0,
						w: w.w ?? WIDGET_BY_TYPE[w.type].w,
						h: w.h ?? WIDGET_BY_TYPE[w.type].h,
						options: w.options
					}));
				return;
			}
		} catch {
			// fall through to default
		}
		layout = defaultLayout();
	}

	// ── Gridstack ──
	async function initGrid() {
		if (grid) return;
		const mod = await import('gridstack');
		GridStackCls = mod.GridStack;
		grid = GridStackCls.init(
			{
				column: 12,
				cellHeight: 54,
				margin: 6,
				// float:true — widgets stay exactly where the user puts them; moving
				// or resizing one never reflows/compacts the others. Only widgets the
				// dragged item directly collides with get pushed.
				float: true,
				// Start read-only. Edit mode flips this + enables move/resize.
				// No `handle` — the whole item drags, and the corner/edge resize
				// handles coexist (they stop propagation), so both gestures work.
				staticGrid: true,
				resizable: { handles: 'se' },
				columnOpts: { breakpoints: [{ w: 768, c: 1 }], breakpointForWindow: true }
			},
			gridEl
		);

		// Only flag dirty. Never write positions back into the reactive `layout`
		// array here — that would re-render items and disturb gridstack. Positions
		// are read straight from gridstack at save time.
		grid.on('change', () => {
			if (editing) dirty = true;
		});
	}

	// ── Edit mode ──
	function enterEdit() {
		editing = true;
		showCatalog = true;
		// setStatic(false) alone does not reliably re-enable per-widget drag/resize
		// across gridstack versions — call the explicit enablers too.
		if (grid) {
			grid.setStatic(false);
			grid.enableMove(true);
			grid.enableResize(true);
		}
	}

	async function exitEdit() {
		editing = false;
		showCatalog = false;
		if (grid) {
			grid.enableMove(false);
			grid.enableResize(false);
			grid.setStatic(true);
		}
		if (dirty) await persist();
	}

	async function persist() {
		saving = true;
		try {
			// Read live positions straight from gridstack, and merge in each
			// widget's type/options (which gridstack doesn't track) by id.
			const meta: Record<string, LayoutItem> = {};
			for (const it of layout) meta[it.id] = it;
			const nodes = (grid ? (grid.save(false) as any[]) : []).filter((n) => n.id != null);
			const payload: DashboardLayout = {
				version: 1,
				widgets: nodes.map((n) => {
					const it = meta[String(n.id)];
					return {
						id: String(n.id),
						type: it?.type,
						x: n.x ?? 0,
						y: n.y ?? 0,
						w: n.w ?? 1,
						h: n.h ?? 1,
						...(it?.options ? { options: it.options } : {})
					};
				}).filter((w) => w.type)
			};
			await saveDashboardLayout(companyId, payload);
			dirty = false;
			toast.success(T('Dashboard saved'));
		} catch (err) {
			toast.error(`${err}`);
		}
		saving = false;
	}

	// ── Add / remove widgets ──
	async function addWidget(type: string) {
		const def = WIDGET_BY_TYPE[type];
		if (!def || !grid) return;
		const nodes = grid.save(false) as any[];
		const maxY = nodes.reduce((m, n) => Math.max(m, (n.y ?? 0) + (n.h ?? 1)), 0);
		const item: LayoutItem = {
			id: newId(),
			type,
			x: 0,
			y: maxY,
			w: def.w,
			h: def.h,
			options: def.options
		};
		layout = [...layout, item];
		dirty = true;
		await tick();
		const el = els[item.id];
		if (el) {
			grid.makeWidget(el);
			el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
		}
	}

	function removeWidget(item: LayoutItem) {
		const el = els[item.id];
		if (grid && el) grid.removeWidget(el, false);
		layout = layout.filter((it) => it.id !== item.id);
		dirty = true;
	}

	async function resetDefault() {
		if (!grid) return;
		// remove all current widgets from gridstack, then swap layout
		for (const it of layout) {
			const el = els[it.id];
			if (el) grid.removeWidget(el, false);
		}
		// Fresh ids so Svelte creates brand-new nodes (the mount-once action re-runs
		// and re-applies the default positions/sizes).
		layout = defaultLayout().map((d) => ({ ...d, id: newId() }));
		dirty = true;
		await tick();
		for (const it of layout) {
			const el = els[it.id];
			if (el) grid.makeWidget(el);
		}
	}

	onDestroy(() => {
		if (grid) {
			try {
				grid.destroy(false);
			} catch {}
			grid = null;
		}
	});
</script>

<div class="py-3">
	<!-- Toolbar -->
	<div class="flex items-center justify-between mb-3 gap-2 flex-wrap">
		<div class="text-xs text-gray-400 dark:text-gray-500">
			{#if editing}
				{T('Drag to move, resize from the corner, or add widgets below.')}
			{/if}
		</div>
		<div class="flex items-center gap-2">
			{#if editing}
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-850 transition"
					on:click={resetDefault}
				>
					{T('Reset to default')}
				</button>
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition flex items-center gap-1.5"
					on:click={exitEdit}
					disabled={saving}
				>
					{#if saving}<Spinner className="size-3" />{/if}
					{T('Done')}
				</button>
			{:else}
				<button
					class="px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-850 transition flex items-center gap-1.5"
					on:click={enterEdit}
				>
					<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" class="w-3.5 h-3.5">
						<path d="M8.34 1.804A1 1 0 0 1 9.32 1h1.36a1 1 0 0 1 .98.804l.295 1.473c.497.144.971.342 1.416.587l1.25-.834a1 1 0 0 1 1.262.125l.962.962a1 1 0 0 1 .125 1.262l-.834 1.25c.245.445.443.919.587 1.416l1.473.294a1 1 0 0 1 .804.98v1.361a1 1 0 0 1-.804.98l-1.473.295a6.95 6.95 0 0 1-.587 1.416l.834 1.25a1 1 0 0 1-.125 1.262l-.962.962a1 1 0 0 1-1.262.125l-1.25-.834a6.953 6.953 0 0 1-1.416.587l-.294 1.473a1 1 0 0 1-.98.804H9.32a1 1 0 0 1-.98-.804l-.295-1.473a6.957 6.957 0 0 1-1.416-.587l-1.25.834a1 1 0 0 1-1.262-.125l-.962-.962a1 1 0 0 1-.125-1.262l.834-1.25a6.957 6.957 0 0 1-.587-1.416l-1.473-.294A1 1 0 0 1 1 10.68V9.32a1 1 0 0 1 .804-.98l1.473-.295c.144-.497.342-.971.587-1.416l-.834-1.25a1 1 0 0 1 .125-1.262l.962-.962A1 1 0 0 1 5.38 3.03l1.25.834a6.957 6.957 0 0 1 1.416-.587l.294-1.473ZM10 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
					</svg>
					{T('Configure')}
				</button>
			{/if}
		</div>
	</div>

	<!-- Catalog (edit mode) -->
	{#if editing && showCatalog}
		<div class="mb-3 p-3 rounded-xl border border-dashed border-gray-300 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-850/30">
			<div class="text-xs font-medium text-gray-600 dark:text-gray-300 mb-2">{T('Add widgets')}</div>
			<div class="space-y-2">
				{#each catalogByCat as group}
					{#if group.items.length}
						<div>
							<div class="text-[10px] uppercase tracking-wide text-gray-400 dark:text-gray-500 mb-1">{T(group.cat)}</div>
							<div class="flex flex-wrap gap-1.5">
								{#each group.items as w}
									<button
										class="group px-2.5 py-1 text-[11px] rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 text-gray-600 dark:text-gray-300 hover:border-blue-400 hover:text-blue-600 dark:hover:text-blue-400 transition flex items-center gap-1"
										title={w.description}
										on:click={() => addWidget(w.type)}
									>
										<span class="text-blue-500 group-hover:text-blue-600">+</span>{T(w.title)}
									</button>
								{/each}
							</div>
						</div>
					{/if}
				{/each}
			</div>
		</div>
	{/if}

	{#if loading}
		<div class="flex justify-center my-10"><Spinner className="size-5" /></div>
	{/if}

	<!-- Grid. NB: use class: directives (classList-based) — never an interpolated
	     class="" attribute here, which would rewrite className on every `editing`
	     change and clobber the classes gridstack adds to this element. -->
	<div class="grid-stack" bind:this={gridEl} class:is-editing={editing} class:invisible={loading}>
		{#each layout as item (item.id)}
			{@const def = WIDGET_BY_TYPE[item.type]}
			<div class="grid-stack-item" use:gsItem={{ item, def }} bind:this={els[item.id]}>
				<div class="grid-stack-item-content">
					{#if def}
						<WidgetCard
							title={def.bare ? '' : T(def.title)}
							href={def.link ? def.link(companyId) : null}
							linkLabel={def.linkLabel ? T(def.linkLabel) : ''}
							padded={!def.flush}
							on:remove={() => removeWidget(item)}
						>
							<svelte:component this={def.component} {companyId} options={item.options ?? def.options ?? {}} />
						</WidgetCard>
					{/if}
				</div>
			</div>
		{/each}
	</div>
</div>

<style>
	/* Let widget cards fill the gridstack item content box. */
	:global(.grid-stack-item-content) {
		inset: 0;
		overflow: visible;
	}
	/* Remove button + edit-only chrome are hidden unless the grid is editing. */
	:global(.grid-stack:not(.is-editing) .wcard-remove) {
		display: none;
	}
	/* Edit-mode affordances: draggable cursor, no text selection, dashed outline. */
	.grid-stack.is-editing :global(.grid-stack-item) {
		cursor: grab;
	}
	.grid-stack.is-editing :global(.grid-stack-item .grid-stack-item-content) {
		cursor: grab;
		user-select: none;
		-webkit-user-select: none;
	}
	.grid-stack.is-editing :global(.grid-stack-item.ui-draggable-dragging),
	.grid-stack.is-editing :global(.grid-stack-item.ui-draggable-dragging .grid-stack-item-content) {
		cursor: grabbing;
	}
	.grid-stack.is-editing :global(.wcard) {
		outline: 1px dashed rgba(59, 130, 246, 0.45);
		outline-offset: -1px;
	}
	.grid-stack.is-editing :global(.wcard-link) {
		display: none;
	}
	/* Resize handles — visible only while editing, with a grabbable hit area. */
	.grid-stack.is-editing :global(.ui-resizable-handle) {
		opacity: 1;
		display: block;
	}
	.grid-stack.is-editing :global(.ui-resizable-se) {
		z-index: 21;
		width: 18px;
		height: 18px;
		right: 3px;
		bottom: 3px;
		background-image: none;
		border-right: 2px solid rgba(59, 130, 246, 0.75);
		border-bottom: 2px solid rgba(59, 130, 246, 0.75);
		border-bottom-right-radius: 5px;
		cursor: se-resize;
	}
	/* Placeholder shown while dragging. */
	:global(.grid-stack .grid-stack-placeholder > .placeholder-content) {
		border: 1px dashed rgba(59, 130, 246, 0.5);
		border-radius: 0.75rem;
		background: rgba(59, 130, 246, 0.06);
	}
</style>
