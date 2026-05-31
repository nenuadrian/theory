/**
 * Browser-side manager for the Pyodide worker.
 *
 * One worker per page (Starlight does full-page navigations, so this is also one
 * worker per article). Every <PyCell> on a page shares it — which means Pyodide
 * downloads once and all cells share a single Python namespace, exactly like the
 * cells of a notebook.
 */
import * as Comlink from 'comlink';
import type { PyodideWorkerApi, RunResult, StatusUpdate, StreamCb } from './pyodide.worker';

let client: Comlink.Remote<PyodideWorkerApi> | null = null;
let readyPromise: Promise<void> | null = null;

let currentStatus: StatusUpdate = { phase: 'download', message: '' };
const statusListeners = new Set<(s: StatusUpdate) => void>();

function broadcast(s: StatusUpdate) {
	currentStatus = s;
	for (const l of statusListeners) l(s);
}

function getClient(): Comlink.Remote<PyodideWorkerApi> {
	if (!client) {
		const worker = new Worker(new URL('./pyodide.worker.ts', import.meta.url), {
			type: 'module',
		});
		client = Comlink.wrap<PyodideWorkerApi>(worker);
	}
	return client;
}

/** Subscribe to boot/status updates. Returns an unsubscribe function. */
export function subscribeStatus(cb: (s: StatusUpdate) => void): () => void {
	statusListeners.add(cb);
	return () => statusListeners.delete(cb);
}

export function getStatus(): StatusUpdate {
	return currentStatus;
}

export function isReady(): boolean {
	return currentStatus.phase === 'ready';
}

/** Boot Pyodide if it isn't already booting/booted. Idempotent and shared. */
export function ensureReady(): Promise<void> {
	if (!readyPromise) {
		broadcast({ phase: 'download', message: 'Downloading the Python runtime…' });
		readyPromise = getClient()
			.init(Comlink.proxy((s: StatusUpdate) => broadcast(s)))
			.then(() => broadcast({ phase: 'ready', message: 'Python is ready.' }));
	}
	return readyPromise;
}

export interface RunHandlers {
	onStdout?: StreamCb;
	onStderr?: StreamCb;
}

export async function runPython(code: string, handlers: RunHandlers = {}): Promise<RunResult> {
	await ensureReady();
	// Callbacks must be passed positionally: Comlink only transfers proxied
	// functions when they are top-level arguments, not nested inside an object.
	return getClient().run(
		code,
		handlers.onStdout ? Comlink.proxy(handlers.onStdout) : undefined,
		handlers.onStderr ? Comlink.proxy(handlers.onStderr) : undefined,
	);
}

export type { RunResult, StatusUpdate } from './pyodide.worker';
