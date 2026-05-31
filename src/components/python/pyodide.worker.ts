/// <reference lib="webworker" />
/**
 * Pyodide host worker.
 *
 * Runs CPython (compiled to WASM) off the main thread so heavy work — training
 * loops, Matplotlib rendering — never freezes the page. Exposed to the main
 * thread via Comlink. Pyodide itself is loaded from the jsDelivr CDN so we don't
 * bundle ~10MB of WASM through Vite.
 */
import * as Comlink from 'comlink';

const PYODIDE_VERSION = '0.29.4';
const INDEX_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

export interface StatusUpdate {
	phase: 'download' | 'setup' | 'packages' | 'ready';
	message: string;
}
export type StreamCb = (chunk: string) => void;
export type StatusCb = (s: StatusUpdate) => void;

export interface RunResult {
	ok: boolean;
	/** Rich representation of the cell's last expression, notebook-style. */
	result?: { html?: string; text?: string } | null;
	/** base64-encoded PNGs of any Matplotlib figures left open by the cell. */
	images?: string[];
	/** Formatted Python traceback, if the cell raised. */
	error?: string;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pyodide: any = null;
let initPromise: Promise<void> | null = null;

// Per-run stream targets, set for the duration of a single run() call.
let outCb: StreamCb | null = null;
let errCb: StreamCb | null = null;

// Python-side helpers installed once at startup.
const SETUP = `
import sys, io, base64

def __learn_format_result(obj):
    if obj is None:
        return None
    html = getattr(obj, "_repr_html_", None)
    if callable(html):
        try:
            return {"html": html()}
        except Exception:
            pass
    try:
        return {"text": repr(obj)}
    except Exception:
        return {"text": str(obj)}

def __learn_capture_figures():
    figs = []
    if "matplotlib" in sys.modules:
        try:
            import matplotlib.pyplot as plt
            for num in plt.get_fignums():
                fig = plt.figure(num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                figs.append(base64.b64encode(buf.getvalue()).decode("ascii"))
            plt.close("all")
        except Exception:
            pass
    return figs
`;

async function doInit(onStatus?: StatusCb): Promise<void> {
	onStatus?.({ phase: 'download', message: 'Downloading the Python runtime…' });
	const { loadPyodide } = await import(/* @vite-ignore */ `${INDEX_URL}pyodide.mjs`);
	pyodide = await loadPyodide({ indexURL: INDEX_URL });

	onStatus?.({ phase: 'setup', message: 'Setting up the environment…' });
	pyodide.setStdout({ batched: (s: string) => outCb?.(s) });
	pyodide.setStderr({ batched: (s: string) => errCb?.(s) });
	// Force a headless Matplotlib backend before pyplot is ever imported.
	await pyodide.runPythonAsync(`import os; os.environ['MPLBACKEND'] = 'AGG'`);
	await pyodide.runPythonAsync(SETUP);
	onStatus?.({ phase: 'ready', message: 'Python is ready.' });
}

const api = {
	version: PYODIDE_VERSION,

	async init(onStatus?: StatusCb): Promise<void> {
		if (!initPromise) initPromise = doInit(onStatus);
		return initPromise;
	},

	async loadPackages(names: string[]): Promise<void> {
		await this.init();
		if (names.length) await pyodide.loadPackage(names);
	},

	async run(
		code: string,
		onStdout?: StreamCb,
		onStderr?: StreamCb,
	): Promise<RunResult> {
		if (!pyodide) throw new Error('Pyodide is not initialized.');
		// Note: Comlink only transfers proxied callbacks as *top-level* arguments,
		// so onStdout/onStderr are passed positionally rather than in an options object.
		outCb = onStdout ?? null;
		errCb = onStderr ?? null;

		const response: RunResult = { ok: true };
		try {
			// Auto-install any importable packages the cell references (numpy, etc.).
			// Silence the loader's "already loaded"/"no new packages" chatter so the
			// cell output shows only what the user's code actually printed.
			await pyodide.loadPackagesFromImports(code, {
				messageCallback: () => {},
				errorCallback: () => {},
			});
			const res = await pyodide.runPythonAsync(code);

			if (res !== undefined && res !== null) {
				if (pyodide.isPyProxy(res)) {
					const fmt = pyodide.globals.get('__learn_format_result');
					const out = fmt(res);
					response.result = out
						? out.toJs({ dict_converter: Object.fromEntries })
						: null;
					out?.destroy?.();
					fmt.destroy();
					res.destroy?.();
				} else {
					response.result = { text: String(res) };
				}
			}

			const figs = pyodide.runPython('__learn_capture_figures()');
			const figsJs: string[] = figs.toJs();
			figs.destroy();
			if (figsJs && figsJs.length) response.images = figsJs;
		} catch (e: unknown) {
			response.ok = false;
			response.error = e instanceof Error ? e.message : String(e);
		} finally {
			outCb = null;
			errCb = null;
		}
		return response;
	},
};

export type PyodideWorkerApi = typeof api;
Comlink.expose(api);
