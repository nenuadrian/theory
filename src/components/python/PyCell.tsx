import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { python } from '@codemirror/lang-python';
import { githubDark, githubLight } from '@uiw/codemirror-theme-github';
import { keymap } from '@codemirror/view';
import { Prec } from '@codemirror/state';
import {
	runPython,
	subscribeStatus,
	isReady,
	type RunResult,
} from './pyodideClient';
import PyOutput from './PyOutput';

export interface PyCellProps {
	/** Initial Python source. */
	code: string;
	/** Stable id — used as the localStorage key for the user's edits. */
	id: string;
	/** Label shown in the cell's title bar. */
	title?: string;
	/** Run automatically once, on first scroll into view. */
	autoRun?: boolean;
	/** Make the editor read-only. */
	readOnly?: boolean;
}

const storageKey = (id: string) => `learn:pycell:${id}`;

export default function PyCell({
	code,
	id,
	title = 'python',
	autoRun = false,
	readOnly = false,
}: PyCellProps) {
	const initial = code.replace(/\n+$/, '');
	const [source, setSource] = useState(initial);
	const [isDark, setIsDark] = useState(true);

	const [running, setRunning] = useState(false);
	const [hasRun, setHasRun] = useState(false);
	const [booting, setBooting] = useState(false);
	const [statusMsg, setStatusMsg] = useState('');

	const [consoleText, setConsoleText] = useState('');
	const [result, setResult] = useState<RunResult['result']>(null);
	const [images, setImages] = useState<string[] | undefined>(undefined);
	const [error, setError] = useState<string | undefined>(undefined);

	const sourceRef = useRef(source);
	const runningRef = useRef(false);
	const rootRef = useRef<HTMLDivElement | null>(null);

	// Restore the reader's saved edits for this cell.
	useEffect(() => {
		try {
			const saved = localStorage.getItem(storageKey(id));
			if (saved !== null) {
				setSource(saved);
				sourceRef.current = saved;
			}
		} catch {
			/* localStorage may be unavailable; ignore. */
		}
	}, [id]);

	// Track the site's light/dark theme so the editor matches the page.
	useEffect(() => {
		const read = () =>
			setIsDark((document.documentElement.dataset.theme ?? 'dark') !== 'light');
		read();
		const obs = new MutationObserver(read);
		obs.observe(document.documentElement, {
			attributes: true,
			attributeFilter: ['data-theme'],
		});
		return () => obs.disconnect();
	}, []);

	// Reflect shared boot progress while Pyodide downloads.
	useEffect(
		() =>
			subscribeStatus((s) =>
				setStatusMsg(s.phase === 'ready' ? '' : s.message),
			),
		[],
	);

	const run = useCallback(async () => {
		if (runningRef.current) return;
		runningRef.current = true;
		setRunning(true);
		setHasRun(true);
		setConsoleText('');
		setResult(null);
		setImages(undefined);
		setError(undefined);
		if (!isReady()) setBooting(true);

		let buf = '';
		const append = (chunk: string) => {
			buf += chunk;
			setConsoleText(buf);
		};

		try {
			const res = await runPython(sourceRef.current, {
				onStdout: append,
				onStderr: append,
			});
			setResult(res.result ?? null);
			setImages(res.images);
			setError(res.error);
		} catch (e) {
			setError(e instanceof Error ? e.message : String(e));
		} finally {
			runningRef.current = false;
			setRunning(false);
			setBooting(false);
		}
	}, []);

	// Keep a live ref so the Shift-Enter keymap always calls the latest run().
	const runRef = useRef(run);
	useEffect(() => {
		runRef.current = run;
	}, [run]);

	// Auto-run once when scrolled into view.
	useEffect(() => {
		if (!autoRun || !rootRef.current) return;
		const el = rootRef.current;
		const obs = new IntersectionObserver(
			(entries) => {
				if (entries.some((e) => e.isIntersecting)) {
					obs.disconnect();
					void runRef.current();
				}
			},
			{ rootMargin: '0px 0px -20% 0px' },
		);
		obs.observe(el);
		return () => obs.disconnect();
	}, [autoRun]);

	const onChange = useCallback(
		(val: string) => {
			setSource(val);
			sourceRef.current = val;
			try {
				localStorage.setItem(storageKey(id), val);
			} catch {
				/* ignore */
			}
		},
		[id],
	);

	const reset = useCallback(() => {
		setSource(initial);
		sourceRef.current = initial;
		setConsoleText('');
		setResult(null);
		setImages(undefined);
		setError(undefined);
		setHasRun(false);
		try {
			localStorage.removeItem(storageKey(id));
		} catch {
			/* ignore */
		}
	}, [id, initial]);

	const extensions = useMemo(
		() => [
			python(),
			Prec.highest(
				keymap.of([
					{
						key: 'Shift-Enter',
						run: () => {
							void runRef.current();
							return true;
						},
					},
				]),
			),
		],
		[],
	);

	return (
		<div className="pycell" ref={rootRef} data-running={running}>
			<div className="pycell__bar">
				<span className="pycell__label">{title}</span>
				<span className="pycell__spacer" />
				{booting && (
					<span className="pycell__status">{statusMsg || 'Booting Python…'}</span>
				)}
				<button
					type="button"
					className="pycell__btn"
					onClick={reset}
					disabled={running}
					title="Restore the original code"
				>
					reset
				</button>
				<button
					type="button"
					className="pycell__btn pycell__btn--run"
					onClick={() => void run()}
					disabled={running}
					title="Run (Shift+Enter)"
				>
					{running ? '…running' : '▸ run'}
				</button>
			</div>

			<CodeMirror
				value={source}
				theme={isDark ? githubDark : githubLight}
				extensions={extensions}
				editable={!readOnly}
				basicSetup={{
					lineNumbers: true,
					foldGutter: false,
					highlightActiveLine: false,
					highlightActiveLineGutter: false,
					highlightSelectionMatches: false,
				}}
				onChange={onChange}
			/>

			<PyOutput
				consoleText={consoleText}
				result={result}
				images={images}
				error={error}
				hasRun={hasRun}
				running={running}
			/>
		</div>
	);
}
