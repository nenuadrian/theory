import type { RunResult } from './pyodideClient';

interface Props {
	consoleText: string;
	result?: RunResult['result'];
	images?: string[];
	error?: string;
	hasRun: boolean;
	running: boolean;
}

export default function PyOutput({
	consoleText,
	result,
	images,
	error,
	hasRun,
	running,
}: Props) {
	const empty = !consoleText && !result && !images?.length && !error;
	if (!hasRun && !running) return null;

	return (
		<div className="pycell__output" aria-live="polite">
			{consoleText && <pre className="pycell__stream">{consoleText}</pre>}

			{images?.map((b64, i) => (
				<img
					key={i}
					className="pycell__image"
					src={`data:image/png;base64,${b64}`}
					alt="Figure output"
				/>
			))}

			{result?.html && (
				<div
					className="pycell__richhtml"
					// Rich reprs (DataFrames, etc.) come from the user's own code.
					dangerouslySetInnerHTML={{ __html: result.html }}
				/>
			)}
			{result?.text && <pre className="pycell__result">{result.text}</pre>}

			{error && <pre className="pycell__error">{error}</pre>}

			{running && empty && <div className="pycell__hint">Running…</div>}
			{!running && empty && hasRun && (
				<div className="pycell__hint">Done — no output.</div>
			)}
		</div>
	);
}
