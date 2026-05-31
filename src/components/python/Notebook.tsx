import { useCallback, useState } from 'react';
import PyCell from './PyCell';

/**
 * A lightweight, on-brand notebook: a stack of PyCells that all share one
 * Pyodide kernel (the singleton worker), so state defined in one cell is
 * available in the next — exactly like a Jupyter notebook. Add and remove cells
 * freely. Edits persist in localStorage per cell id.
 */

export interface NotebookProps {
	/** Starter cells. */
	cells: { id: string; code: string; title?: string }[];
}

let added = 0;

export default function Notebook({ cells: starter }: NotebookProps) {
	const [cells, setCells] = useState(starter);

	const addCell = useCallback(() => {
		added += 1;
		setCells((cs) => [...cs, { id: `scratch-${added}`, code: '', title: 'cell' }]);
	}, []);

	const removeCell = useCallback((id: string) => {
		setCells((cs) => cs.filter((c) => c.id !== id));
		try {
			localStorage.removeItem(`learn:pycell:nb:${id}`);
		} catch {
			/* ignore */
		}
	}, []);

	return (
		<div className="notebook">
			<div className="notebook__bar">
				<span className="notebook__title">notebook</span>
				<span className="notebook__hint">cells share one Python kernel</span>
			</div>

			{cells.map((c, i) => (
				<div className="notebook__cell" key={c.id}>
					<PyCell id={`nb:${c.id}`} code={c.code} title={c.title ?? `[${i + 1}]`} />
					{cells.length > 1 && (
						<button
							type="button"
							className="notebook__del"
							onClick={() => removeCell(c.id)}
							title="Delete this cell"
							aria-label="Delete cell"
						>
							✕
						</button>
					)}
				</div>
			))}

			<button type="button" className="viz__btn notebook__add" onClick={addCell}>
				+ add cell
			</button>
		</div>
	);
}
