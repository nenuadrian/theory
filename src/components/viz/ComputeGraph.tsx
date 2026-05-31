import { useEffect, useState } from 'react';

/**
 * Interactive computational graph for a single neuron with an MSE loss:
 *
 *     n = w·x → q = n + b → a = tanh(q) → L = (a − y)²
 *
 * Press "run" to sweep the forward pass left→right (values), then the backward
 * pass right→left (gradients). Drag the inputs and watch every value — and every
 * gradient — update live. This is backprop, made mechanical.
 */

type Id = 'w' | 'x' | 'b' | 'y' | 'n' | 'q' | 'a' | 'L';

interface NodeDef {
	x: number;
	y: number;
	label: string;
	kind: 'leaf' | 'op';
	w: number;
}
const H = 46;
const NODES: Record<Id, NodeDef> = {
	w: { x: 24, y: 22, label: 'w', kind: 'leaf', w: 78 },
	x: { x: 24, y: 92, label: 'x', kind: 'leaf', w: 78 },
	b: { x: 24, y: 182, label: 'b', kind: 'leaf', w: 78 },
	y: { x: 24, y: 300, label: 'y', kind: 'leaf', w: 78 },
	n: { x: 210, y: 56, label: 'n = w·x', kind: 'op', w: 132 },
	q: { x: 372, y: 120, label: 'q = n + b', kind: 'op', w: 132 },
	a: { x: 524, y: 184, label: 'a = tanh q', kind: 'op', w: 132 },
	L: { x: 656, y: 250, label: 'L = (a−y)²', kind: 'op', w: 132 },
};
const EDGES: [Id, Id][] = [
	['w', 'n'],
	['x', 'n'],
	['n', 'q'],
	['b', 'q'],
	['q', 'a'],
	['a', 'L'],
	['y', 'L'],
];

const FORWARD: Id[] = ['w', 'x', 'n', 'b', 'q', 'a', 'y', 'L'];
const BACKWARD: Id[] = ['L', 'y', 'a', 'q', 'b', 'n', 'x', 'w'];

const rightAnchor = (n: NodeDef) => [n.x + n.w, n.y + H / 2] as const;
const leftAnchor = (n: NodeDef) => [n.x, n.y + H / 2] as const;

function edgePath(a: Id, b: Id) {
	const [x1, y1] = rightAnchor(NODES[a]);
	const [x2, y2] = leftAnchor(NODES[b]);
	const dx = Math.max(28, (x2 - x1) / 2);
	return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`;
}

function Slider({ label, value, set }: { label: string; value: number; set: (v: number) => void }) {
	return (
		<label className="viz__slider">
			<span className="viz__slider-label">
				{label} <b>{value.toFixed(2)}</b>
			</span>
			<input
				type="range"
				min={-2}
				max={2}
				step={0.05}
				value={value}
				onChange={(e) => set(parseFloat(e.target.value))}
			/>
		</label>
	);
}

export default function ComputeGraph() {
	const [w, setW] = useState(0.9);
	const [x, setX] = useState(1.0);
	const [b, setB] = useState(-0.4);
	const [y, setY] = useState(0.5);

	// phase: idle | forward | backward | done ; idx = sweep position
	const [phase, setPhase] = useState<'idle' | 'forward' | 'backward' | 'done'>('idle');
	const [idx, setIdx] = useState(-1);

	// Forward values
	const nVal = w * x;
	const qVal = nVal + b;
	const aVal = Math.tanh(qVal);
	const diff = aVal - y;
	const LVal = diff * diff;
	const vals: Record<Id, number> = { w, x, b, y, n: nVal, q: qVal, a: aVal, L: LVal };

	// Backward gradients (chain rule, by hand)
	const qg = 2 * diff * (1 - aVal * aVal);
	const grads: Record<Id, number> = {
		L: 1,
		a: 2 * diff,
		y: -2 * diff,
		q: qg,
		n: qg,
		b: qg,
		w: qg * x,
		x: qg * w,
	};

	// Animation driver
	useEffect(() => {
		if (phase !== 'forward' && phase !== 'backward') return;
		const order = phase === 'forward' ? FORWARD : BACKWARD;
		if (idx >= order.length - 1) {
			const t = setTimeout(() => {
				if (phase === 'forward') {
					setPhase('backward');
					setIdx(-1);
				} else {
					setPhase('done');
					setIdx(-1);
				}
			}, 500);
			return () => clearTimeout(t);
		}
		const t = setTimeout(() => setIdx((i) => i + 1), phase === 'forward' ? 380 : 480);
		return () => clearTimeout(t);
	}, [phase, idx]);

	const run = () => {
		setPhase('forward');
		setIdx(-1);
	};
	const reset = () => {
		setPhase('idle');
		setIdx(-1);
	};

	const activeId = phase === 'forward' ? FORWARD[idx] : phase === 'backward' ? BACKWARD[idx] : null;
	const gradVisible = (id: Id) =>
		phase === 'done' || (phase === 'backward' && BACKWARD.indexOf(id) <= idx);
	const edgeActive = (a: Id, bb: Id) =>
		(phase === 'forward' && activeId === bb) || (phase === 'backward' && activeId === a);

	return (
		<div className="viz">
			<div className="viz__row">
				<svg
					className="cg"
					viewBox="0 0 800 360"
					role="img"
					aria-label="Computational graph of a neuron and its loss"
				>
					{EDGES.map(([a, bb]) => (
						<path
							key={`${a}-${bb}`}
							d={edgePath(a, bb)}
							className={`cg__edge${edgeActive(a, bb) ? ' cg__edge--active' : ''}`}
							fill="none"
						/>
					))}
					{(Object.keys(NODES) as Id[]).map((id) => {
						const n = NODES[id];
						const active = activeId === id;
						const showGrad = gradVisible(id);
						return (
							<g key={id} transform={`translate(${n.x},${n.y})`}>
								<rect
									width={n.w}
									height={H}
									className={`cg__node cg__node--${n.kind}${active ? ' cg__node--active' : ''}`}
								/>
								<text x={8} y={17} className="cg__label">
									{n.label}
								</text>
								<text x={8} y={36} className="cg__val">
									{vals[id].toFixed(2)}
								</text>
								{showGrad && (
									<text x={n.w - 8} y={36} textAnchor="end" className="cg__grad">
										∂L/∂{id}={grads[id].toFixed(2)}
									</text>
								)}
							</g>
						);
					})}
				</svg>

				<div className="viz__controls">
					<Slider label="w" value={w} set={setW} />
					<Slider label="x" value={x} set={setX} />
					<Slider label="b" value={b} set={setB} />
					<Slider label="y" value={y} set={setY} />
					<div className="cg__buttons">
						<button
							type="button"
							className="viz__btn"
							onClick={run}
							disabled={phase === 'forward' || phase === 'backward'}
						>
							▸ run
						</button>
						<button type="button" className="viz__btn" onClick={reset}>
							reset
						</button>
					</div>
					<p className="cg__caption">
						{phase === 'forward' && 'Forward: values flow →'}
						{phase === 'backward' && 'Backward: gradients flow ←'}
						{phase === 'done' && 'Drag any input — gradients update live.'}
						{phase === 'idle' && 'Press run to sweep forward, then backward.'}
					</p>
				</div>
			</div>
		</div>
	);
}
