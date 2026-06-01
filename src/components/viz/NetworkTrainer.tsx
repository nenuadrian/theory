import { useEffect, useMemo, useState } from 'react';

/**
 * A complete neural network, trained step by step.
 *
 *   2 → 3 → 1  MLP   ·   tanh hidden, sigmoid output   ·   L = (ŷ − y)²
 *
 * It learns XOR by per-example stochastic gradient descent. Every training
 * iteration is broken into nine inspectable phases —
 *
 *   load · forward(z¹,a¹,z²,ŷ) · loss · backward(output, hidden) · update
 *
 * — and at each phase the panel prints the *exact arithmetic*, with the real
 * numbers substituted, that produced every value and every gradient on the
 * diagram. Step forward and back, play, or scrub across all 600 iterations and
 * watch the decision boundary resolve into the XOR checkerboard.
 *
 * The whole training run is replayed once (deterministically) into a snapshot
 * per iteration, so any (iteration, phase) is a pure derivation — scrubbing and
 * back-stepping are exact, never an approximation.
 */

// ---- problem & fixed initialization (seed-18 NumPy draw, see prototype) -----
const X: [number, number][] = [[0, 0], [0, 1], [1, 0], [1, 1]];
const Y = [0, 1, 1, 0];
const LR = 1.0;
const ITERS = 600;

const W1_0: number[][] = [
	[-0.43234, -1.130097],
	[0.673844, -1.107815],
	[2.013916, 0.924112],
];
const b1_0 = [-0.359263, 0.570516, 1.611589];
const W2_0 = [2.833969, -0.923265, 1.065074];
const b2_0 = 0.517741;

interface Params {
	W1: number[][]; // 3×2
	b1: number[]; // 3
	W2: number[]; // 3
	b2: number; // 1
}
const clone = (P: Params): Params => ({
	W1: P.W1.map((r) => r.slice()),
	b1: P.b1.slice(),
	W2: P.W2.slice(),
	b2: P.b2,
});

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function forward(P: Params, x: [number, number]) {
	const z1 = P.W1.map((row, j) => row[0] * x[0] + row[1] * x[1] + P.b1[j]);
	const a1 = z1.map(Math.tanh);
	const z2 = P.W2[0] * a1[0] + P.W2[1] * a1[1] + P.W2[2] * a1[2] + P.b2;
	const yhat = sigmoid(z2);
	return { z1, a1, z2, yhat };
}
function backward(P: Params, x: [number, number], y: number, F: ReturnType<typeof forward>) {
	const d2 = 2 * (F.yhat - y) * F.yhat * (1 - F.yhat); // ∂L/∂z²
	const dW2 = F.a1.map((a) => d2 * a); // 3
	const db2 = d2;
	const d1 = P.W2.map((w, j) => w * d2 * (1 - F.a1[j] * F.a1[j])); // 3
	const dW1 = d1.map((dj) => [dj * x[0], dj * x[1]]); // 3×2
	const db1 = d1.slice();
	return { d2, dW2, db2, d1, dW1, db1 };
}
function descend(P: Params, G: ReturnType<typeof backward>): Params {
	return {
		W1: P.W1.map((row, j) => [row[0] - LR * G.dW1[j][0], row[1] - LR * G.dW1[j][1]]),
		b1: P.b1.map((b, j) => b - LR * G.db1[j]),
		W2: P.W2.map((w, j) => w - LR * G.dW2[j]),
		b2: P.b2 - LR * G.db2,
	};
}

// ---- phases -----------------------------------------------------------------
type PhaseKey =
	| 'input' | 'fwd1z' | 'fwd1a' | 'fwd2z' | 'fwd2a'
	| 'loss' | 'bwd2' | 'bwd1' | 'update';
const PHASES: { key: PhaseKey; title: string }[] = [
	{ key: 'input', title: 'Load example' },
	{ key: 'fwd1z', title: 'Forward — hidden pre-activations  z⁽¹⁾' },
	{ key: 'fwd1a', title: 'Forward — hidden activations  a⁽¹⁾' },
	{ key: 'fwd2z', title: 'Forward — output pre-activation  z⁽²⁾' },
	{ key: 'fwd2a', title: 'Forward — output activation  ŷ' },
	{ key: 'loss', title: 'Loss' },
	{ key: 'bwd2', title: 'Backward — output layer' },
	{ key: 'bwd1', title: 'Backward — hidden layer' },
	{ key: 'update', title: 'Gradient-descent update' },
];
const LAST = PHASES.length - 1;

// ---- node layout (SVG viewBox 0 0 640 320) ----------------------------------
const R = 22;
const POS = {
	x0: { x: 66, y: 116 }, x1: { x: 66, y: 212 },
	h0: { x: 258, y: 64 }, h1: { x: 258, y: 160 }, h2: { x: 258, y: 256 },
	o: { x: 452, y: 160 }, L: { x: 580, y: 160 },
} as const;
type NodeId = keyof typeof POS;
const IN_H: { k: 0 | 1; j: 0 | 1 | 2 }[] = [
	{ k: 0, j: 0 }, { k: 0, j: 1 }, { k: 0, j: 2 },
	{ k: 1, j: 0 }, { k: 1, j: 1 }, { k: 1, j: 2 },
];

const f = (v: number, d = 3) => v.toFixed(d);
const sgn = (v: number) => (v < 0 ? '−' : '');
const abs = (v: number, d = 3) => Math.abs(v).toFixed(d);
// signed value as it should read inside an expression, e.g. (−0.432)
const par = (v: number, d = 3) => `(${v < 0 ? '−' : ''}${Math.abs(v).toFixed(d)})`;
// a squared term that stays unambiguous when negative: 0.52² or (−0.34)²
const sq = (v: number, d = 2) => (v < 0 ? `(−${Math.abs(v).toFixed(d)})²` : `${v.toFixed(d)}²`);

// diverging colour for the decision-boundary heat cells
function heatColor(p: number) {
	// p<0.5 → indigo, p>0.5 → amber, intensity ∝ |p−0.5|
	const t = (p - 0.5) * 2; // −1..1
	if (t >= 0) {
		const a = 0.12 + 0.78 * t;
		return `rgba(251,146,60,${a.toFixed(3)})`;
	}
	const a = 0.12 + 0.78 * -t;
	return `rgba(99,102,241,${a.toFixed(3)})`;
}

export default function NetworkTrainer() {
	const [iter, setIter] = useState(0);
	const [phase, setPhase] = useState(0);
	const [running, setRunning] = useState(false);

	// Replay the whole run once; snapshot params at the start of each iteration.
	const { snaps, losses } = useMemo(() => {
		let P: Params = { W1: W1_0.map((r) => r.slice()), b1: b1_0.slice(), W2: W2_0.slice(), b2: b2_0 };
		const snaps: Params[] = [];
		const losses: number[] = [];
		for (let it = 0; it < ITERS; it++) {
			snaps.push(clone(P));
			const x = X[it % 4];
			const y = Y[it % 4];
			const F = forward(P, x);
			losses.push((F.yhat - y) * (F.yhat - y));
			P = descend(P, backward(P, x, y, F));
		}
		return { snaps, losses };
	}, []);

	const P = snaps[iter];
	const i = iter % 4;
	const x = X[i];
	const y = Y[i];
	const F = useMemo(() => forward(P, x), [P, x]);
	const G = useMemo(() => backward(P, x, y, F), [P, x, y, F]);
	const Pnext = snaps[iter + 1] ?? descend(P, G);

	const pk = PHASES[phase].key;
	const fwdReached = (p: PhaseKey) => PHASES.findIndex((q) => q.key === p) <= phase;

	// ---- step / play / reset ----
	const stepFwd = () => {
		if (phase < LAST) setPhase(phase + 1);
		else if (iter < ITERS - 1) {
			setIter(iter + 1);
			setPhase(0);
		} else setRunning(false);
	};
	const stepBack = () => {
		if (phase > 0) setPhase(phase - 1);
		else if (iter > 0) {
			setIter(iter - 1);
			setPhase(LAST);
		}
	};
	useEffect(() => {
		if (!running) return;
		if (iter >= ITERS - 1 && phase >= LAST) {
			setRunning(false);
			return;
		}
		const t = setTimeout(stepFwd, 520);
		return () => clearTimeout(t);
	}, [running, iter, phase]);

	// ---- which value a node currently shows ----
	const nodeValue = (id: NodeId): string => {
		if (id === 'x0') return f(x[0], 0);
		if (id === 'x1') return f(x[1], 0);
		if (id === 'L') return fwdReached('loss') ? f((F.yhat - y) ** 2, 3) : '·';
		if (id === 'o') {
			if (fwdReached('fwd2a')) return f(F.yhat, 2);
			if (fwdReached('fwd2z')) return f(F.z2, 2);
			return '·';
		}
		const j = id === 'h0' ? 0 : id === 'h1' ? 1 : 2;
		if (fwdReached('fwd1a')) return f(F.a1[j], 2);
		if (fwdReached('fwd1z')) return f(F.z1[j], 2);
		return '·';
	};
	// gradient δ shown beneath a node once backward has reached it
	const nodeGrad = (id: NodeId): string | null => {
		if (id === 'o' && fwdReached('bwd2')) return f(G.d2, 3);
		if ((id === 'h0' || id === 'h1' || id === 'h2') && fwdReached('bwd1')) {
			const j = id === 'h0' ? 0 : id === 'h1' ? 1 : 2;
			return f(G.d1[j], 3);
		}
		return null;
	};

	// ---- which nodes / edges are active this phase ----
	const activeNodes = new Set<NodeId>();
	const inHActive = pk === 'fwd1z' || pk === 'bwd1' || pk === 'update';
	const hOActive = pk === 'fwd2z' || pk === 'bwd2' || pk === 'update';
	const oLActive = pk === 'fwd2a' || pk === 'loss' || pk === 'bwd2';
	if (pk === 'input') (['x0', 'x1'] as NodeId[]).forEach((n) => activeNodes.add(n));
	if (pk === 'fwd1z' || pk === 'fwd1a' || pk === 'bwd1')
		(['h0', 'h1', 'h2'] as NodeId[]).forEach((n) => activeNodes.add(n));
	if (pk === 'fwd2z' || pk === 'fwd2a' || pk === 'bwd2') activeNodes.add('o');
	if (pk === 'loss') activeNodes.add('L');

	// ---- decision-boundary heat grid from current weights ----
	const NG = 12;
	const heat = useMemo(() => {
		const cells: { gx: number; gy: number; p: number }[] = [];
		for (let r = 0; r < NG; r++)
			for (let c = 0; c < NG; c++) {
				const xx = (c + 0.5) / NG;
				const yy = 1 - (r + 0.5) / NG;
				cells.push({ gx: c, gy: r, p: forward(P, [xx, yy]).yhat });
			}
		return cells;
	}, [P]);

	// ---- loss sparkline path ----
	const spark = useMemo(() => {
		const w = 168, h = 46, n = losses.length;
		const max = Math.max(...losses);
		const pts = losses.map((l, k) => {
			const px = (k / (n - 1)) * w;
			const py = h - (l / max) * (h - 4) - 2;
			return [px, py] as const;
		});
		const d = pts.map(([px, py], k) => `${k ? 'L' : 'M'}${px.toFixed(1)} ${py.toFixed(1)}`).join(' ');
		const cur = pts[iter];
		return { w, h, d, cur };
	}, [losses, iter]);

	const center = (id: NodeId) => POS[id];
	const edge = (a: NodeId, b: NodeId, active: boolean, w: number, key: string, showW: boolean) => {
		const A = center(a), B = center(b);
		const mag = Math.min(1, Math.abs(w) / 2.5);
		const hue = w >= 0 ? '45,212,191' : '236,72,153'; // teal + / pink −
		const t = 0.5;
		const lx = A.x + (B.x - A.x) * t;
		const ly = A.y + (B.y - A.y) * t;
		return (
			<g key={key}>
				<line
					x1={A.x} y1={A.y} x2={B.x} y2={B.y}
					stroke={`rgba(${hue},${(0.22 + 0.55 * mag).toFixed(2)})`}
					strokeWidth={active ? 2.5 + 2.5 * mag : 1 + 2 * mag}
					className={active ? 'nt__edge--on' : undefined}
				/>
				{showW && (
					<>
						<rect x={lx - 17} y={ly - 9} width={34} height={15} rx={2} className="nt__wpill" />
						<text x={lx} y={ly + 2} textAnchor="middle" className="nt__wt">
							{sgn(w)}{abs(w, 2)}
						</text>
					</>
				)}
			</g>
		);
	};

	const node = (id: NodeId, label: string, kind: 'in' | 'hid' | 'out' | 'loss') => {
		const c = center(id);
		const active = activeNodes.has(id);
		const grad = nodeGrad(id);
		const r = kind === 'out' || kind === 'loss' ? R + 2 : R;
		return (
			<g key={id}>
				<circle
					cx={c.x} cy={c.y} r={r}
					className={`nt__node nt__node--${kind}${active ? ' nt__node--on' : ''}`}
				/>
				<text x={c.x} y={c.y - r - 6} textAnchor="middle" className="nt__nlabel">{label}</text>
				<text x={c.x} y={c.y + 4} textAnchor="middle" className="nt__nval">{nodeValue(id)}</text>
				{grad !== null && (
					<text x={c.x} y={c.y + r + 13} textAnchor="middle" className="nt__ngrad">δ {grad}</text>
				)}
			</g>
		);
	};

	return (
		<div className="viz">
			<div className="viz__row">
				<svg className="nt" viewBox="0 0 640 320" role="img"
					aria-label="A 2-3-1 neural network training on XOR, step by step">
					{/* layer captions */}
					<text x={66} y={26} textAnchor="middle" className="nt__cap">input</text>
					<text x={258} y={26} textAnchor="middle" className="nt__cap">hidden · tanh</text>
					<text x={452} y={26} textAnchor="middle" className="nt__cap">output · σ</text>
					<text x={580} y={26} textAnchor="middle" className="nt__cap">loss</text>

					{/* edges (drawn first, behind nodes) */}
					{IN_H.map(({ k, j }) =>
						edge(
							(k === 0 ? 'x0' : 'x1') as NodeId,
							(['h0', 'h1', 'h2'][j]) as NodeId,
							inHActive, P.W1[j][k], `e-${k}-${j}`, inHActive,
						),
					)}
					{[0, 1, 2].map((j) =>
						edge((['h0', 'h1', 'h2'][j]) as NodeId, 'o', hOActive, P.W2[j], `eo-${j}`, hOActive),
					)}
					{edge('o', 'L', oLActive, 1, 'e-oL', false)}

					{/* nodes */}
					{node('x0', 'x₁', 'in')}
					{node('x1', 'x₂', 'in')}
					{node('h0', 'h₁', 'hid')}
					{node('h1', 'h₂', 'hid')}
					{node('h2', 'h₃', 'hid')}
					{node('o', 'ŷ', 'out')}
					{node('L', 'L', 'loss')}
					{/* target badge by the loss node */}
					<text x={580} y={160 + R + 26} textAnchor="middle" className="nt__cap">target y = {y}</text>
				</svg>

				<div className="viz__controls">
					<div className="nt__phasebar">
						<span className="nt__phasekey">{phase + 1}/9</span>
						<span className="nt__phasetitle">{PHASES[phase].title}</span>
					</div>

					<div className="cg__buttons">
						<button type="button" className="viz__btn" onClick={stepBack} disabled={running || (iter === 0 && phase === 0)}>◂ back</button>
						<button type="button" className="viz__btn" onClick={stepFwd} disabled={running || (iter === ITERS - 1 && phase === LAST)}>step ▸</button>
						<button type="button" className="viz__btn" onClick={() => {
							if (iter >= ITERS - 1 && phase >= LAST) { setIter(0); setPhase(0); }
							setRunning((r) => !r);
						}}>{running ? '❚❚ pause' : '▶ play'}</button>
						<button type="button" className="viz__btn" onClick={() => { setRunning(false); setIter(0); setPhase(0); }}>reset</button>
					</div>

					<label className="viz__slider">
						<span className="viz__slider-label">iteration <b>{iter}</b> / {ITERS - 1}</span>
						<input type="range" min={0} max={ITERS - 1} step={1} value={iter}
							onChange={(e) => { setRunning(false); setIter(parseInt(e.target.value, 10)); setPhase(0); }} />
					</label>
					<p className="nt__ex">
						example {i + 1}/4 · x = ({f(x[0], 0)}, {f(x[1], 0)}) → y = {y}
						<span className="nt__loss"> · loss {f(losses[iter], 4)}</span>
					</p>

					{/* decision boundary + loss curve */}
					<div className="nt__minis">
						<div>
							<svg viewBox="0 0 120 120" className="nt__heat" aria-label="decision boundary">
								{heat.map((c) => (
									<rect key={`${c.gx}-${c.gy}`} x={(c.gx / NG) * 120} y={(c.gy / NG) * 120}
										width={120 / NG + 0.6} height={120 / NG + 0.6} fill={heatColor(c.p)} />
								))}
								{X.map((p, k) => (
									<circle key={k} cx={p[0] * 120} cy={(1 - p[1]) * 120} r={6}
										fill={Y[k] ? 'rgba(251,146,60,1)' : 'rgba(99,102,241,1)'}
										stroke="#111" strokeWidth={1.5} />
								))}
							</svg>
							<span className="nt__minilabel">ŷ over input space</span>
						</div>
						<div>
							<svg viewBox={`0 0 ${spark.w} ${spark.h}`} className="nt__spark" aria-label="loss curve">
								<path d={spark.d} fill="none" stroke="rgba(96,211,197,0.85)" strokeWidth={1.3} />
								{spark.cur && <circle cx={spark.cur[0]} cy={spark.cur[1]} r={3} fill="#fb923c" />}
							</svg>
							<span className="nt__minilabel">per-iteration loss</span>
						</div>
					</div>
				</div>
			</div>

			{/* full-width calculation panel */}
			<div className="nt__calc">{renderCalc(pk, P, Pnext, x, y, i, F, G)}</div>

			<code className="viz__formula">
				z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾ · a⁽ˡ⁾ = σ(z⁽ˡ⁾) · δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ σ′(z⁽ˡ⁾) · θ ← θ − η ∂L/∂θ
			</code>
		</div>
	);
}

// ---- the calculation panel: exact arithmetic per phase ----------------------
function row(formula: string, result?: string, key?: string | number) {
	return (
		<div className="nt__crow" key={key}>
			<span className="nt__cform">{formula}</span>
			{result !== undefined && <span className="nt__cres">{result}</span>}
		</div>
	);
}

function renderCalc(
	pk: PhaseKey, P: Params, Pn: Params, x: [number, number], y: number, i: number,
	F: ReturnType<typeof forward>, G: ReturnType<typeof backward>,
) {
	const sub = ['₁', '₂', '₃'];
	switch (pk) {
		case 'input':
			return (
				<>
					<div className="nt__ctitle">Load training example {i + 1} of 4</div>
					{row(`x = (x₁, x₂) = (${f(x[0], 0)}, ${f(x[1], 0)})`, `target  y = ${y}`)}
					<div className="nt__cnote">Activations are cleared; the input vector enters the network. Values flow left → right.</div>
				</>
			);
		case 'fwd1z':
			return (
				<>
					<div className="nt__ctitle">Hidden pre-activations &nbsp; z⁽¹⁾ⱼ = Wⱼ₁ x₁ + Wⱼ₂ x₂ + bⱼ</div>
					{[0, 1, 2].map((j) =>
						row(
							`z${sub[j]} = ${par(P.W1[j][0])}·${f(x[0], 0)} + ${par(P.W1[j][1])}·${f(x[1], 0)} + ${par(P.b1[j])}`,
							`= ${f(F.z1[j])}`,
							j,
						),
					)}
				</>
			);
		case 'fwd1a':
			return (
				<>
					<div className="nt__ctitle">Hidden activations &nbsp; a⁽¹⁾ⱼ = tanh(z⁽¹⁾ⱼ)</div>
					{[0, 1, 2].map((j) => row(`a${sub[j]} = tanh(${f(F.z1[j])})`, `= ${f(F.a1[j])}`, j))}
				</>
			);
		case 'fwd2z':
			return (
				<>
					<div className="nt__ctitle">Output pre-activation &nbsp; z⁽²⁾ = Σⱼ W⁽²⁾ⱼ a⁽¹⁾ⱼ + b⁽²⁾</div>
					{row(
						`z⁽²⁾ = ${par(P.W2[0])}${par(F.a1[0])} + ${par(P.W2[1])}${par(F.a1[1])} + ${par(P.W2[2])}${par(F.a1[2])} + ${par(P.b2)}`,
						`= ${f(F.z2)}`,
					)}
				</>
			);
		case 'fwd2a':
			return (
				<>
					<div className="nt__ctitle">Output activation &nbsp; ŷ = σ(z⁽²⁾) = 1 / (1 + e⁻ᶻ)</div>
					{row(`ŷ = 1 / (1 + e^(−${f(F.z2)}))`, `= ${f(F.yhat)}`)}
				</>
			);
		case 'loss':
			return (
				<>
					<div className="nt__ctitle">Squared-error loss &nbsp; L = (ŷ − y)²</div>
					{row(`L = (${f(F.yhat)} − ${y})²`, `= ${f((F.yhat - y) ** 2, 4)}`)}
					<div className="nt__cnote">The single scalar L is what backpropagation now differentiates. Gradients flow right → left.</div>
				</>
			);
		case 'bwd2':
			return (
				<>
					<div className="nt__ctitle">Output layer &nbsp; δ⁽²⁾ = ∂L/∂z⁽²⁾ = 2(ŷ − y)·ŷ(1 − ŷ)</div>
					{row(
						`δ⁽²⁾ = 2(${f(F.yhat)} − ${y})·${f(F.yhat)}(1 − ${f(F.yhat)})`,
						`= ${f(G.d2)}`,
					)}
					{row(`∂L/∂W⁽²⁾ⱼ = δ⁽²⁾·a⁽¹⁾ⱼ`, `= [ ${G.dW2.map((v) => f(v)).join(',  ')} ]`)}
					{row(`∂L/∂b⁽²⁾ = δ⁽²⁾`, `= ${f(G.db2)}`)}
				</>
			);
		case 'bwd1': {
			const allzero = x[0] === 0 && x[1] === 0;
			return (
				<>
					<div className="nt__ctitle">Hidden layer &nbsp; δ⁽¹⁾ⱼ = W⁽²⁾ⱼ·δ⁽²⁾·(1 − a⁽¹⁾ⱼ²)</div>
					{[0, 1, 2].map((j) =>
						row(
							`δ${sub[j]} = ${par(P.W2[j])}·${f(G.d2)}·(1 − ${sq(F.a1[j])})`,
							`= ${f(G.d1[j])}`,
							j,
						),
					)}
					{row(
						`∂L/∂W⁽¹⁾ⱼₖ = δ⁽¹⁾ⱼ·xₖ`,
						`= [ ${G.dW1.map((r) => `[${f(r[0], 2)}, ${f(r[1], 2)}]`).join(', ')} ]`,
					)}
					{row(`∂L/∂b⁽¹⁾ = δ⁽¹⁾`, `= [ ${G.db1.map((v) => f(v)).join(',  ')} ]`)}
					{allzero && (
						<div className="nt__cnote">
							Every ∂L/∂W⁽¹⁾ is 0 here because the input is (0, 0): a weight’s gradient is scaled by
							its input, so a zero input freezes its incoming weights this step. The hidden biases still move.
						</div>
					)}
				</>
			);
		}
		case 'update':
			return (
				<>
					<div className="nt__ctitle">Gradient descent &nbsp; θ ← θ − η · ∂L/∂θ &nbsp; (η = {f(LR, 1)})</div>
					{row(
						`W⁽²⁾ : [ ${P.W2.map((v) => f(v, 2)).join(', ')} ]`,
						`→ [ ${Pn.W2.map((v) => f(v, 2)).join(', ')} ]`,
					)}
					{row(`b⁽²⁾ : ${f(P.b2, 2)}`, `→ ${f(Pn.b2, 2)}`)}
					{[0, 1, 2].map((j) =>
						row(
							`W⁽¹⁾${sub[j]} : [ ${P.W1[j].map((v) => f(v, 2)).join(', ')} ]`,
							`→ [ ${Pn.W1[j].map((v) => f(v, 2)).join(', ')} ]`,
							j,
						),
					)}
					{row(
						`b⁽¹⁾ : [ ${P.b1.map((v) => f(v, 2)).join(', ')} ]`,
						`→ [ ${Pn.b1.map((v) => f(v, 2)).join(', ')} ]`,
					)}
					<div className="nt__cnote">One iteration complete. Step forward to load the next example and repeat — 600 times, the boundary at right resolves into XOR.</div>
				</>
			);
	}
}
