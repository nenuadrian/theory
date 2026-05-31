import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Monte-Carlo tree search over a learned model (MuZero-style).
 *
 * A depth-3 decision tree. The best leaf (0.95) hides down the LEFT path, but the
 * prior policy is biased toward the RIGHT (0.8). Each simulation selects a path by
 * PUCT — balancing the value estimate Q against the prior P and the visit counts —
 * expands a node, evaluates it with the value head, and backs the value up. Step
 * through and watch the visit counts concentrate on the left: the search *corrects*
 * the prior. The resulting root visit distribution is the improved policy MuZero
 * distills back into the network.
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const D = 3;
const N_INTERNAL = 2 ** D - 1; // 7
const N_TOTAL = 2 ** (D + 1) - 1; // 15
const C_PUCT = 1.4;
const MAXSIM = 120;

const LEAF: Record<number, number> = {
	7: 0.95, 8: 0.3, 9: 0.25, 10: 0.2, 11: 0.55, 12: 0.5, 13: 0.45, 14: 0.6,
};

const isLeaf = (i: number) => i >= N_INTERNAL;
const child = (i: number, a: number) => 2 * i + 1 + a;
const depth = (i: number) => Math.floor(Math.log2(i + 1));
const prior = (i: number): [number, number] => (i === 0 ? [0.2, 0.8] : [0.5, 0.5]);

function bestReachable(i: number): number {
	if (isLeaf(i)) return LEAF[i];
	return Math.max(bestReachable(child(i, 0)), bestReachable(child(i, 1)));
}

function mulberry32(seed: number) {
	return () => {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

// node layout (computed once)
const POS: { x: number; y: number }[] = (() => {
	const x = new Array(N_TOTAL).fill(0);
	const padX = 26,
		padY = 30,
		levelGap = (HGT - 86 - padY) / D;
	for (let leaf = N_INTERNAL; leaf < N_TOTAL; leaf++)
		x[leaf] = padX + ((leaf - N_INTERNAL) / (N_TOTAL - N_INTERNAL - 1)) * (W - 2 * padX);
	for (let i = N_INTERNAL - 1; i >= 0; i--) x[i] = (x[child(i, 0)] + x[child(i, 1)]) / 2;
	return Array.from({ length: N_TOTAL }, (_, i) => ({ x: x[i], y: padY + depth(i) * levelGap }));
})();

interface Snap {
	N: Record<string, number>; // edge "i_a" -> visits
	path: [number, number][];
	leafReached: number;
}

export default function MCTSTree() {
	const [sim, setSim] = useState(0);
	const [running, setRunning] = useState(false);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const raf = useRef<number | null>(null);

	// Run the whole search, snapshotting after each simulation for scrubbing.
	const { snaps, vnet } = useMemo(() => {
		const rand = mulberry32(20240531);
		const vnet: Record<number, number> = {};
		for (let i = 0; i < N_TOTAL; i++)
			vnet[i] = Math.max(0, Math.min(1, bestReachable(i) + 0.1 * (rand() * 2 - 1)));
		const N: Record<string, number> = {};
		const Wv: Record<string, number> = {};
		const expanded = new Set<number>();
		const ek = (i: number, a: number) => `${i}_${a}`;
		const expand = (i: number) => {
			expanded.add(i);
			N[ek(i, 0)] = 0;
			N[ek(i, 1)] = 0;
			Wv[ek(i, 0)] = 0;
			Wv[ek(i, 1)] = 0;
		};
		expand(0);
		const snaps: Snap[] = [];
		for (let s = 0; s < MAXSIM; s++) {
			const path: [number, number][] = [];
			let i = 0;
			while (!isLeaf(i) && expanded.has(i)) {
				const sumN = N[ek(i, 0)] + N[ek(i, 1)];
				let best = -1e9,
					ba = 0;
				const p = prior(i);
				for (let a = 0; a < 2; a++) {
					const q = N[ek(i, a)] > 0 ? Wv[ek(i, a)] / N[ek(i, a)] : 0;
					const u = (C_PUCT * p[a] * Math.sqrt(sumN + 1)) / (1 + N[ek(i, a)]);
					if (q + u > best) {
						best = q + u;
						ba = a;
					}
				}
				path.push([i, ba]);
				i = child(i, ba);
			}
			let v: number;
			if (isLeaf(i)) v = LEAF[i];
			else {
				expand(i);
				v = vnet[i];
			}
			for (const [node, a] of path) {
				N[ek(node, a)] += 1;
				Wv[ek(node, a)] += v;
			}
			snaps.push({ N: { ...N }, path, leafReached: i });
		}
		return { snaps, vnet };
	}, []);

	useEffect(() => {
		if (!running) return;
		const tick = () => {
			setSim((s) => {
				if (s >= MAXSIM) {
					setRunning(false);
					return s;
				}
				raf.current = window.setTimeout(() => requestAnimationFrame(tick), 150) as unknown as number;
				return s + 1;
			});
		};
		raf.current = window.setTimeout(() => requestAnimationFrame(tick), 150) as unknown as number;
		return () => {
			if (raf.current) clearTimeout(raf.current);
		};
	}, [running]);

	const snap = sim > 0 ? snaps[sim - 1] : null;
	const Nedge = (i: number, a: number) => (snap ? snap.N[`${i}_${a}`] ?? 0 : 0);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes
		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		const pathSet = new Set((snap?.path ?? []).map(([i, a]) => `${i}_${a}`));
		let maxN = 1;
		for (let i = 0; i < N_INTERNAL; i++)
			for (let a = 0; a < 2; a++) maxN = Math.max(maxN, Nedge(i, a));

		// edges
		for (let i = 0; i < N_INTERNAL; i++)
			for (let a = 0; a < 2; a++) {
				const c = child(i, a);
				const onPath = pathSet.has(`${i}_${a}`);
				const n = Nedge(i, a);
				ctx.strokeStyle = onPath
					? 'rgba(251,146,60,0.95)'
					: `rgba(45,212,191,${0.18 + 0.5 * (n / maxN)})`;
				ctx.lineWidth = onPath ? 3 : 1 + 3 * (n / maxN);
				ctx.beginPath();
				ctx.moveTo(POS[i].x, POS[i].y);
				ctx.lineTo(POS[c].x, POS[c].y);
				ctx.stroke();
				// visit count on the edge
				if (n > 0) {
					ctx.fillStyle = 'rgba(196,196,203,0.8)';
					ctx.font = '9px ui-monospace, monospace';
					ctx.textAlign = 'center';
					ctx.fillText(`${n}`, (POS[i].x + POS[c].x) / 2 + 6, (POS[i].y + POS[c].y) / 2);
				}
			}

		// nodes
		for (let i = 0; i < N_TOTAL; i++) {
			const onPath = i === snap?.leafReached && snap;
			ctx.beginPath();
			ctx.arc(POS[i].x, POS[i].y, isLeaf(i) ? 11 : 8, 0, 2 * Math.PI);
			if (isLeaf(i)) {
				const g = Math.round(40 + LEAF[i] * 180);
				ctx.fillStyle = i === 7 ? 'rgba(45,212,191,0.9)' : `rgb(${g},${g},${g})`;
			} else {
				ctx.fillStyle = 'rgba(60,60,70,0.95)';
			}
			ctx.fill();
			ctx.strokeStyle = onPath ? '#fb923c' : 'rgba(150,150,160,0.7)';
			ctx.lineWidth = onPath ? 2.5 : 1;
			ctx.stroke();
			if (isLeaf(i)) {
				ctx.fillStyle = LEAF[i] > 0.6 ? '#0b0f14' : fg;
				ctx.font = '600 9px ui-monospace, monospace';
				ctx.textAlign = 'center';
				ctx.textBaseline = 'middle';
				ctx.fillText(LEAF[i].toFixed(2), POS[i].x, POS[i].y);
			}
		}
		ctx.textBaseline = 'alphabetic';

		// labels: prior on the two root edges
		ctx.fillStyle = 'rgba(236,72,153,0.9)';
		ctx.font = '9px ui-monospace, monospace';
		ctx.textAlign = 'center';
		ctx.fillText('prior .20', POS[0].x - 60, POS[0].y + 14);
		ctx.fillText('prior .80', POS[0].x + 60, POS[0].y + 14);
		ctx.fillStyle = '#2dd4bf';
		ctx.fillText('best leaf', POS[7].x, POS[7].y + 24);

		// root visit policy bars
		const nl = Nedge(0, 0),
			nr = Nedge(0, 1),
			tot = Math.max(1, nl + nr);
		const by = HGT - 30,
			bw = 150,
			bx = W / 2 - bw / 2;
		ctx.fillStyle = 'rgba(140,140,150,0.25)';
		ctx.fillRect(bx, by, bw, 14);
		ctx.fillStyle = 'rgba(45,212,191,0.85)';
		ctx.fillRect(bx, by, bw * (nl / tot), 14);
		ctx.fillStyle = fg;
		ctx.font = '10px ui-monospace, monospace';
		ctx.textAlign = 'center';
		ctx.fillText(
			`root visit policy:  LEFT ${(nl / tot).toFixed(2)}   RIGHT ${(nr / tot).toFixed(2)}`,
			W / 2,
			by - 5,
		);
	}, [sim, snap]);

	const nl = Nedge(0, 0),
		nr = Nedge(0, 1);

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: W, maxWidth: '100%' }}>
					<canvas
						ref={canvasRef}
						width={W * SCALE}
						height={HGT * SCALE}
						className="viz__canvas"
						style={{ width: '100%', height: 'auto' }}
						aria-label="Monte-Carlo tree search concentrating visits on the best path"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							simulations <b>{sim}</b>
						</span>
						<input
							type="range"
							min={0}
							max={MAXSIM}
							step={1}
							value={sim}
							onChange={(e) => setSim(parseInt(e.target.value, 10))}
						/>
					</label>

					<div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
						<button
							type="button"
							className="viz__btn"
							onClick={() => setSim((s) => Math.min(MAXSIM, s + 1))}
							disabled={running}
						>
							▸ step
						</button>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								if (sim >= MAXSIM) setSim(0);
								setRunning((r) => !r);
							}}
						>
							{running ? '❚❚ pause' : '▶ play'}
						</button>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								setRunning(false);
								setSim(0);
							}}
						>
							reset
						</button>
					</div>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						<span style={{ opacity: 0.85 }}>
							The prior says go <b>right</b> (0.80). Search finds the 0.95 leaf on the{' '}
							<b>left</b> and shifts the visit policy to {nl + nr > 0 ? (nl / (nl + nr)).toFixed(2) : '—'} left.
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">
				select aₜ = arg max_a [ Q(s,a) + c · P(s,a) · √(Σ N) / (1 + N(s,a)) ]
			</code>
		</div>
	);
}
