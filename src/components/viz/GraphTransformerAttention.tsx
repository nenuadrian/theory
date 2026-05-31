import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';

/**
 * A graph transformer attends *globally*: the query node attends to every other
 * node at once, not merely its neighbours. Structure re-enters not as a hard mask
 * but as a soft bias on the attention logits — Graphormer's spatial encoding adds
 * a learnable term that decays with the shortest-path distance d(q, j).
 *
 *   α_{qj} = softmax_j( s_{qj} − γ · d(q, j) )
 *
 * Click a node to make it the query. The slider γ is the distance penalty: at
 * γ = 0 the layer is pure content attention reaching the whole graph; as γ grows
 * the mass collapses onto near nodes and the layer behaves like a local GNN. The
 * dashed ring marks the 2-hop horizon — the entire receptive field of a 2-layer
 * message-passing network — so the attention placed *beyond* it is range that no
 * shallow GNN can obtain.
 */

const W = 540;
const HGT = 360;
const R = 15; // node radius (logical px)
const SCALE = 2; // device-pixel oversampling for crisp text

const N = 14;
const EDGES: [number, number][] = [
	[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5],
	[4, 6], [6, 7],
	[7, 8], [7, 9], [8, 9], [8, 10], [9, 10], [9, 11], [10, 11], [10, 12], [11, 12], [11, 13], [12, 13],
];

function mulberry32(seed: number) {
	return () => {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function buildAdj(): number[][] {
	const adj: number[][] = Array.from({ length: N }, () => []);
	EDGES.forEach(([a, b]) => {
		adj[a].push(b);
		adj[b].push(a);
	});
	return adj;
}

/** A fixed, symmetric "content similarity" score s_{ij}, seeded so it is stable. */
function buildScores(): number[][] {
	const rand = mulberry32(11);
	const S = Array.from({ length: N }, () => Array(N).fill(0));
	for (let i = 0; i < N; i++) {
		for (let j = i; j < N; j++) {
			const v = i === j ? 0.6 : rand() * 1.6;
			S[i][j] = v;
			S[j][i] = v;
		}
	}
	return S;
}

/** Deterministic Fruchterman–Reingold layout (no RNG, so it's stable). */
function layout(): { x: number; y: number }[] {
	const pos = Array.from({ length: N }, (_, i) => ({
		x: W / 2 + 0.32 * W * Math.cos((2 * Math.PI * i) / N) + ((i % 3) - 1) * 6,
		y: HGT / 2 + 0.3 * HGT * Math.sin((2 * Math.PI * i) / N) + (((i * 7) % 5) - 2) * 5,
	}));
	const k = Math.sqrt((W * HGT) / N) * 0.62;
	let temp = W * 0.12;
	for (let it = 0; it < 500; it++) {
		const disp = Array.from({ length: N }, () => ({ x: 0, y: 0 }));
		for (let i = 0; i < N; i++) {
			for (let j = i + 1; j < N; j++) {
				const dx = pos[i].x - pos[j].x;
				const dy = pos[i].y - pos[j].y;
				const d = Math.hypot(dx, dy) || 0.01;
				const f = (k * k) / d;
				disp[i].x += (dx / d) * f;
				disp[i].y += (dy / d) * f;
				disp[j].x -= (dx / d) * f;
				disp[j].y -= (dy / d) * f;
			}
		}
		EDGES.forEach(([a, b]) => {
			const dx = pos[a].x - pos[b].x;
			const dy = pos[a].y - pos[b].y;
			const d = Math.hypot(dx, dy) || 0.01;
			const f = (d * d) / k;
			disp[a].x -= (dx / d) * f;
			disp[a].y -= (dy / d) * f;
			disp[b].x += (dx / d) * f;
			disp[b].y += (dy / d) * f;
		});
		for (let i = 0; i < N; i++) {
			const d = Math.hypot(disp[i].x, disp[i].y) || 0.01;
			pos[i].x += (disp[i].x / d) * Math.min(d, temp);
			pos[i].y += (disp[i].y / d) * Math.min(d, temp);
			pos[i].x = Math.max(R + 10, Math.min(W - R - 10, pos[i].x));
			pos[i].y = Math.max(R + 10, Math.min(HGT - R - 10, pos[i].y));
		}
		temp *= 0.985;
	}
	return pos;
}

function bfsDist(adj: number[][], src: number): number[] {
	const dist = Array(N).fill(Infinity);
	if (src < 0) return dist;
	dist[src] = 0;
	const queue = [src];
	while (queue.length) {
		const u = queue.shift()!;
		for (const v of adj[u]) {
			if (dist[v] === Infinity) {
				dist[v] = dist[u] + 1;
				queue.push(v);
			}
		}
	}
	return dist;
}

/** value in [0,1] → deep-slate → teal → amber heat ramp (matches the track). */
function heat(t: number): string {
	t = Math.max(0, Math.min(1, t));
	const stops: [number, [number, number, number]][] = [
		[0.0, [38, 46, 58]],
		[0.5, [45, 212, 191]],
		[1.0, [251, 146, 60]],
	];
	for (let i = 0; i < stops.length - 1; i++) {
		const [t0, c0] = stops[i];
		const [t1, c1] = stops[i + 1];
		if (t <= t1) {
			const u = (t - t0) / (t1 - t0);
			const r = Math.round(c0[0] + (c1[0] - c0[0]) * u);
			const g = Math.round(c0[1] + (c1[1] - c0[1]) * u);
			const b = Math.round(c0[2] + (c1[2] - c0[2]) * u);
			return `rgb(${r},${g},${b})`;
		}
	}
	return 'rgb(251,146,60)';
}

export default function GraphTransformerAttention() {
	const adj = useMemo(buildAdj, []);
	const pos = useMemo(layout, []);
	const scores = useMemo(buildScores, []);

	const [query, setQuery] = useState(0);
	const [gamma, setGamma] = useState(0.6);
	const [showHorizon, setShowHorizon] = useState(true);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	const dist = useMemo(() => bfsDist(adj, query), [adj, query]);

	// α_{qj} = softmax_j( s_{qj} − γ d(q,j) ), over ALL nodes j (the query included).
	const { alpha, beyond2 } = useMemo(() => {
		const logits = dist.map((d, j) =>
			d === Infinity ? -Infinity : scores[query][j] - gamma * d
		);
		const m = Math.max(...logits);
		const ex = logits.map((l) => (l === -Infinity ? 0 : Math.exp(l - m)));
		const Z = ex.reduce((s, v) => s + v, 0) || 1;
		const alpha = ex.map((v) => v / Z);
		let beyond2 = 0;
		for (let j = 0; j < N; j++) if (dist[j] > 2 && dist[j] !== Infinity) beyond2 += alpha[j];
		return { alpha, beyond2 };
	}, [dist, scores, query, gamma]);

	function handleClick(e: MouseEvent<HTMLCanvasElement>) {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = (e.clientX - rect.left) * (W / rect.width);
		const my = (e.clientY - rect.top) * (HGT / rect.height);
		for (let i = 0; i < N; i++) {
			if (Math.hypot(pos[i].x - mx, pos[i].y - my) <= R + 6) {
				setQuery(i);
				return;
			}
		}
	}

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes

		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		const maxA = Math.max(1e-9, ...alpha);
		const qx = pos[query].x;
		const qy = pos[query].y;

		// attention "rays": the query reaches every node, opacity ∝ α_{qj}
		for (let j = 0; j < N; j++) {
			if (j === query) continue;
			const a = alpha[j] / maxA;
			if (a < 0.02) continue;
			ctx.beginPath();
			ctx.moveTo(qx, qy);
			ctx.lineTo(pos[j].x, pos[j].y);
			ctx.strokeStyle = `rgba(251,146,60,${0.12 + 0.7 * a})`;
			ctx.lineWidth = 0.6 + 3.2 * a;
			ctx.stroke();
		}

		// the graph's own edges, faint, drawn over the rays
		ctx.strokeStyle = 'rgba(140,140,150,0.32)';
		ctx.lineWidth = 1.2;
		EDGES.forEach(([a, b]) => {
			ctx.beginPath();
			ctx.moveTo(pos[a].x, pos[a].y);
			ctx.lineTo(pos[b].x, pos[b].y);
			ctx.stroke();
		});

		for (let i = 0; i < N; i++) {
			const a = alpha[i] / maxA;
			// 2-hop horizon: the whole receptive field of a 2-layer message-passing GNN
			if (showHorizon && dist[i] > 2 && dist[i] !== Infinity) {
				ctx.beginPath();
				ctx.setLineDash([3, 3]);
				ctx.arc(pos[i].x, pos[i].y, R + 4, 0, 2 * Math.PI);
				ctx.strokeStyle = 'rgba(120,130,145,0.7)';
				ctx.lineWidth = 1;
				ctx.stroke();
				ctx.setLineDash([]);
			}
			// attention halo, radius ∝ α
			if (i !== query && a > 0.02) {
				ctx.beginPath();
				ctx.arc(pos[i].x, pos[i].y, R + 4 + 12 * a, 0, 2 * Math.PI);
				ctx.fillStyle = `rgba(251,146,60,${0.16 * a})`;
				ctx.fill();
			}
			// node
			ctx.beginPath();
			ctx.arc(pos[i].x, pos[i].y, R, 0, 2 * Math.PI);
			ctx.fillStyle = i === query ? '#1f2937' : heat(a);
			ctx.fill();
			ctx.lineWidth = i === query ? 3 : 1.2;
			ctx.strokeStyle = i === query ? '#fb923c' : 'rgba(150,150,160,0.7)';
			ctx.stroke();
			// label: attention weight (the query shows "q")
			ctx.fillStyle = i === query ? '#fb923c' : a > 0.55 ? '#0b0f14' : fg;
			ctx.font = '600 11px ui-monospace, monospace';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText(i === query ? 'q' : alpha[i].toFixed(2), pos[i].x, pos[i].y);
		}

		// legend for the horizon ring
		if (showHorizon) {
			ctx.fillStyle = fg;
			ctx.font = '11px ui-monospace, monospace';
			ctx.textAlign = 'left';
			ctx.globalAlpha = 0.8;
			ctx.fillText('⌝ dashed = beyond a 2-layer GNN', 12, HGT - 10);
			ctx.globalAlpha = 1;
		}
	}, [alpha, dist, query, showHorizon, pos]);

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: W, maxWidth: '100%' }}>
					<canvas
						ref={canvasRef}
						width={W * SCALE}
						height={HGT * SCALE}
						className="viz__canvas"
						style={{ width: '100%', height: 'auto', cursor: 'pointer' }}
						onClick={handleClick}
						aria-label="A query node attending over an entire graph, modulated by shortest-path distance"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							distance penalty γ <b>{gamma.toFixed(1)}</b>
						</span>
						<input
							type="range"
							min={0}
							max={3}
							step={0.1}
							value={gamma}
							onChange={(e) => setGamma(parseFloat(e.target.value))}
						/>
					</label>

					<label
						className="viz__slider"
						style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}
					>
						<input
							type="checkbox"
							checked={showHorizon}
							onChange={(e) => setShowHorizon(e.target.checked)}
						/>
						<span className="viz__slider-label">show 2-hop horizon</span>
					</label>

					<button type="button" className="viz__btn" onClick={() => { setQuery(0); setGamma(0.6); }}>
						reset
					</button>

					<p className="viz__slider-label" style={{ lineHeight: 1.6 }}>
						query: node <b>q = {query}</b>
						<br />
						attention beyond 2 hops:
						<br />
						<b>{(100 * beyond2).toFixed(0)}%</b>{' '}
						<span style={{ opacity: 0.8 }}>(a 2-layer GNN: 0%)</span>
						<br />
						<span style={{ opacity: 0.8 }}>
							{gamma < 0.3
								? 'γ≈0 — global content attention'
								: beyond2 < 0.08
									? 'localized — GNN-like'
									: 'reaching across the graph'}
						</span>
						<br />
						<span style={{ opacity: 0.8 }}>click a node to move q</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">αqⱼ = softmax₍ⱼ₎( sqⱼ − γ · d(q,j) ) ,  Σⱼ αqⱼ = 1</code>
		</div>
	);
}
