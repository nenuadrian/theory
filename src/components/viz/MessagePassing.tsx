import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';

/**
 * Message passing on a graph, made visible.
 *
 * Click any node to inject a signal of 1 (everything else 0), then press "step"
 * to run one round of neighbourhood aggregation. Watch the signal diffuse — and
 * watch the k-hop *receptive field* of the source grow by exactly one ring per
 * layer. That growing ring is why a GNN with k layers sees a k-hop neighbourhood.
 */

type Agg = 'mean' | 'sum' | 'max';

const W = 520;
const HGT = 380;
const R = 15; // node radius (logical px)
const SCALE = 2; // device-pixel oversampling for crisp text

const N = 14;
const EDGES: [number, number][] = [
	[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 5],
	[4, 6], [6, 7],
	[7, 8], [7, 9], [8, 9], [8, 10], [9, 10], [9, 11], [10, 11], [10, 12], [11, 12], [11, 13], [12, 13],
];

const FORMULA: Record<Agg, string> = {
	mean: 'hᵢ ← ( hᵢ + Σⱼ hⱼ ) / ( 1 + deg(i) )',
	sum: 'hᵢ ← hᵢ + Σⱼ∈𝒩(i) hⱼ',
	max: 'hᵢ ← max( hᵢ , maxⱼ∈𝒩(i) hⱼ )',
};

function buildAdj(): number[][] {
	const adj: number[][] = Array.from({ length: N }, () => []);
	EDGES.forEach(([a, b]) => {
		adj[a].push(b);
		adj[b].push(a);
	});
	return adj;
}

/** Deterministic Fruchterman–Reingold layout (no RNG, so it's stable). */
function layout(adj: number[][]): { x: number; y: number }[] {
	const pos = Array.from({ length: N }, (_, i) => ({
		// circle init + tiny deterministic jitter to break perfect symmetry
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

/** value in [0,1] → deep-slate → teal → amber heat ramp. */
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

export default function MessagePassing() {
	const adj = useMemo(buildAdj, []);
	const pos = useMemo(() => layout(adj), [adj]);

	const [values, setValues] = useState<number[]>(() => {
		const v = Array(N).fill(0);
		v[0] = 1;
		return v;
	});
	const [source, setSource] = useState(0);
	const [steps, setSteps] = useState(0);
	const [agg, setAgg] = useState<Agg>('mean');
	const [showField, setShowField] = useState(true);

	const dist = useMemo(() => bfsDist(adj, source), [adj, source]);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	function reset(src = source) {
		const v = Array(N).fill(0);
		if (src >= 0) v[src] = 1;
		setValues(v);
		setSteps(0);
	}

	function step() {
		setValues((prev) => {
			const next = prev.slice();
			for (let i = 0; i < N; i++) {
				if (agg === 'mean') {
					let s = prev[i];
					let c = 1;
					for (const j of adj[i]) {
						s += prev[j];
						c++;
					}
					next[i] = s / c;
				} else if (agg === 'sum') {
					let s = prev[i];
					for (const j of adj[i]) s += prev[j];
					next[i] = s;
				} else {
					let m = prev[i];
					for (const j of adj[i]) m = Math.max(m, prev[j]);
					next[i] = m;
				}
			}
			return next;
		});
		setSteps((s) => s + 1);
	}

	function handleClick(e: MouseEvent<HTMLCanvasElement>) {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = (e.clientX - rect.left) * (W / rect.width);
		const my = (e.clientY - rect.top) * (HGT / rect.height);
		for (let i = 0; i < N; i++) {
			if (Math.hypot(pos[i].x - mx, pos[i].y - my) <= R + 5) {
				setSource(i);
				reset(i);
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

		// edges
		ctx.strokeStyle = 'rgba(140,140,150,0.45)';
		ctx.lineWidth = 1.4;
		EDGES.forEach(([a, b]) => {
			ctx.beginPath();
			ctx.moveTo(pos[a].x, pos[a].y);
			ctx.lineTo(pos[b].x, pos[b].y);
			ctx.stroke();
		});

		const maxV = Math.max(1e-9, ...values);

		for (let i = 0; i < N; i++) {
			const t = values[i] / maxV;
			// k-hop receptive-field ring
			if (showField && dist[i] <= steps && dist[i] !== Infinity) {
				ctx.beginPath();
				ctx.arc(pos[i].x, pos[i].y, R + 5, 0, 2 * Math.PI);
				ctx.strokeStyle = 'rgba(45,212,191,0.9)';
				ctx.lineWidth = 2;
				ctx.stroke();
			}
			// node
			ctx.beginPath();
			ctx.arc(pos[i].x, pos[i].y, R, 0, 2 * Math.PI);
			ctx.fillStyle = heat(t);
			ctx.fill();
			ctx.lineWidth = i === source ? 3 : 1.2;
			ctx.strokeStyle = i === source ? '#fb923c' : 'rgba(150,150,160,0.7)';
			ctx.stroke();
			// value label
			ctx.fillStyle = t > 0.55 ? '#0b0f14' : fg;
			ctx.font = '600 11px ui-monospace, monospace';
			ctx.textAlign = 'center';
			ctx.textBaseline = 'middle';
			ctx.fillText(values[i].toFixed(2), pos[i].x, pos[i].y);
		}
	}, [values, source, steps, dist, showField, pos]);

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
						aria-label="Graph with a signal diffusing under message passing"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__select">
						<span className="viz__slider-label">aggregator</span>
						<select value={agg} onChange={(e) => setAgg(e.target.value as Agg)}>
							<option value="mean">mean</option>
							<option value="sum">sum</option>
							<option value="max">max</option>
						</select>
					</label>

					<label className="viz__slider" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
						<input
							type="checkbox"
							checked={showField}
							onChange={(e) => setShowField(e.target.checked)}
						/>
						<span className="viz__slider-label">show k-hop field</span>
					</label>

					<div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
						<button type="button" className="viz__btn" onClick={step}>
							▸ step
						</button>
						<button type="button" className="viz__btn" onClick={() => reset()}>
							reset
						</button>
					</div>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						layer <b>k = {steps}</b>
						<br />
						source: node <b>{source}</b>
						<br />
						<span style={{ opacity: 0.8 }}>click a node to move the source</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">{FORMULA[agg]}</code>
		</div>
	);
}
