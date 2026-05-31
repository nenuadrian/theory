import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Why you can't just stack GNN layers forever. Each layer applies the propagation
 * operator Â = D̃^{-1/2}(A+I)D̃^{-1/2}. Apply it enough times and *every* node's
 * representation converges to the same thing — the two communities, clearly
 * separated at layer 0, blur into one indistinguishable blob. The Dirichlet
 * energy (total disagreement across edges) decays to zero. That's over-smoothing.
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const MAXL = 30;
const N = 20; // two communities of 10

const TEAL = [45, 212, 191];
const PINK = [236, 72, 153];

function mulberry32(seed: number) {
	return () => {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function buildEdges(): [number, number][] {
	const edges: [number, number][] = [];
	for (const base of [0, 10]) {
		for (let i = 0; i < 10; i++) {
			edges.push([base + i, base + ((i + 1) % 10)]);
			edges.push([base + i, base + ((i + 2) % 10)]);
		}
	}
	edges.push([0, 10], [5, 15], [9, 11]); // a few inter-community bridges
	return edges;
}

/** Â = D̃^{-1/2} (A + I) D̃^{-1/2}. */
function propagationOp(edges: [number, number][]) {
	const A = Array.from({ length: N }, () => Array(N).fill(0));
	for (let i = 0; i < N; i++) A[i][i] = 1; // self-loops
	edges.forEach(([a, b]) => {
		A[a][b] = 1;
		A[b][a] = 1;
	});
	const deg = A.map((row) => row.reduce((s, v) => s + v, 0));
	const Ahat = Array.from({ length: N }, () => Array(N).fill(0));
	for (let i = 0; i < N; i++)
		for (let j = 0; j < N; j++) Ahat[i][j] = A[i][j] / Math.sqrt(deg[i] * deg[j]);
	return { Ahat, deg };
}

function layout(edges: [number, number][], rect: { x: number; y: number; w: number; h: number }) {
	const rand = mulberry32(7);
	const pos = Array.from({ length: N }, (_, i) => ({
		x: rect.x + rect.w * (0.2 + 0.6 * rand()),
		y: rect.y + rect.h * (0.2 + 0.6 * rand()),
	}));
	const k = Math.sqrt((rect.w * rect.h) / N) * 0.7;
	let temp = rect.w * 0.1;
	for (let it = 0; it < 400; it++) {
		const disp = Array.from({ length: N }, () => ({ x: 0, y: 0 }));
		for (let i = 0; i < N; i++)
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
		edges.forEach(([a, b]) => {
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
			pos[i].x = Math.max(rect.x + 12, Math.min(rect.x + rect.w - 12, pos[i].x));
			pos[i].y = Math.max(rect.y + 12, Math.min(rect.y + rect.h - 12, pos[i].y));
		}
		temp *= 0.98;
	}
	return pos;
}

export default function OverSmoothing() {
	const [L, setL] = useState(0);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	const edges = useMemo(buildEdges, []);
	const pos = useMemo(
		() => layout(edges, { x: 12, y: 16, w: W * 0.46 - 12, h: HGT - 32 }),
		[edges]
	);

	// Precompute g_i = (Â^L X0)_i / sqrt(d̃_i) and the Dirichlet energy for each L.
	const { gSeries, energy, range } = useMemo(() => {
		const { Ahat, deg } = propagationOp(edges);
		const rand = mulberry32(42);
		let X = Array.from({ length: N }, () => [rand() * 2 - 1, rand() * 2 - 1]);
		const gOf = (Xc: number[][]) => Xc.map((row, i) => [row[0] / Math.sqrt(deg[i]), row[1] / Math.sqrt(deg[i])]);
		const gSeries: number[][][] = [];
		const energy: number[] = [];
		for (let l = 0; l <= MAXL; l++) {
			const g = gOf(X);
			gSeries.push(g);
			let e = 0;
			edges.forEach(([a, b]) => {
				const dx = g[a][0] - g[b][0];
				const dy = g[a][1] - g[b][1];
				e += dx * dx + dy * dy;
			});
			energy.push(e);
			// X <- Â X
			const Xn = Array.from({ length: N }, () => [0, 0]);
			for (let i = 0; i < N; i++)
				for (let j = 0; j < N; j++) {
					Xn[i][0] += Ahat[i][j] * X[j][0];
					Xn[i][1] += Ahat[i][j] * X[j][1];
				}
			X = Xn;
		}
		const g0 = gSeries[0];
		const xs = g0.map((p) => p[0]);
		const ys = g0.map((p) => p[1]);
		const pad = 0.15;
		const range = {
			x0: Math.min(...xs),
			x1: Math.max(...xs),
			y0: Math.min(...ys),
			y1: Math.max(...ys),
		};
		const dx = (range.x1 - range.x0) * pad || 0.1;
		const dy = (range.y1 - range.y0) * pad || 0.1;
		range.x0 -= dx;
		range.x1 += dx;
		range.y0 -= dy;
		range.y1 += dy;
		return { gSeries, energy, range };
	}, [edges]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes
		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		const color = (i: number, alpha = 1) => {
			const c = i < 10 ? TEAL : PINK;
			return `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
		};

		// ---- left: the graph ------------------------------------------------
		ctx.strokeStyle = 'rgba(140,140,150,0.35)';
		ctx.lineWidth = 1;
		edges.forEach(([a, b]) => {
			ctx.beginPath();
			ctx.moveTo(pos[a].x, pos[a].y);
			ctx.lineTo(pos[b].x, pos[b].y);
			ctx.stroke();
		});
		for (let i = 0; i < N; i++) {
			ctx.beginPath();
			ctx.arc(pos[i].x, pos[i].y, 8, 0, 2 * Math.PI);
			ctx.fillStyle = color(i);
			ctx.fill();
		}
		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'center';
		ctx.fillText('the graph (two communities)', W * 0.23, HGT - 6);

		// ---- right-top: feature scatter ------------------------------------
		const sx0 = W * 0.54;
		const sx1 = W - 14;
		const sy0 = 16;
		const sy1 = HGT * 0.52;
		const mapX = (x: number) => sx0 + ((x - range.x0) / (range.x1 - range.x0)) * (sx1 - sx0);
		const mapY = (y: number) => sy1 - ((y - range.y0) / (range.y1 - range.y0)) * (sy1 - sy0);
		ctx.strokeStyle = 'rgba(140,140,150,0.3)';
		ctx.lineWidth = 1;
		ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
		const g = gSeries[L];
		for (let i = 0; i < N; i++) {
			ctx.beginPath();
			ctx.arc(mapX(g[i][0]), mapY(g[i][1]), 4.5, 0, 2 * Math.PI);
			ctx.fillStyle = color(i, 0.95);
			ctx.fill();
		}
		ctx.fillStyle = fg;
		ctx.textAlign = 'left';
		ctx.globalAlpha = 0.85;
		ctx.fillText('node features  hᵢ / √d̃ᵢ', sx0, sy0 - 4);
		ctx.globalAlpha = 1;

		// ---- right-bottom: Dirichlet energy curve --------------------------
		const ex0 = W * 0.54;
		const ex1 = W - 14;
		const ey0 = HGT * 0.62;
		const ey1 = HGT - 22;
		const maxE = Math.max(1e-9, ...energy);
		ctx.strokeStyle = 'rgba(140,140,150,0.3)';
		ctx.strokeRect(ex0, ey0, ex1 - ex0, ey1 - ey0);
		ctx.beginPath();
		for (let l = 0; l <= MAXL; l++) {
			const px = ex0 + (l / MAXL) * (ex1 - ex0);
			const py = ey1 - (energy[l] / maxE) * (ey1 - ey0);
			if (l === 0) ctx.moveTo(px, py);
			else ctx.lineTo(px, py);
		}
		ctx.strokeStyle = '#fb923c';
		ctx.lineWidth = 2;
		ctx.stroke();
		// marker at current L
		const mxp = ex0 + (L / MAXL) * (ex1 - ex0);
		const myp = ey1 - (energy[L] / maxE) * (ey1 - ey0);
		ctx.beginPath();
		ctx.arc(mxp, myp, 4, 0, 2 * Math.PI);
		ctx.fillStyle = '#fb923c';
		ctx.fill();
		ctx.strokeStyle = 'rgba(251,146,60,0.4)';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(mxp, ey0);
		ctx.lineTo(mxp, ey1);
		ctx.stroke();
		ctx.fillStyle = fg;
		ctx.globalAlpha = 0.85;
		ctx.fillText('Dirichlet energy  vs  depth', ex0, ey0 - 4);
		ctx.globalAlpha = 1;
	}, [L, gSeries, energy, range, edges, pos]);

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
						aria-label="Node features collapsing and Dirichlet energy decaying with depth"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							layers L <b>{L}</b>
						</span>
						<input
							type="range"
							min={0}
							max={MAXL}
							step={1}
							value={L}
							onChange={(e) => setL(parseInt(e.target.value, 10))}
						/>
					</label>

					<p className="viz__slider-label" style={{ lineHeight: 1.6 }}>
						Dirichlet energy
						<br />
						<b>E = {energy[L].toExponential(2)}</b>
						<br />
						<span style={{ opacity: 0.8 }}>
							{L === 0 ? 'two clusters, well separated' : energy[L] / energy[0] < 0.05 ? 'collapsed — nodes are ~identical' : 'clusters merging…'}
						</span>
					</p>

					<button type="button" className="viz__btn" onClick={() => setL(0)}>
						reset
					</button>
				</div>
			</div>
			<code className="viz__formula">H⁽ˡ⁺¹⁾ = Â H⁽ˡ⁾ ,  E(H) = Σ₍ᵢ,ⱼ₎∈E ‖hᵢ/√d̃ᵢ − hⱼ/√d̃ⱼ‖² → 0</code>
		</div>
	);
}
