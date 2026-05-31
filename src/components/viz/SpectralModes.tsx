import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';

/**
 * The graph Fourier basis. Every graph has a Laplacian L = I − D^{-1/2} A D^{-1/2};
 * its eigenvectors are the graph's "frequencies". Low eigenvalues are smooth modes
 * (neighbours agree); high eigenvalues oscillate (neighbours disagree). Slide
 * through the spectrum — or click a bar — and watch the eigenvector painted on the
 * nodes. This is exactly the basis a spectral graph convolution filters in.
 */

type GraphKind = 'path' | 'ring' | 'grid' | 'communities';

const W = 520;
const HGT = 360;
const SPEC_H = 70; // spectrum strip height
const R = 13;
const SCALE = 2;

interface G {
	n: number;
	edges: [number, number][];
	pos: { x: number; y: number }[];
}

function ringPos(n: number, cx: number, cy: number, rad: number) {
	return Array.from({ length: n }, (_, i) => ({
		x: cx + rad * Math.cos((2 * Math.PI * i) / n - Math.PI / 2),
		y: cy + rad * Math.sin((2 * Math.PI * i) / n - Math.PI / 2),
	}));
}

function buildGraph(kind: GraphKind): G {
	const top = 18;
	const h = HGT - SPEC_H - 30;
	if (kind === 'path') {
		const n = 12;
		const edges: [number, number][] = [];
		for (let i = 0; i < n - 1; i++) edges.push([i, i + 1]);
		const pos = Array.from({ length: n }, (_, i) => ({
			x: 40 + (i * (W - 80)) / (n - 1),
			y: top + h / 2,
		}));
		return { n, edges, pos };
	}
	if (kind === 'ring') {
		const n = 12;
		const edges: [number, number][] = [];
		for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
		return { n, edges, pos: ringPos(n, W / 2, top + h / 2, Math.min(W, h) * 0.4) };
	}
	if (kind === 'grid') {
		const cols = 5;
		const rows = 4;
		const n = cols * rows;
		const edges: [number, number][] = [];
		const id = (r: number, c: number) => r * cols + c;
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				if (c < cols - 1) edges.push([id(r, c), id(r, c + 1)]);
				if (r < rows - 1) edges.push([id(r, c), id(r + 1, c)]);
			}
		}
		const pos = [];
		for (let r = 0; r < rows; r++)
			for (let c = 0; c < cols; c++)
				pos.push({
					x: 70 + (c * (W - 140)) / (cols - 1),
					y: top + 20 + (r * (h - 40)) / (rows - 1),
				});
		return { n, edges, pos };
	}
	// two communities (a stochastic-block-model-ish fixed graph)
	const n = 14;
	const edges: [number, number][] = [
		[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [3, 4], [4, 5], [4, 6], [5, 6], [2, 5], [0, 6],
		[7, 8], [7, 9], [7, 10], [8, 9], [8, 10], [9, 10], [10, 11], [11, 12], [11, 13], [12, 13], [9, 12], [7, 13],
		[6, 7], // the single bridge
	];
	const left = ringPos(7, W * 0.3, top + h / 2, Math.min(W, h) * 0.26);
	const right = ringPos(7, W * 0.7, top + h / 2, Math.min(W, h) * 0.26);
	return { n, edges, pos: [...left, ...right] };
}

/** Symmetric normalized Laplacian L = I − D^{-1/2} A D^{-1/2}. */
function symLaplacian(g: G): number[][] {
	const { n, edges } = g;
	const A = Array.from({ length: n }, () => Array(n).fill(0));
	const deg = Array(n).fill(0);
	edges.forEach(([a, b]) => {
		A[a][b] = 1;
		A[b][a] = 1;
	});
	for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) deg[i] += A[i][j];
	const L = Array.from({ length: n }, () => Array(n).fill(0));
	for (let i = 0; i < n; i++) {
		for (let j = 0; j < n; j++) {
			const norm = deg[i] > 0 && deg[j] > 0 ? A[i][j] / Math.sqrt(deg[i] * deg[j]) : 0;
			L[i][j] = (i === j ? 1 : 0) - norm;
		}
	}
	return L;
}

/** Jacobi eigensolver for a symmetric matrix. Returns ascending-sorted pairs. */
function eigSym(Ain: number[][]): { values: number[]; vecs: number[][] } {
	const n = Ain.length;
	const A = Ain.map((r) => r.slice());
	const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
	for (let sweep = 0; sweep < 100; sweep++) {
		let p = 0;
		let q = 1;
		let off = 0;
		for (let i = 0; i < n; i++)
			for (let j = i + 1; j < n; j++) {
				const a = Math.abs(A[i][j]);
				if (a > off) {
					off = a;
					p = i;
					q = j;
				}
			}
		if (off < 1e-10) break;
		const app = A[p][p];
		const aqq = A[q][q];
		const apq = A[p][q];
		const theta = (aqq - app) / (2 * apq);
		const t = Math.abs(theta) < 1e-12 ? 1 : Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
		const c = 1 / Math.sqrt(t * t + 1);
		const s = t * c;
		for (let i = 0; i < n; i++) {
			const aip = A[i][p];
			const aiq = A[i][q];
			A[i][p] = c * aip - s * aiq;
			A[i][q] = s * aip + c * aiq;
		}
		for (let i = 0; i < n; i++) {
			const api = A[p][i];
			const aqi = A[q][i];
			A[p][i] = c * api - s * aqi;
			A[q][i] = s * api + c * aqi;
		}
		for (let i = 0; i < n; i++) {
			const vip = V[i][p];
			const viq = V[i][q];
			V[i][p] = c * vip - s * viq;
			V[i][q] = s * vip + c * viq;
		}
	}
	const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => A[a][a] - A[b][b]);
	return {
		values: idx.map((i) => A[i][i]),
		vecs: idx.map((k) => V.map((row) => row[k])), // vecs[k][i] = component i of mode k
	};
}

/** signed value in [-1,1] → blue (−) / neutral / pink (+). */
function diverging(v: number): string {
	const t = Math.max(-1, Math.min(1, v));
	const neutral = [110, 116, 128];
	const pos = [236, 72, 153];
	const neg = [59, 130, 246];
	const target = t >= 0 ? pos : neg;
	const a = Math.abs(t);
	const r = Math.round(neutral[0] + (target[0] - neutral[0]) * a);
	const g = Math.round(neutral[1] + (target[1] - neutral[1]) * a);
	const b = Math.round(neutral[2] + (target[2] - neutral[2]) * a);
	return `rgb(${r},${g},${b})`;
}

export default function SpectralModes() {
	const [kind, setKind] = useState<GraphKind>('ring');
	const [mode, setMode] = useState(1);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	const g = useMemo(() => buildGraph(kind), [kind]);
	const spectrum = useMemo(() => eigSym(symLaplacian(g)), [g]);
	const m = Math.min(mode, g.n - 1);

	const signFlips = useMemo(() => {
		const vec = spectrum.vecs[m];
		let flips = 0;
		g.edges.forEach(([a, b]) => {
			if (vec[a] * vec[b] < 0) flips++;
		});
		return flips;
	}, [spectrum, m, g]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes

		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		// edges
		ctx.strokeStyle = 'rgba(140,140,150,0.4)';
		ctx.lineWidth = 1.4;
		g.edges.forEach(([a, b]) => {
			ctx.beginPath();
			ctx.moveTo(g.pos[a].x, g.pos[a].y);
			ctx.lineTo(g.pos[b].x, g.pos[b].y);
			ctx.stroke();
		});

		const vec = spectrum.vecs[m];
		const maxAbs = Math.max(1e-9, ...vec.map((x) => Math.abs(x)));
		for (let i = 0; i < g.n; i++) {
			const v = vec[i] / maxAbs;
			ctx.beginPath();
			ctx.arc(g.pos[i].x, g.pos[i].y, R, 0, 2 * Math.PI);
			ctx.fillStyle = diverging(v);
			ctx.fill();
			ctx.lineWidth = 1.2;
			ctx.strokeStyle = 'rgba(150,150,160,0.7)';
			ctx.stroke();
		}

		// spectrum strip
		const stripY = HGT - SPEC_H + 8;
		const stripH = SPEC_H - 28;
		const maxLam = Math.max(1e-9, ...spectrum.values);
		const bw = (W - 40) / spectrum.values.length;
		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'left';
		ctx.textBaseline = 'alphabetic';
		ctx.globalAlpha = 0.85;
		ctx.fillText('eigenvalue spectrum  λ₀ … λₙ₋₁  (frequency →)', 20, stripY - 4);
		ctx.globalAlpha = 1;
		for (let k = 0; k < spectrum.values.length; k++) {
			const bh = (spectrum.values[k] / maxLam) * stripH;
			const x = 20 + k * bw;
			ctx.fillStyle = k === m ? '#2dd4bf' : 'rgba(140,140,150,0.5)';
			ctx.fillRect(x + 1, stripY + (stripH - bh), bw - 2, bh);
		}
	}, [g, spectrum, m]);

	function pickBar(e: MouseEvent<HTMLCanvasElement>) {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const rect = canvas.getBoundingClientRect();
		const mx = (e.clientX - rect.left) * (W / rect.width);
		const my = (e.clientY - rect.top) * (HGT / rect.height);
		if (my < HGT - SPEC_H) return;
		const bw = (W - 40) / spectrum.values.length;
		const k = Math.floor((mx - 20) / bw);
		if (k >= 0 && k < spectrum.values.length) setMode(k);
	}

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
						onClick={pickBar}
						aria-label="Graph coloured by a Laplacian eigenvector, with the eigenvalue spectrum"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__select">
						<span className="viz__slider-label">graph</span>
						<select
							value={kind}
							onChange={(e) => {
								setKind(e.target.value as GraphKind);
								setMode(1);
							}}
						>
							<option value="path">path</option>
							<option value="ring">ring</option>
							<option value="grid">grid</option>
							<option value="communities">two communities</option>
						</select>
					</label>

					<label className="viz__slider">
						<span className="viz__slider-label">
							mode k <b>{m}</b>
						</span>
						<input
							type="range"
							min={0}
							max={g.n - 1}
							step={1}
							value={m}
							onChange={(e) => setMode(parseInt(e.target.value, 10))}
						/>
					</label>

					<p className="viz__slider-label" style={{ lineHeight: 1.6 }}>
						eigenvalue <b>λ = {spectrum.values[m].toFixed(3)}</b>
						<br />
						sign flips across edges: <b>{signFlips}</b>
						<br />
						<span style={{ opacity: 0.8 }}>
							{m === 0 ? 'the smoothest mode (λ≈0)' : 'higher λ ⇒ more oscillation'}
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">L = I − D^(−1/2) A D^(−1/2) ,  L uₖ = λₖ uₖ</code>
		</div>
	);
}
