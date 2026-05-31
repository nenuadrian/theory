import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Dreaming with a mixture-density transition.
 *
 * The latent follows a 2-component mixture each step: one component drifts toward
 * +1, the other toward −1, with state-dependent weights (a soft double well).
 * Sampling many rollouts from z0 = 0 produces a *bifurcating cloud* of imagined
 * futures — genuinely multimodal. The deterministic predictor a plain MSE network
 * would learn is the conditional mean E[z'|z]; rolled out it threads the empty
 * gap between the modes, predicting a "future" that never occurs. Temperature τ
 * scales the sampling noise: the dream's diversity.
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const M = 140; // number of imagined trajectories
const T = 14; // rollout horizon
const SIG = 0.11; // base step noise
const ZLIM = 1.75;

function mulberry32(seed: number) {
	return () => {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const piUp = (z: number) => sigmoid(3 * z); // weight of the "up" component
const up = (z: number) => 0.5 * z + 0.5; // drift toward +1
const down = (z: number) => 0.5 * z - 0.5; // drift toward −1

export default function DreamRollout() {
	const [tau, setTau] = useState(1.0);
	const [seed, setSeed] = useState(7);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	// Fixed random draws per (trajectory, step): a uniform for the component choice
	// and a normal for the noise. Seeded so τ scales spread without reshuffling.
	const draws = useMemo(() => {
		const r = mulberry32(seed * 2654435761);
		const u: number[][] = [];
		const g: number[][] = [];
		for (let i = 0; i < M; i++) {
			const ur: number[] = [];
			const gr: number[] = [];
			for (let t = 0; t < T; t++) {
				ur.push(r());
				// Box–Muller normal
				let a = 0;
				while (a === 0) a = r();
				const b = r();
				gr.push(Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b));
			}
			u.push(ur);
			g.push(gr);
		}
		return { u, g };
	}, [seed]);

	// Roll out all trajectories and the deterministic mean predictor.
	const { trajs, mean } = useMemo(() => {
		const trajs: number[][] = [];
		for (let i = 0; i < M; i++) {
			const path = [0];
			let z = 0;
			for (let t = 0; t < T; t++) {
				const chooseUp = draws.u[i][t] < piUp(z);
				const center = chooseUp ? up(z) : down(z);
				z = center + tau * SIG * draws.g[i][t];
				path.push(z);
			}
			trajs.push(path);
		}
		const mean = [0];
		let m = 0;
		for (let t = 0; t < T; t++) {
			m = piUp(m) * up(m) + (1 - piUp(m)) * down(m);
			mean.push(m);
		}
		return { trajs, mean };
	}, [draws, tau]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes
		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		const x0 = 40,
			x1 = W - 84,
			y0 = 16,
			y1 = HGT - 28;
		const mapX = (t: number) => x0 + (t / T) * (x1 - x0);
		const mapY = (z: number) => y1 - ((z + ZLIM) / (2 * ZLIM)) * (y1 - y0);

		// zero axis
		ctx.strokeStyle = 'rgba(140,140,150,0.3)';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(x0, mapY(0));
		ctx.lineTo(x1, mapY(0));
		ctx.stroke();

		// imagined trajectories, coloured by branch (sign of the latent)
		ctx.lineWidth = 1;
		for (const path of trajs) {
			for (let t = 0; t < T; t++) {
				const za = path[t],
					zb = path[t + 1];
				const s = (za + zb) / 2;
				ctx.strokeStyle =
					s >= 0 ? 'rgba(45,212,191,0.22)' : 'rgba(236,72,153,0.22)';
				ctx.beginPath();
				ctx.moveTo(mapX(t), mapY(za));
				ctx.lineTo(mapX(t + 1), mapY(zb));
				ctx.stroke();
			}
		}

		// deterministic mean predictor
		ctx.strokeStyle = '#fb923c';
		ctx.lineWidth = 2.5;
		ctx.beginPath();
		mean.forEach((z, t) => {
			const px = mapX(t),
				py = mapY(z);
			t === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
		});
		ctx.stroke();

		// marginal histogram of z_T on the right
		const hx0 = W - 78,
			hx1 = W - 12;
		const BINS = 24;
		const counts = new Array(BINS).fill(0);
		for (const path of trajs) {
			const z = path[T];
			let bi = Math.floor(((z + ZLIM) / (2 * ZLIM)) * BINS);
			bi = Math.max(0, Math.min(BINS - 1, bi));
			counts[bi]++;
		}
		const maxC = Math.max(1, ...counts);
		for (let b = 0; b < BINS; b++) {
			const zc = -ZLIM + ((b + 0.5) / BINS) * 2 * ZLIM;
			const yy = mapY(zc);
			const wbar = (counts[b] / maxC) * (hx1 - hx0);
			ctx.fillStyle = zc >= 0 ? 'rgba(45,212,191,0.7)' : 'rgba(236,72,153,0.7)';
			ctx.fillRect(hx0, yy - (y1 - y0) / BINS / 2, wbar, (y1 - y0) / BINS - 1);
		}

		// labels
		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'center';
		ctx.fillText('imagined latent  zₜ   vs   time →', (x0 + x1) / 2, HGT - 9);
		ctx.textAlign = 'left';
		ctx.fillStyle = '#fb923c';
		ctx.fillText('— deterministic E[z′|z]', x0 + 4, y0 + 4);
		ctx.fillStyle = 'rgba(196,196,203,0.7)';
		ctx.textAlign = 'center';
		ctx.fillText('p(z_T)', (hx0 + hx1) / 2, y0 + 4);
	}, [trajs, mean]);

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
						aria-label="Imagined latent rollouts bifurcating into a multimodal cloud"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							temperature τ <b>{tau.toFixed(2)}</b>
						</span>
						<input
							type="range"
							min={0.2}
							max={1.6}
							step={0.05}
							value={tau}
							onChange={(e) => setTau(parseFloat(e.target.value))}
						/>
					</label>

					<button
						type="button"
						className="viz__btn"
						onClick={() => setSeed((s) => s + 1)}
					>
						▸ resample dream
					</button>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						<span style={{ opacity: 0.85 }}>
							The future <b>bifurcates</b> — two modes, drawn in teal and pink. The
							deterministic mean (amber) runs through the gap between them, a future
							that never happens.
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">
				p(zₜ₊₁ | zₜ) = Σₖ πₖ(zₜ) · 𝒩( μₖ(zₜ), (τσₖ)² )   —   a mixture, not a point
			</code>
		</div>
	);
}
