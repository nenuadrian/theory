import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Planning by the cross-entropy method, inside a (here exact) model.
 *
 * A point mass must reach the goal while avoiding a circular obstacle. CEM keeps a
 * Gaussian over whole action sequences: each iteration it samples a population,
 * rolls every sample through the model, keeps the lowest-cost "elite", and refits
 * the Gaussian to them. Scrub the iterations and watch the cloud of imagined
 * trajectories tighten onto a single plan that curves around the obstacle. This is
 * what PlaNet does at every step of control (executing only the first action, then
 * replanning — model-predictive control).
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const H = 24; // planning horizon
const N = 80; // population size
const ELITE = 12;
const ITERS = 14;
const SHOW = 36; // sample trajectories drawn per iteration
const UMAX = 0.28;

const START: [number, number] = [-1.6, 0];
const GOAL: [number, number] = [1.6, 0];
const OBS_C: [number, number] = [0, 0];
const OBS_R = 0.6;

function mulberry32(seed: number) {
	return () => {
		seed |= 0;
		seed = (seed + 0x6d2b79f5) | 0;
		let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
		t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

type Path = [number, number][];

function rollout(U: number[][]): { cost: number; path: Path } {
	let px = START[0],
		py = START[1];
	let pen = 0;
	let ctrl = 0;
	const path: Path = [[px, py]];
	for (let t = 0; t < H; t++) {
		const ux = Math.max(-UMAX, Math.min(UMAX, U[t][0]));
		const uy = Math.max(-UMAX, Math.min(UMAX, U[t][1]));
		px += ux;
		py += uy;
		const d = Math.hypot(px - OBS_C[0], py - OBS_C[1]);
		if (d < OBS_R + 0.1) pen += OBS_R + 0.1 - d; // soft barrier with margin
		ctrl += ux * ux + uy * uy;
		path.push([px, py]);
	}
	const dgx = px - GOAL[0],
		dgy = py - GOAL[1];
	const cost = 6 * (dgx * dgx + dgy * dgy) + 40 * pen + 0.5 * ctrl;
	return { cost, path };
}

interface IterRecord {
	samples: { path: Path; elite: boolean }[];
	meanPath: Path;
	bestCost: number;
}

export default function CEMPlanner() {
	const [iter, setIter] = useState(0);
	const [seed, setSeed] = useState(3);
	const [running, setRunning] = useState(false);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const raf = useRef<number | null>(null);

	// Run the full CEM optimization once, recording every iteration for scrubbing.
	const history = useMemo<IterRecord[]>(() => {
		const rand = mulberry32(seed * 2654435761);
		const normal = () => {
			let a = 0;
			while (a === 0) a = rand();
			const b = rand();
			return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
		};
		const mu = Array.from({ length: H }, () => [0, 0]);
		const sig = Array.from({ length: H }, () => [0.18, 0.18]);
		const out: IterRecord[] = [];
		for (let it = 0; it < ITERS; it++) {
			const samples: { U: number[][]; cost: number; path: Path }[] = [];
			for (let i = 0; i < N; i++) {
				const U = Array.from({ length: H }, (_, t) => [
					mu[t][0] + sig[t][0] * normal(),
					mu[t][1] + sig[t][1] * normal(),
				]);
				const { cost, path } = rollout(U);
				samples.push({ U, cost, path });
			}
			samples.sort((p, q) => p.cost - q.cost);
			const eliteSet = samples.slice(0, ELITE);
			// refit Gaussian to the elite
			for (let t = 0; t < H; t++) {
				for (let d = 0; d < 2; d++) {
					const vals = eliteSet.map((e) => e.U[t][d]);
					const m = vals.reduce((s, v) => s + v, 0) / ELITE;
					const va = vals.reduce((s, v) => s + (v - m) * (v - m), 0) / ELITE;
					mu[t][d] = m;
					sig[t][d] = Math.sqrt(va) + 1e-3;
				}
			}
			const meanPath = rollout(mu.map((r) => r.slice())).path;
			// record a readable subset of samples, flagging the elite
			const eliteIds = new Set(eliteSet.map((e) => e));
			const shown = samples.slice(0, SHOW).map((s) => ({
				path: s.path,
				elite: eliteIds.has(s),
			}));
			out.push({ samples: shown, meanPath, bestCost: eliteSet[0].cost });
		}
		return out;
	}, [seed]);

	useEffect(() => {
		if (!running) return;
		const tick = () => {
			setIter((i) => {
				if (i >= ITERS - 1) {
					setRunning(false);
					return i;
				}
				raf.current = window.setTimeout(
					() => requestAnimationFrame(tick),
					420,
				) as unknown as number;
				return i + 1;
			});
		};
		raf.current = window.setTimeout(() => requestAnimationFrame(tick), 420) as unknown as number;
		return () => {
			if (raf.current) clearTimeout(raf.current);
		};
	}, [running]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes
		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		const XR = [-2.1, 2.1],
			YR = [-1.35, 1.35];
		const pad = 16;
		const mapX = (x: number) => pad + ((x - XR[0]) / (XR[1] - XR[0])) * (W - 2 * pad);
		const mapY = (y: number) => (HGT - 24) - ((y - YR[0]) / (YR[1] - YR[0])) * (HGT - 24 - pad);
		const sx = (W - 2 * pad) / (XR[1] - XR[0]);

		const rec = history[Math.min(iter, history.length - 1)];

		// obstacle
		ctx.beginPath();
		ctx.arc(mapX(OBS_C[0]), mapY(OBS_C[1]), OBS_R * sx, 0, 2 * Math.PI);
		ctx.fillStyle = 'rgba(236,72,153,0.16)';
		ctx.fill();
		ctx.strokeStyle = 'rgba(236,72,153,0.6)';
		ctx.lineWidth = 1.5;
		ctx.stroke();

		// sampled trajectories (non-elite faint, elite teal)
		const drawPath = (path: Path, stroke: string, lw: number) => {
			ctx.strokeStyle = stroke;
			ctx.lineWidth = lw;
			ctx.beginPath();
			path.forEach(([x, y], i) => {
				const px = mapX(x),
					py = mapY(y);
				i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
			});
			ctx.stroke();
		};
		for (const s of rec.samples)
			if (!s.elite) drawPath(s.path, 'rgba(150,150,160,0.18)', 1);
		for (const s of rec.samples)
			if (s.elite) drawPath(s.path, 'rgba(45,212,191,0.5)', 1.2);
		// the mean plan
		drawPath(rec.meanPath, '#fb923c', 2.6);

		// start and goal
		const marker = (p: [number, number], color: string, r: number) => {
			ctx.beginPath();
			ctx.arc(mapX(p[0]), mapY(p[1]), r, 0, 2 * Math.PI);
			ctx.fillStyle = color;
			ctx.fill();
		};
		marker(START, '#c4c4cb', 5);
		ctx.beginPath();
		ctx.arc(mapX(GOAL[0]), mapY(GOAL[1]), 8, 0, 2 * Math.PI);
		ctx.strokeStyle = '#fbbf24';
		ctx.lineWidth = 2.5;
		ctx.stroke();
		marker(GOAL, 'rgba(251,191,36,0.5)', 4);

		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'left';
		ctx.fillText('start', mapX(START[0]) - 12, mapY(START[1]) - 10);
		ctx.fillText('goal', mapX(GOAL[0]) - 10, mapY(GOAL[1]) - 12);
		ctx.fillStyle = 'rgba(236,72,153,0.85)';
		ctx.textAlign = 'center';
		ctx.fillText('obstacle', mapX(OBS_C[0]), mapY(OBS_C[1]) + 4);
	}, [iter, history]);

	const rec = history[Math.min(iter, history.length - 1)];

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
						aria-label="CEM planning: a cloud of action sequences tightening onto a path around an obstacle"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							CEM iteration <b>{iter}</b>
						</span>
						<input
							type="range"
							min={0}
							max={ITERS - 1}
							step={1}
							value={iter}
							onChange={(e) => setIter(parseInt(e.target.value, 10))}
						/>
					</label>

					<div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								if (iter >= ITERS - 1) setIter(0);
								setRunning((r) => !r);
							}}
						>
							{running ? '❚❚ pause' : '▸ play'}
						</button>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								setRunning(false);
								setIter(0);
								setSeed((s) => s + 1);
							}}
						>
							resample
						</button>
					</div>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						best plan cost <b>{rec.bestCost.toFixed(2)}</b>
						<br />
						<span style={{ opacity: 0.85 }}>
							gray: sampled sequences · teal: elite · amber: the mean plan. The
							spread collapses as the elite agree.
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">
				aₜ* = arg min_a 𝔼[ Σₜ cost(ŝₜ, aₜ) ] under the model — refit 𝒩(μ, σ²) to the elite, repeat
			</code>
		</div>
	);
}
