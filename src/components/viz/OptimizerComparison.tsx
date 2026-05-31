import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/**
 * Watch six optimizers descend the same 2-D loss surface from the same start.
 * SGD crawls and oscillates; momentum builds speed; the adaptive methods
 * (RMSprop, Adam) rescale each axis and cut straight to the minimum. AdamW adds
 * Adam's step plus a decoupled pull toward the origin — visible on the offset
 * bowl, whose minimum lies away from 0.
 */

type Vec = [number, number];

interface Surface {
	label: string;
	R: number;
	start: Vec;
	lr0: number;
	f: (x: number, y: number) => number;
	grad: (x: number, y: number) => Vec;
}

const SURFACES: Record<string, Surface> = {
	ravine: {
		label: 'ravine (ill-conditioned)',
		R: 5,
		start: [-4.3, 3.4],
		lr0: 0.1,
		f: (x, y) => 0.05 * x * x + 0.8 * y * y,
		grad: (x, y) => [0.1 * x, 1.6 * y],
	},
	saddle: {
		label: 'saddle point',
		R: 3,
		start: [0.05, 0.04],
		lr0: 0.05,
		f: (x, y) => 0.2 * x * x - 0.2 * y * y + 0.05 * y * y * y * y,
		grad: (x, y) => [0.4 * x, -0.4 * y + 0.2 * y * y * y],
	},
	bumpy: {
		label: 'bumpy (local minima)',
		R: 4,
		start: [2.8, 2.5],
		lr0: 0.08,
		f: (x, y) => 0.1 * (x * x + y * y) + 0.5 * Math.sin(1.3 * x) * Math.cos(1.3 * y),
		grad: (x, y) => [
			0.2 * x + 0.65 * Math.cos(1.3 * x) * Math.cos(1.3 * y),
			0.2 * y - 0.65 * Math.sin(1.3 * x) * Math.sin(1.3 * y),
		],
	},
	offset: {
		label: 'offset bowl (decay matters)',
		R: 5,
		start: [-3, 3],
		lr0: 0.12,
		f: (x, y) => 0.25 * (x - 2.4) * (x - 2.4) + 0.25 * (y + 1.6) * (y + 1.6),
		grad: (x, y) => [0.5 * (x - 2.4), 0.5 * (y + 1.6)],
	},
};

interface Optimizer {
	key: string;
	name: string;
	color: string;
	// returns next position given current state; mutates `s` accumulators
	step: (s: any, gradFn: (p: Vec) => Vec, lr: number, wd: number) => Vec;
}

const MU = 0.9;
const OPTIMIZERS: Optimizer[] = [
	{
		key: 'sgd',
		name: 'SGD',
		color: '#c2c2cc',
		step: (s, gf, lr) => {
			const g = gf(s.pos);
			return [s.pos[0] - lr * g[0], s.pos[1] - lr * g[1]];
		},
	},
	{
		key: 'mom',
		name: 'Momentum',
		color: '#e0b15e',
		step: (s, gf, lr) => {
			const g = gf(s.pos);
			s.v = [MU * s.v[0] - lr * g[0], MU * s.v[1] - lr * g[1]];
			return [s.pos[0] + s.v[0], s.pos[1] + s.v[1]];
		},
	},
	{
		key: 'nag',
		name: 'Nesterov',
		color: '#e0738f',
		step: (s, gf, lr) => {
			const look: Vec = [s.pos[0] + MU * s.v[0], s.pos[1] + MU * s.v[1]];
			const g = gf(look);
			s.v = [MU * s.v[0] - lr * g[0], MU * s.v[1] - lr * g[1]];
			return [s.pos[0] + s.v[0], s.pos[1] + s.v[1]];
		},
	},
	{
		key: 'rms',
		name: 'RMSprop',
		color: '#6db5e0',
		step: (s, gf, lr) => {
			const g = gf(s.pos);
			s.sq = [0.9 * s.sq[0] + 0.1 * g[0] * g[0], 0.9 * s.sq[1] + 0.1 * g[1] * g[1]];
			return [
				s.pos[0] - (lr * g[0]) / (Math.sqrt(s.sq[0]) + 1e-8),
				s.pos[1] - (lr * g[1]) / (Math.sqrt(s.sq[1]) + 1e-8),
			];
		},
	},
	{
		key: 'adam',
		name: 'Adam',
		color: '#60d3c5',
		step: (s, gf, lr) => {
			const g = gf(s.pos);
			s.t += 1;
			s.m = [0.9 * s.m[0] + 0.1 * g[0], 0.9 * s.m[1] + 0.1 * g[1]];
			s.v = [0.999 * s.v[0] + 0.001 * g[0] * g[0], 0.999 * s.v[1] + 0.001 * g[1] * g[1]];
			const mh: Vec = [s.m[0] / (1 - 0.9 ** s.t), s.m[1] / (1 - 0.9 ** s.t)];
			const vh: Vec = [s.v[0] / (1 - 0.999 ** s.t), s.v[1] / (1 - 0.999 ** s.t)];
			return [
				s.pos[0] - (lr * mh[0]) / (Math.sqrt(vh[0]) + 1e-8),
				s.pos[1] - (lr * mh[1]) / (Math.sqrt(vh[1]) + 1e-8),
			];
		},
	},
	{
		key: 'adamw',
		name: 'AdamW',
		color: '#b78bf0',
		step: (s, gf, lr, wd) => {
			const g = gf(s.pos);
			s.t += 1;
			s.m = [0.9 * s.m[0] + 0.1 * g[0], 0.9 * s.m[1] + 0.1 * g[1]];
			s.v = [0.999 * s.v[0] + 0.001 * g[0] * g[0], 0.999 * s.v[1] + 0.001 * g[1] * g[1]];
			const mh: Vec = [s.m[0] / (1 - 0.9 ** s.t), s.m[1] / (1 - 0.9 ** s.t)];
			const vh: Vec = [s.v[0] / (1 - 0.999 ** s.t), s.v[1] / (1 - 0.999 ** s.t)];
			// Adam step, then weight decay applied directly to the weights (decoupled).
			return [
				s.pos[0] - (lr * mh[0]) / (Math.sqrt(vh[0]) + 1e-8) - lr * wd * s.pos[0],
				s.pos[1] - (lr * mh[1]) / (Math.sqrt(vh[1]) + 1e-8) - lr * wd * s.pos[1],
			];
		},
	},
];

const SIZE = 320;
const MAX_STEPS = 200;

function freshState(start: Vec) {
	return { pos: [...start] as Vec, v: [0, 0] as Vec, sq: [0, 0] as Vec, m: [0, 0] as Vec, t: 0 };
}

export default function OptimizerComparison() {
	const [fnKey, setFnKey] = useState<keyof typeof SURFACES>('ravine');
	const [lr, setLr] = useState(SURFACES.ravine.lr0);
	const [wd, setWd] = useState(0.1);
	const [running, setRunning] = useState(false);
	const [, force] = useState(0);

	const surf = SURFACES[fnKey];
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const states = useRef(OPTIMIZERS.map(() => freshState(surf.start)));
	const trails = useRef<Vec[][]>(OPTIMIZERS.map(() => [surf.start]));
	const stepCount = useRef(0);
	const raf = useRef<number | null>(null);

	const xToPx = (x: number) => ((x + surf.R) / (2 * surf.R)) * SIZE;
	const yToPx = (y: number) => SIZE - ((y + surf.R) / (2 * surf.R)) * SIZE;

	// Draw the banded contour map once per surface.
	useEffect(() => {
		const c = canvasRef.current;
		if (!c) return;
		const ctx = c.getContext('2d');
		if (!ctx) return;
		const RES = 160;
		const img = ctx.createImageData(RES, RES);
		let lo = Infinity;
		let hi = -Infinity;
		const zs: number[] = [];
		for (let j = 0; j < RES; j++) {
			for (let i = 0; i < RES; i++) {
				const x = -surf.R + (2 * surf.R * i) / (RES - 1);
				const y = surf.R - (2 * surf.R * j) / (RES - 1);
				const z = surf.f(x, y);
				zs.push(z);
				lo = Math.min(lo, z);
				hi = Math.max(hi, z);
			}
		}
		for (let k = 0; k < zs.length; k++) {
			const t = (zs[k] - lo) / (hi - lo || 1);
			const band = Math.floor(t * 11) / 11; // topographic banding
			const g = Math.round(20 + band * 168);
			img.data[k * 4] = g;
			img.data[k * 4 + 1] = g;
			img.data[k * 4 + 2] = g + 4;
			img.data[k * 4 + 3] = 255;
		}
		const off = document.createElement('canvas');
		off.width = RES;
		off.height = RES;
		off.getContext('2d')!.putImageData(img, 0, 0);
		ctx.imageSmoothingEnabled = true;
		ctx.clearRect(0, 0, SIZE, SIZE);
		ctx.drawImage(off, 0, 0, SIZE, SIZE);
	}, [fnKey, surf]);

	const reset = useCallback(() => {
		setRunning(false);
		states.current = OPTIMIZERS.map(() => freshState(surf.start));
		trails.current = OPTIMIZERS.map(() => [surf.start]);
		stepCount.current = 0;
		force((n) => n + 1);
	}, [surf]);

	// reset whenever the surface changes
	useEffect(() => {
		setLr(surf.lr0);
		reset();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [fnKey]);

	const doStep = useCallback(() => {
		if (stepCount.current >= MAX_STEPS) return false;
		const gradFn = (p: Vec): Vec => surf.grad(p[0], p[1]);
		OPTIMIZERS.forEach((opt, i) => {
			const s = states.current[i];
			const last = s.pos;
			if (Math.hypot(last[0], last[1]) > surf.R * 1.8) return; // diverged — freeze
			const next = opt.step(s, gradFn, lr, wd);
			s.pos = next;
			trails.current[i].push(next);
		});
		stepCount.current += 1;
		force((n) => n + 1);
		return true;
	}, [surf, lr, wd]);

	// animation loop
	useEffect(() => {
		if (!running) return;
		const tick = () => {
			const ok = doStep();
			if (!ok || stepCount.current >= MAX_STEPS) {
				setRunning(false);
				return;
			}
			raf.current = requestAnimationFrame(tick);
		};
		raf.current = requestAnimationFrame(tick);
		return () => {
			if (raf.current) cancelAnimationFrame(raf.current);
		};
	}, [running, doStep]);

	const path = (trail: Vec[]) =>
		trail.map((p, i) => `${i ? 'L' : 'M'} ${xToPx(p[0]).toFixed(1)} ${yToPx(p[1]).toFixed(1)}`).join(' ');

	const losses = useMemo(
		() => OPTIMIZERS.map((_, i) => surf.f(states.current[i].pos[0], states.current[i].pos[1])),
		// recompute on every forced render
		// eslint-disable-next-line react-hooks/exhaustive-deps
		[fnKey, lr, running, stepCount.current],
	);

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: SIZE }}>
					<canvas ref={canvasRef} width={SIZE} height={SIZE} className="viz__canvas" />
					<svg className="opt-overlay" viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE}>
						{OPTIMIZERS.map((opt, i) => (
							<path key={opt.key} d={path(trails.current[i])} fill="none" stroke={opt.color} strokeWidth={1.8} opacity={0.95} />
						))}
						{/* start marker */}
						<circle cx={xToPx(surf.start[0])} cy={yToPx(surf.start[1])} r={4} className="opt-start" />
						{/* current heads */}
						{OPTIMIZERS.map((opt, i) => {
							const p = states.current[i].pos;
							return <circle key={opt.key} cx={xToPx(p[0])} cy={yToPx(p[1])} r={4.5} fill={opt.color} stroke="#0a0a0b" strokeWidth={1} />;
						})}
					</svg>
				</div>

				<div className="viz__controls">
					<label className="viz__select">
						<span className="viz__slider-label">surface</span>
						<select value={fnKey} onChange={(e) => setFnKey(e.target.value as keyof typeof SURFACES)}>
							{Object.entries(SURFACES).map(([k, s]) => (
								<option key={k} value={k}>{s.label}</option>
							))}
						</select>
					</label>
					<label className="viz__slider">
						<span className="viz__slider-label">learning rate η <b>{lr.toFixed(3)}</b></span>
						<input type="range" min={0.005} max={surf.lr0 * 3} step={0.005} value={lr} onChange={(e) => setLr(parseFloat(e.target.value))} />
					</label>
					<label className="viz__slider">
							<span className="viz__slider-label">weight decay λ (AdamW) <b>{wd.toFixed(2)}</b></span>
							<input type="range" min={0} max={0.25} step={0.01} value={wd} onChange={(e) => setWd(parseFloat(e.target.value))} />
						</label>
						<div className="cg__buttons">
						<button type="button" className="viz__btn" onClick={() => setRunning((r) => !r)} disabled={stepCount.current >= MAX_STEPS}>
							{running ? '❚❚ pause' : '▸ play'}
						</button>
						<button type="button" className="viz__btn" onClick={doStep} disabled={running || stepCount.current >= MAX_STEPS}>step</button>
						<button type="button" className="viz__btn" onClick={reset}>reset</button>
					</div>
					<ul className="opt-legend">
						{OPTIMIZERS.map((opt, i) => (
							<li key={opt.key}>
								<span className="opt-swatch" style={{ background: opt.color }} />
								<span className="opt-name">{opt.name}</span>
								<span className="opt-loss">{losses[i] < 1e3 ? losses[i].toFixed(3) : '—'}</span>
							</li>
						))}
					</ul>
					<p className="cg__caption">step {stepCount.current}/{MAX_STEPS} · loss at each optimizer's current point</p>
				</div>
			</div>
		</div>
	);
}
