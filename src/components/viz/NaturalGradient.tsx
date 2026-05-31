import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/**
 * Vanilla gradient descent vs the natural gradient, racing on the statistical
 * manifold of 1-D Gaussians q = N(mu, sigma). The loss is the forward KL to a
 * fixed target N(0, 1); the bullseye is the center.
 *
 *   vanilla   theta <- theta - lr * grad L
 *   natural   theta <- theta - lr * F^{-1} grad L,   F = diag(1/s^2, 2/s^2)
 *
 * The Fisher metric F^{-1} = diag(s^2, s^2/2) rescales the step by the local
 * geometry. The faint ellipses are the metric's "unit balls" (sets of equal KL
 * length): tiny where sigma is small — a small parameter move is a big change in
 * distribution — and large where sigma is big. Vanilla GD ignores them and
 * either explodes (small sigma, where d/dmu L ~ 1/s^2 blows up) or crawls (large
 * sigma); the natural gradient follows them and glides straight to the target.
 */

type Vec = [number, number];

const TARGET: Vec = [0, 1]; // mu*, sigma*
const MU_R = 3; // mu in [-3, 3]
const SIG_LO = 0.05;
const SIG_HI = 3;
const SIZE = 320;
const MAX_STEPS = 160;

const VANILLA = '#e0b15e';
const NATURAL = '#60d3c5';

// forward KL( N(mu*,s*) || N(mu,s) ), minimized at the target
function loss(mu: number, s: number): number {
	const A = TARGET[1] * TARGET[1] + (TARGET[0] - mu) ** 2;
	return Math.log(s / TARGET[1]) + A / (2 * s * s) - 0.5;
}
function grad(mu: number, s: number): Vec {
	const A = TARGET[1] * TARGET[1] + (TARGET[0] - mu) ** 2;
	return [(mu - TARGET[0]) / (s * s), 1 / s - A / (s * s * s)];
}
// natural gradient = F^{-1} grad, with F = diag(1/s^2, 2/s^2):
//   nat_mu = (mu - mu*)   (sigma-independent!),  nat_s = s/2 - A/(2s)
function natGrad(mu: number, s: number): Vec {
	const A = TARGET[1] * TARGET[1] + (TARGET[0] - mu) ** 2;
	return [mu - TARGET[0], s / 2 - A / (2 * s)];
}

interface Start {
	label: string;
	at: Vec;
}
const STARTS: Record<string, Start> = {
	tight: { label: 'tight σ — vanilla explodes', at: [-2.1, 0.4] },
	wide: { label: 'wide σ — vanilla crawls', at: [-2.4, 2.5] },
	high: { label: 'far + narrow', at: [2.2, 0.55] },
};

type Method = { key: string; name: string; color: string; nat: boolean };
const METHODS: Method[] = [
	{ key: 'van', name: 'vanilla GD', color: VANILLA, nat: false },
	{ key: 'nat', name: 'natural GD', color: NATURAL, nat: true },
];

function clampStep(p: Vec): { p: Vec; alive: boolean } {
	const s = Math.max(SIG_LO, p[1]);
	const alive = p[0] > -MU_R - 0.3 && p[0] < MU_R + 0.3 && p[1] < SIG_HI + 0.2 && p[1] > 0;
	return { p: [p[0], s], alive };
}

export default function NaturalGradient() {
	const [startKey, setStartKey] = useState<keyof typeof STARTS>('tight');
	const [lr, setLr] = useState(0.06);
	const [running, setRunning] = useState(false);
	const [showEllipses, setShowEllipses] = useState(true);
	const [, force] = useState(0);

	const start = STARTS[startKey].at;
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const pos = useRef<Vec[]>(METHODS.map(() => [...start] as Vec));
	const alive = useRef<boolean[]>(METHODS.map(() => true));
	const trails = useRef<Vec[][]>(METHODS.map(() => [[...start] as Vec]));
	const stepCount = useRef(0);
	const raf = useRef<number | null>(null);

	const xToPx = (mu: number) => ((mu + MU_R) / (2 * MU_R)) * SIZE;
	const yToPx = (s: number) => SIZE - ((s - SIG_LO) / (SIG_HI - SIG_LO)) * SIZE;
	const pxPerMu = SIZE / (2 * MU_R);
	const pxPerSig = SIZE / (SIG_HI - SIG_LO);

	// Banded contour map of the KL loss (drawn once).
	useEffect(() => {
		const c = canvasRef.current;
		if (!c) return;
		const ctx = c.getContext('2d');
		if (!ctx) return;
		const RES = 160;
		const img = ctx.createImageData(RES, RES);
		const zs: number[] = [];
		let lo = Infinity;
		let hi = -Infinity;
		for (let j = 0; j < RES; j++) {
			for (let i = 0; i < RES; i++) {
				const mu = -MU_R + (2 * MU_R * i) / (RES - 1);
				const s = SIG_HI - ((SIG_HI - SIG_LO) * j) / (RES - 1);
				const z = Math.min(6, loss(mu, s)); // clip the spike near sigma->0
				zs.push(z);
				lo = Math.min(lo, z);
				hi = Math.max(hi, z);
			}
		}
		for (let k = 0; k < zs.length; k++) {
			const t = (zs[k] - lo) / (hi - lo || 1);
			const band = Math.floor(t * 11) / 11;
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
	}, []);

	const reset = useCallback(() => {
		setRunning(false);
		pos.current = METHODS.map(() => [...start] as Vec);
		alive.current = METHODS.map(() => true);
		trails.current = METHODS.map(() => [[...start] as Vec]);
		stepCount.current = 0;
		force((n) => n + 1);
	}, [start]);

	useEffect(() => {
		reset();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [startKey]);

	const doStep = useCallback(() => {
		if (stepCount.current >= MAX_STEPS) return false;
		METHODS.forEach((m, i) => {
			if (!alive.current[i]) return;
			const [mu, s] = pos.current[i];
			const g = m.nat ? natGrad(mu, s) : grad(mu, s);
			const next: Vec = [mu - lr * g[0], s - lr * g[1]];
			const { p, alive: ok } = clampStep(next);
			pos.current[i] = p;
			alive.current[i] = ok;
			trails.current[i].push(p);
		});
		stepCount.current += 1;
		force((n) => n + 1);
		return alive.current.some(Boolean);
	}, [lr]);

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
		trail
			.map((p, i) => `${i ? 'L' : 'M'} ${xToPx(p[0]).toFixed(1)} ${yToPx(p[1]).toFixed(1)}`)
			.join(' ');

	// Sparse grid of Fisher "unit balls": equal-KL ellipses. Axis-aligned since F
	// is diagonal; semi-axes ∝ sigma (mu) and sigma/√2 (sigma), so they grow upward.
	const ellipses = useMemo(() => {
		const out: { cx: number; cy: number; rx: number; ry: number }[] = [];
		const R = 0.42; // Fisher radius -> visible size
		for (let mi = 0; mi < 5; mi++) {
			for (let si = 0; si < 4; si++) {
				const mu = -2.2 + (4.4 * mi) / 4;
				const s = 0.45 + (2.3 * si) / 3;
				out.push({
					cx: xToPx(mu),
					cy: yToPx(s),
					rx: R * s * pxPerMu,
					ry: ((R * s) / Math.SQRT2) * pxPerSig,
				});
			}
		}
		return out;
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, []);

	const losses = useMemo(
		() => METHODS.map((_, i) => loss(pos.current[i][0], pos.current[i][1])),
		// eslint-disable-next-line react-hooks/exhaustive-deps
		[startKey, lr, running, stepCount.current],
	);

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: SIZE }}>
					<canvas ref={canvasRef} width={SIZE} height={SIZE} className="viz__canvas" />
					<svg className="opt-overlay" viewBox={`0 0 ${SIZE} ${SIZE}`} width={SIZE} height={SIZE}>
						{showEllipses &&
							ellipses.map((e, i) => (
								<ellipse
									key={i}
									cx={e.cx}
									cy={e.cy}
									rx={e.rx}
									ry={e.ry}
									fill="none"
									stroke="rgba(150,150,160,0.30)"
									strokeWidth={1}
								/>
							))}
						{METHODS.map((m, i) => (
							<path
								key={m.key}
								d={path(trails.current[i])}
								fill="none"
								stroke={m.color}
								strokeWidth={2}
								opacity={0.95}
							/>
						))}
						{/* target bullseye */}
						<circle cx={xToPx(TARGET[0])} cy={yToPx(TARGET[1])} r={5} fill="none" stroke="#fff" strokeWidth={1.5} opacity={0.85} />
						<circle cx={xToPx(TARGET[0])} cy={yToPx(TARGET[1])} r={1.6} fill="#fff" opacity={0.85} />
						{/* start marker */}
						<circle cx={xToPx(start[0])} cy={yToPx(start[1])} r={4} className="opt-start" />
						{/* current heads */}
						{METHODS.map((m, i) => {
							const p = pos.current[i];
							return (
								<circle
									key={m.key}
									cx={xToPx(p[0])}
									cy={yToPx(p[1])}
									r={4.5}
									fill={m.color}
									stroke="#0a0a0b"
									strokeWidth={1}
									opacity={alive.current[i] ? 1 : 0.4}
								/>
							);
						})}
					</svg>
					<span className="viz__axis viz__axis--x">μ →</span>
					<span className="viz__axis viz__axis--y">σ →</span>
				</div>

				<div className="viz__controls">
					<label className="viz__select">
						<span className="viz__slider-label">start point</span>
						<select value={startKey} onChange={(e) => setStartKey(e.target.value as keyof typeof STARTS)}>
							{Object.entries(STARTS).map(([k, s]) => (
								<option key={k} value={k}>
									{s.label}
								</option>
							))}
						</select>
					</label>
					<label className="viz__slider">
						<span className="viz__slider-label">
							learning rate η <b>{lr.toFixed(3)}</b>
						</span>
						<input type="range" min={0.01} max={0.3} step={0.005} value={lr} onChange={(e) => setLr(parseFloat(e.target.value))} />
					</label>
					<div className="cg__buttons">
						<button type="button" className="viz__btn" onClick={() => setRunning((r) => !r)} disabled={stepCount.current >= MAX_STEPS}>
							{running ? '❚❚ pause' : '▸ play'}
						</button>
						<button type="button" className="viz__btn" onClick={doStep} disabled={running || stepCount.current >= MAX_STEPS}>
							step
						</button>
						<button type="button" className="viz__btn" onClick={reset}>
							reset
						</button>
					</div>
					<button
						type="button"
						className={`viz__btn${showEllipses ? ' viz__btn--on' : ''}`}
						onClick={() => setShowEllipses((s) => !s)}
					>
						Fisher ellipses
					</button>
					<ul className="opt-legend">
						{METHODS.map((m, i) => (
							<li key={m.key}>
								<span className="opt-swatch" style={{ background: m.color }} />
								<span className="opt-name">{m.name}</span>
								<span className="opt-loss">{alive.current[i] ? losses[i].toFixed(3) : 'diverged'}</span>
							</li>
						))}
					</ul>
					<p className="cg__caption">
						step {stepCount.current}/{MAX_STEPS} · loss = KL to target N(0, 1) · ⊕ = target
					</p>
				</div>
			</div>
		</div>
	);
}
