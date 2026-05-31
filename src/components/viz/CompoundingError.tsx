import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Why a near-perfect one-step model is still useless over a long rollout.
 *
 * The environment is the Hénon map — a deterministic but *chaotic* 2-D system.
 * The "learned model" is the true map plus a tiny per-step error of magnitude ε.
 * Roll both forward from the same start: the trajectories track for a while, then
 * separate exponentially (the Lyapunov signature). Each 10× reduction in ε only
 * postpones the divergence by a roughly constant number of steps — it never
 * prevents it. This is the obstacle every world model must work around.
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const MAXH = 60;
const A = 1.4;
const B = 0.3;

// Hénon step.
function step(x: number, y: number): [number, number] {
	return [1 - A * x * x + y, B * x];
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

// A fixed, seeded sequence of per-step perturbation directions in [-1,1]^2, so
// the picture is reproducible and the divergence curve scales smoothly with ε.
const PERTURB: [number, number][] = (() => {
	const r = mulberry32(12345);
	return Array.from({ length: MAXH }, () => [r() * 2 - 1, r() * 2 - 1] as [number, number]);
})();

// The Hénon attractor, drawn faintly as the backdrop.
const ATTRACTOR: [number, number][] = (() => {
	let x = 0.1,
		y = 0.3;
	const pts: [number, number][] = [];
	for (let i = 0; i < 60; i++) [x, y] = step(x, y); // burn-in
	for (let i = 0; i < 2600; i++) {
		[x, y] = step(x, y);
		pts.push([x, y]);
	}
	return pts;
})();

const X0: [number, number] = [0.1, 0.1];
// Phase-space window (attractor spans x∈[-1.3,1.3], y∈[-0.4,0.4]).
const XR = [-1.55, 1.55];
const YR = [-0.62, 0.62];

export default function CompoundingError() {
	const [H, setH] = useState(20);
	const [epsExp, setEpsExp] = useState(-3); // ε = 10^epsExp, range −4 … −1
	const [running, setRunning] = useState(false);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const raf = useRef<number | null>(null);

	const eps = Math.pow(10, epsExp);

	// Roll true and model trajectories forward, recording the divergence at each
	// step. Recomputed only when ε changes.
	const { truePath, modelPath, divs } = useMemo(() => {
		const truePath: [number, number][] = [X0.slice() as [number, number]];
		const modelPath: [number, number][] = [X0.slice() as [number, number]];
		const divs: number[] = [0];
		let tx = X0[0],
			ty = X0[1],
			mx = X0[0],
			my = X0[1];
		for (let t = 0; t < MAXH; t++) {
			[tx, ty] = step(tx, ty);
			const [sx, sy] = step(mx, my);
			mx = sx + eps * PERTURB[t][0];
			my = sy + eps * PERTURB[t][1];
			truePath.push([tx, ty]);
			modelPath.push([mx, my]);
			divs.push(Math.hypot(tx - mx, ty - my));
		}
		return { truePath, modelPath, divs };
	}, [eps]);

	// Horizon at which the model and reality first differ by more than 0.5 — about
	// a fifth of the attractor's width, i.e. "no longer the same trajectory".
	const breakH = useMemo(() => {
		for (let i = 1; i <= MAXH; i++) if (divs[i] > 0.5) return i;
		return -1;
	}, [divs]);

	useEffect(() => {
		if (!running) return;
		const tick = () => {
			setH((h) => {
				if (h >= MAXH) {
					setRunning(false);
					return h;
				}
				raf.current = requestAnimationFrame(tick);
				return h + 1;
			});
		};
		raf.current = requestAnimationFrame(tick);
		return () => {
			if (raf.current) cancelAnimationFrame(raf.current);
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

		// ---- left: phase space ---------------------------------------------
		const px0 = 12,
			px1 = W * 0.6,
			py0 = 14,
			py1 = HGT - 26;
		const mapX = (x: number) => px0 + ((x - XR[0]) / (XR[1] - XR[0])) * (px1 - px0);
		const mapY = (y: number) => py1 - ((y - YR[0]) / (YR[1] - YR[0])) * (py1 - py0);

		// attractor backdrop
		ctx.fillStyle = 'rgba(150,150,160,0.16)';
		for (const [x, y] of ATTRACTOR) {
			ctx.fillRect(mapX(x), mapY(y), 1, 1);
		}

		// the two rollouts up to step H
		const drawPath = (path: [number, number][], color: string) => {
			ctx.strokeStyle = color;
			ctx.lineWidth = 1.6;
			ctx.beginPath();
			for (let i = 0; i <= H; i++) {
				const [x, y] = path[i];
				if (i === 0) ctx.moveTo(mapX(x), mapY(y));
				else ctx.lineTo(mapX(x), mapY(y));
			}
			ctx.stroke();
		};
		drawPath(truePath, 'rgba(45,212,191,0.95)');
		drawPath(modelPath, 'rgba(251,146,60,0.95)');

		// current positions
		const dot = (p: [number, number], color: string) => {
			ctx.beginPath();
			ctx.arc(mapX(p[0]), mapY(p[1]), 4.5, 0, 2 * Math.PI);
			ctx.fillStyle = color;
			ctx.fill();
		};
		dot(truePath[H], '#2dd4bf');
		dot(modelPath[H], '#fb923c');
		// the gap between them
		ctx.strokeStyle = 'rgba(236,72,153,0.8)';
		ctx.lineWidth = 1.4;
		ctx.setLineDash([3, 3]);
		ctx.beginPath();
		ctx.moveTo(mapX(truePath[H][0]), mapY(truePath[H][1]));
		ctx.lineTo(mapX(modelPath[H][0]), mapY(modelPath[H][1]));
		ctx.stroke();
		ctx.setLineDash([]);

		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'left';
		ctx.fillText('phase space — Hénon attractor', px0, HGT - 10);
		// legend
		ctx.fillStyle = '#2dd4bf';
		ctx.fillText('— reality', px0, 12 + 0);
		ctx.fillStyle = '#fb923c';
		ctx.fillText('— model', px0 + 66, 12 + 0);

		// ---- right: divergence vs step (log scale) -------------------------
		const ex0 = W * 0.66,
			ex1 = W - 12,
			ey0 = 26,
			ey1 = HGT - 26;
		const logMin = -5,
			logMax = Math.log10(3);
		const mapStep = (s: number) => ex0 + (s / MAXH) * (ex1 - ex0);
		const mapLog = (d: number) => {
			const l = Math.log10(Math.max(d, 1e-6));
			return ey1 - ((l - logMin) / (logMax - logMin)) * (ey1 - ey0);
		};
		// frame + decade gridlines
		ctx.strokeStyle = 'rgba(140,140,150,0.25)';
		ctx.lineWidth = 1;
		ctx.strokeRect(ex0, ey0, ex1 - ex0, ey1 - ey0);
		ctx.fillStyle = 'rgba(196,196,203,0.55)';
		ctx.font = '9px ui-monospace, monospace';
		ctx.textAlign = 'right';
		for (let e = -4; e <= 0; e++) {
			const yy = mapLog(Math.pow(10, e));
			ctx.strokeStyle = 'rgba(140,140,150,0.16)';
			ctx.beginPath();
			ctx.moveTo(ex0, yy);
			ctx.lineTo(ex1, yy);
			ctx.stroke();
			ctx.fillText(`1e${e}`, ex0 - 3, yy + 3);
		}
		// the "indistinguishable" threshold at 0.5
		const ty = mapLog(0.5);
		ctx.strokeStyle = 'rgba(236,72,153,0.5)';
		ctx.setLineDash([4, 3]);
		ctx.beginPath();
		ctx.moveTo(ex0, ty);
		ctx.lineTo(ex1, ty);
		ctx.stroke();
		ctx.setLineDash([]);

		// divergence curve up to H
		ctx.strokeStyle = '#fb923c';
		ctx.lineWidth = 2;
		ctx.beginPath();
		for (let s = 1; s <= H; s++) {
			const xx = mapStep(s);
			const yy = mapLog(divs[s]);
			if (s === 1) ctx.moveTo(xx, yy);
			else ctx.lineTo(xx, yy);
		}
		ctx.stroke();
		// marker
		ctx.beginPath();
		ctx.arc(mapStep(H), mapLog(divs[H] || 1e-6), 3.5, 0, 2 * Math.PI);
		ctx.fillStyle = '#fb923c';
		ctx.fill();

		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'left';
		ctx.fillText('‖model − reality‖  (log)', ex0, ey0 - 8);
		ctx.textAlign = 'center';
		ctx.fillText('rollout step  →', (ex0 + ex1) / 2, HGT - 10);
	}, [H, truePath, modelPath, divs]);

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
						aria-label="A learned model's rollout diverging exponentially from reality on a chaotic system"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							per-step error ε <b>{eps.toExponential(0)}</b>
						</span>
						<input
							type="range"
							min={-4}
							max={-1}
							step={0.25}
							value={epsExp}
							onChange={(e) => setEpsExp(parseFloat(e.target.value))}
						/>
					</label>

					<label className="viz__slider">
						<span className="viz__slider-label">
							rollout horizon H <b>{H}</b>
						</span>
						<input
							type="range"
							min={0}
							max={MAXH}
							step={1}
							value={H}
							onChange={(e) => setH(parseInt(e.target.value, 10))}
						/>
					</label>

					<div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								if (H >= MAXH) setH(0);
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
								setH(0);
							}}
						>
							reset
						</button>
					</div>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						gap now <b>{(divs[H] ?? 0).toExponential(2)}</b>
						<br />
						<span style={{ opacity: 0.85 }}>
							{breakH < 0
								? 'tracks reality the whole way'
								: H < breakH
									? `still tracking — breaks at step ${breakH}`
									: 'model and reality now unrelated'}
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">
				‖x̂_H − x_H‖ ≲ ε · (Lᴴ − 1)/(L − 1)  →  grows exponentially when L &gt; 1
			</code>
		</div>
	);
}
