import { useEffect, useRef, useState } from 'react';

/**
 * Bregman divergence, drawn as the gap a convex function opens above its own
 * tangent. Pick a convex generator φ, then drag the two points:
 *
 *   D_φ(p ‖ q) = φ(p) − φ(q) − φ'(q)·(p − q)
 *
 * i.e. the value of φ at p minus the first-order Taylor estimate of φ made at q.
 * Convexity keeps the tangent below the curve, so the gap — the teal segment —
 * is always ≥ 0, and vanishes only when p = q. Swapping p and q changes the
 * answer (except for ½x², the one symmetric generator), which is the whole point
 * of the double bar.
 */

type GenKey = 'sq' | 'neg' | 'burg';

interface Gen {
	fn: (x: number) => number; // φ
	d: (x: number) => number; // φ'
	dom: [number, number]; // p, q slider range
	plot: [number, number]; // x range drawn
	dflt: [number, number]; // default [p, q]
}

const GENS: Record<GenKey, Gen> = {
	sq: { fn: (x) => 0.5 * x * x, d: (x) => x, dom: [-2.5, 2.5], plot: [-2.7, 2.7], dflt: [1.5, -0.8] },
	neg: { fn: (x) => x * Math.log(x), d: (x) => Math.log(x) + 1, dom: [0.16, 3], plot: [0.05, 3], dflt: [2.4, 0.6] },
	burg: { fn: (x) => -Math.log(x), d: (x) => -1 / x, dom: [0.2, 3.3], plot: [0.1, 3.3], dflt: [2.3, 0.8] },
};

const ACCENT = '#60d3c5';

export default function BregmanDivergence() {
	const [genKey, setGenKey] = useState<GenKey>('sq');
	const gen = GENS[genKey];
	const [p, setP] = useState(gen.dflt[0]);
	const [q, setQ] = useState(gen.dflt[1]);

	// Responsive canvas: track the wrapper's width and size the canvas to fit.
	const wrapRef = useRef<HTMLDivElement | null>(null);
	const [cw, setCw] = useState(460);
	useEffect(() => {
		const el = wrapRef.current;
		if (!el) return;
		const ro = new ResizeObserver((entries) =>
			setCw(Math.max(260, Math.min(460, Math.floor(entries[0].contentRect.width)))),
		);
		ro.observe(el);
		return () => ro.disconnect();
	}, []);
	const W = cw;
	const H = Math.round(cw * 0.66);

	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	const div = (a: number, b: number) => gen.fn(a) - gen.fn(b) - gen.d(b) * (a - b);
	const Dpq = div(p, q);
	const Dqp = div(q, p);

	// Switching generators resets both points in one batch — no stale-domain frame.
	const changeGen = (k: GenKey) => {
		setGenKey(k);
		setP(GENS[k].dflt[0]);
		setQ(GENS[k].dflt[1]);
	};

	useEffect(() => {
		const cv = canvasRef.current;
		if (!cv) return;
		const ctx = cv.getContext('2d');
		if (!ctx) return;
		const dpr = Math.min(2, window.devicePixelRatio || 1);
		cv.width = W * dpr;
		cv.height = H * dpr;
		cv.style.width = `${W}px`;
		cv.style.height = `${H}px`;
		ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

		// Sample the curve and track its vertical extent…
		const [x0, x1] = gen.plot;
		const NS = 240;
		const xs: number[] = [];
		const ys: number[] = [];
		let ylo = Infinity;
		let yhi = -Infinity;
		for (let i = 0; i <= NS; i++) {
			const x = x0 + ((x1 - x0) * i) / NS;
			const y = gen.fn(x);
			if (Number.isFinite(y)) {
				xs.push(x);
				ys.push(y);
				if (y < ylo) ylo = y;
				if (y > yhi) yhi = y;
			}
		}
		// …then widen it so the gap at p is always on screen.
		const tanAt = (x: number) => gen.fn(q) + gen.d(q) * (x - q);
		const tanP = tanAt(p);
		for (const v of [gen.fn(p), tanP, gen.fn(q)]) {
			if (v < ylo) ylo = v;
			if (v > yhi) yhi = v;
		}
		const padY = (yhi - ylo) * 0.12 || 1;
		ylo -= padY;
		yhi += padY;
		const padX = (x1 - x0) * 0.06;
		const sx = (x: number) => ((x - (x0 - padX)) / (x1 + padX - (x0 - padX))) * W;
		const sy = (y: number) => H - ((y - ylo) / (yhi - ylo)) * H;

		ctx.clearRect(0, 0, W, H);
		ctx.fillStyle = '#0d0d0f';
		ctx.fillRect(0, 0, W, H);

		// baseline at y = 0 if visible
		const zeroShown = ylo < 0 && yhi > 0;
		if (zeroShown) {
			ctx.strokeStyle = '#26262b';
			ctx.lineWidth = 1;
			ctx.beginPath();
			ctx.moveTo(0, sy(0));
			ctx.lineTo(W, sy(0));
			ctx.stroke();
		}

		// dashed verticals dropping from q and p
		ctx.setLineDash([3, 3]);
		ctx.strokeStyle = '#2f2f35';
		for (const x of [q, p]) {
			ctx.beginPath();
			ctx.moveTo(sx(x), 0);
			ctx.lineTo(sx(x), H);
			ctx.stroke();
		}
		ctx.setLineDash([]);

		// tangent to φ at q, drawn full width
		ctx.strokeStyle = '#6c6c76';
		ctx.lineWidth = 1.5;
		ctx.setLineDash([6, 4]);
		ctx.beginPath();
		ctx.moveTo(sx(x0 - padX), sy(tanAt(x0 - padX)));
		ctx.lineTo(sx(x1 + padX), sy(tanAt(x1 + padX)));
		ctx.stroke();
		ctx.setLineDash([]);

		// the curve φ
		ctx.strokeStyle = '#e4e4e7';
		ctx.lineWidth = 2;
		ctx.beginPath();
		xs.forEach((x, i) => {
			const X = sx(x);
			const Y = sy(ys[i]);
			if (i) ctx.lineTo(X, Y);
			else ctx.moveTo(X, Y);
		});
		ctx.stroke();

		// the divergence: the gap at p between curve and tangent
		ctx.strokeStyle = ACCENT;
		ctx.lineWidth = 3;
		ctx.beginPath();
		ctx.moveTo(sx(p), sy(tanP));
		ctx.lineTo(sx(p), sy(gen.fn(p)));
		ctx.stroke();

		const dot = (x: number, y: number, c: string, r = 4) => {
			ctx.fillStyle = c;
			ctx.beginPath();
			ctx.arc(sx(x), sy(y), r, 0, Math.PI * 2);
			ctx.fill();
		};
		dot(q, gen.fn(q), '#c4c4cb');
		dot(p, tanP, '#6c6c76', 3);
		dot(p, gen.fn(p), ACCENT);

		// p / q tick labels
		ctx.fillStyle = '#9a9aa4';
		ctx.font = '12px ui-monospace, monospace';
		ctx.textAlign = 'center';
		const labY = zeroShown ? sy(0) + 14 : H - 6;
		ctx.fillText('q', sx(q), labY);
		ctx.fillText('p', sx(p), labY);
	}, [genKey, p, q, W, H, gen]);

	return (
		<div className="viz">
			<div ref={wrapRef} style={{ width: '100%' }}>
				<canvas
					ref={canvasRef}
					className="viz__canvas"
					style={{ display: 'block' }}
					aria-label="A convex generator and its tangent; the Bregman divergence is the vertical gap between them at p"
				/>
			</div>

			<div className="r3f-controls">
				<label className="viz__select" style={{ flex: '0 0 auto' }}>
					<span className="viz__slider-label">generator φ</span>
					<select value={genKey} onChange={(e) => changeGen(e.target.value as GenKey)}>
						<option value="sq">½x² · squared</option>
						<option value="neg">x ln x · entropy</option>
						<option value="burg">−ln x · Itakura–Saito</option>
					</select>
				</label>
				<label className="viz__slider" style={{ flex: '1 1 11rem' }}>
					<span className="viz__slider-label">
						p <b>{p.toFixed(2)}</b>
					</span>
					<input
						type="range"
						min={gen.dom[0]}
						max={gen.dom[1]}
						step={0.01}
						value={p}
						onChange={(e) => setP(parseFloat(e.target.value))}
					/>
				</label>
				<label className="viz__slider" style={{ flex: '1 1 11rem' }}>
					<span className="viz__slider-label">
						q <b>{q.toFixed(2)}</b>
					</span>
					<input
						type="range"
						min={gen.dom[0]}
						max={gen.dom[1]}
						step={0.01}
						value={q}
						onChange={(e) => setQ(parseFloat(e.target.value))}
					/>
				</label>
				<button type="button" className="viz__btn" onClick={() => changeGen(genKey)}>
					reset
				</button>
			</div>

			<p className="cg__caption">
				D<sub>φ</sub>(p ∥ q) = <b style={{ color: ACCENT }}>{Dpq.toFixed(3)}</b> — the teal gap.
				{'   '}Swapped, D<sub>φ</sub>(q ∥ p) = {Dqp.toFixed(3)}.{' '}
				{genKey === 'sq'
					? 'Equal — ½x² is the one symmetric generator.'
					: 'Different — the order of the arguments matters.'}
			</p>
		</div>
	);
}
