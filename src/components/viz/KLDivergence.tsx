import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Forward vs reverse KL, made visible. A fixed bimodal target p sits in gray;
 * you steer a single Gaussian q (mean μ, width σ) over it. Two readouts update
 * live:
 *
 *   forward   KL(p ‖ q) = ∫ p ln(p/q)   — penalizes q ≈ 0 where p > 0  → covers
 *   reverse   KL(q ‖ p) = ∫ q ln(q/p)   — penalizes q > 0 where p ≈ 0  → seeks
 *
 * "fit forward" snaps q to the moment-matched optimum (mean and variance of p) —
 * a wide bell straddling both modes. "fit reverse" snaps to the grid-searched
 * optimum — a narrow bell that locks onto a single mode and ignores the other.
 * Both are animated so you can watch mode-covering and mode-seeking happen.
 */

const ACCENT = '#60d3c5';
const COMPS = [
	{ w: 0.6, m: -2.5, s: 0.6 },
	{ w: 0.4, m: 2.8, s: 0.9 },
];
const SQ2PI = Math.sqrt(2 * Math.PI);
const npdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * SQ2PI);
const pPdf = (x: number) => COMPS.reduce((a, c) => a + c.w * npdf(x, c.m, c.s), 0);

// Integration grid for the KL integrals.
const GLO = -12;
const GHI = 12;
const GN = 720;
const GDX = (GHI - GLO) / (GN - 1);
const GX = Array.from({ length: GN }, (_, i) => GLO + (GHI - GLO) * (i / (GN - 1)));
const PG = GX.map(pPdf);

const klForward = (mu: number, sig: number) => {
	let s = 0;
	for (let i = 0; i < GN; i++) {
		const p = PG[i];
		if (p < 1e-12) continue;
		const q = Math.max(npdf(GX[i], mu, sig), 1e-12);
		s += p * Math.log(p / q);
	}
	return s * GDX;
};
const klReverse = (mu: number, sig: number) => {
	let s = 0;
	for (let i = 0; i < GN; i++) {
		const q = npdf(GX[i], mu, sig);
		if (q < 1e-12) continue;
		const p = Math.max(PG[i], 1e-12);
		s += q * Math.log(q / p);
	}
	return s * GDX;
};

// Forward optimum is exact: match the mean and variance of the target.
const FWD_MEAN = COMPS.reduce((a, c) => a + c.w * c.m, 0);
const FWD_VAR = COMPS.reduce((a, c) => a + c.w * (c.s * c.s + c.m * c.m), 0) - FWD_MEAN * FWD_MEAN;
const FWD = { mu: FWD_MEAN, sig: Math.sqrt(FWD_VAR) };

export default function KLDivergence() {
	const [mu, setMu] = useState(1.2);
	const [sig, setSig] = useState(1.3);
	const muRef = useRef(mu);
	const sigRef = useRef(sig);
	muRef.current = mu;
	sigRef.current = sig;

	// Reverse optimum has no closed form — grid-search it once.
	const REV = useMemo(() => {
		let best = { mu: 0, sig: 1, kl: Infinity };
		for (let m = -5; m <= 5.0001; m += 0.1) {
			for (let sg = 0.35; sg <= 3.0001; sg += 0.05) {
				const kl = klReverse(m, sg);
				if (kl < best.kl) best = { mu: m, sig: sg, kl };
			}
		}
		return best;
	}, []);

	const rafRef = useRef<number | undefined>(undefined);
	const animateTo = (tmu: number, tsig: number) => {
		if (rafRef.current) cancelAnimationFrame(rafRef.current);
		const smu = muRef.current;
		const ssig = sigRef.current;
		const dur = 650;
		let t0: number | null = null;
		const step = (ts: number) => {
			if (t0 === null) t0 = ts;
			const k = Math.min(1, (ts - t0) / dur);
			const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2; // easeInOutQuad
			setMu(smu + (tmu - smu) * e);
			setSig(ssig + (tsig - ssig) * e);
			if (k < 1) rafRef.current = requestAnimationFrame(step);
		};
		rafRef.current = requestAnimationFrame(step);
	};
	useEffect(() => () => {
		if (rafRef.current) cancelAnimationFrame(rafRef.current);
	}, []);

	const fwdKL = klForward(mu, sig);
	const revKL = klReverse(mu, sig);

	// Responsive canvas.
	const wrapRef = useRef<HTMLDivElement | null>(null);
	const [cw, setCw] = useState(620);
	useEffect(() => {
		const el = wrapRef.current;
		if (!el) return;
		const ro = new ResizeObserver((entries) =>
			setCw(Math.max(280, Math.min(620, Math.floor(entries[0].contentRect.width)))),
		);
		ro.observe(el);
		return () => ro.disconnect();
	}, []);
	const W = cw;
	const H = Math.round(cw * 0.42);

	const canvasRef = useRef<HTMLCanvasElement | null>(null);
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

		const X0 = -7;
		const X1 = 7;
		const peakQ = 1 / (sig * SQ2PI);
		const yMax = Math.max(0.62, peakQ * 1.06);
		const sx = (x: number) => ((x - X0) / (X1 - X0)) * W;
		const sy = (y: number) => H - (y / yMax) * (H - 6) - 1;

		ctx.clearRect(0, 0, W, H);
		ctx.fillStyle = '#0d0d0f';
		ctx.fillRect(0, 0, W, H);

		// faint ticks at the target's modes — anchors for "which mode?"
		ctx.setLineDash([2, 4]);
		ctx.strokeStyle = '#2a2a30';
		for (const c of COMPS) {
			ctx.beginPath();
			ctx.moveTo(sx(c.m), 0);
			ctx.lineTo(sx(c.m), H);
			ctx.stroke();
		}
		ctx.setLineDash([]);

		const NS = 260;
		const curve = (f: (x: number) => number) => {
			const pts: [number, number][] = [];
			for (let i = 0; i <= NS; i++) {
				const x = X0 + ((X1 - X0) * i) / NS;
				pts.push([sx(x), sy(f(x))]);
			}
			return pts;
		};

		// target p: filled
		const pc = curve(pPdf);
		ctx.beginPath();
		ctx.moveTo(pc[0][0], H);
		pc.forEach(([X, Y]) => ctx.lineTo(X, Y));
		ctx.lineTo(pc[pc.length - 1][0], H);
		ctx.closePath();
		ctx.fillStyle = 'rgba(150,150,160,0.18)';
		ctx.fill();
		ctx.strokeStyle = '#8a8a93';
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		pc.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
		ctx.stroke();

		// model q: teal
		const qc = curve((x) => npdf(x, mu, sig));
		ctx.beginPath();
		ctx.moveTo(qc[0][0], H);
		qc.forEach(([X, Y]) => ctx.lineTo(X, Y));
		ctx.lineTo(qc[qc.length - 1][0], H);
		ctx.closePath();
		ctx.fillStyle = 'rgba(96,211,197,0.10)';
		ctx.fill();
		ctx.strokeStyle = ACCENT;
		ctx.lineWidth = 2;
		ctx.beginPath();
		qc.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
		ctx.stroke();

		// legend
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'left';
		ctx.fillStyle = '#8a8a93';
		ctx.fillText('■ p  target', 10, 16);
		ctx.fillStyle = ACCENT;
		ctx.fillText('■ q  model', 10, 32);
	}, [mu, sig, W, H]);

	return (
		<div className="viz">
			<div ref={wrapRef} style={{ width: '100%' }}>
				<canvas
					ref={canvasRef}
					className="viz__canvas"
					style={{ display: 'block' }}
					aria-label="A bimodal target distribution p with an adjustable Gaussian q drawn over it"
				/>
			</div>

			<div className="r3f-controls">
				<label className="viz__slider" style={{ flex: '1 1 12rem' }}>
					<span className="viz__slider-label">
						mean μ <b>{mu.toFixed(2)}</b>
					</span>
					<input
						type="range"
						min={-6}
						max={6}
						step={0.01}
						value={mu}
						onChange={(e) => setMu(parseFloat(e.target.value))}
					/>
				</label>
				<label className="viz__slider" style={{ flex: '1 1 12rem' }}>
					<span className="viz__slider-label">
						width σ <b>{sig.toFixed(2)}</b>
					</span>
					<input
						type="range"
						min={0.3}
						max={4}
						step={0.01}
						value={sig}
						onChange={(e) => setSig(parseFloat(e.target.value))}
					/>
				</label>
				<button type="button" className="viz__btn" onClick={() => animateTo(FWD.mu, FWD.sig)}>
					▸ fit forward
				</button>
				<button type="button" className="viz__btn" onClick={() => animateTo(REV.mu, REV.sig)}>
					▸ fit reverse
				</button>
			</div>

			<p className="cg__caption">
				forward KL(p ∥ q) = <b style={{ color: ACCENT }}>{fwdKL.toFixed(3)}</b>
				{'    '}·{'    '}reverse KL(q ∥ p) ={' '}
				<b style={{ color: ACCENT }}>{revKL.toFixed(3)}</b>
			</p>
		</div>
	);
}
