import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * The λ-return as a geometric mixture of n-step returns.
 *
 * Along an imagined rollout, each n-step return rolls n real rewards and then
 * bootstraps with the critic v(s_n). The λ-return averages them with weights
 * w_n = (1−λ)λ^{n−1} (the tail absorbing the remainder). λ = 0 places all weight
 * on n = 1 — bootstrap immediately, the low-variance, critic-biased TD(0) target.
 * λ = 1 places all weight on the full rollout — the unbiased, high-variance Monte
 * Carlo return. Dreamer learns its critic on this target with λ ≈ 0.95.
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const H = 8; // rollout length (rewards r_0..r_{H-1}, nodes s_0..s_H)
const GAMMA = 0.95;
const BIAS = 0.65; // the critic systematically under-estimates

const R = [1.0, 0.6, 0.9, 0.4, 0.8, 0.5, 0.7, 0.3]; // per-step rewards

// True values and the biased critic.
const VTRUE: number[] = (() => {
	const v = new Array(H + 1).fill(0);
	for (let t = H - 1; t >= 0; t--) v[t] = R[t] + GAMMA * v[t + 1];
	return v;
})();
const VHAT = VTRUE.map((v, t) => (t === H ? 0 : BIAS * v));

export default function LambdaReturns() {
	const [lam, setLam] = useState(0.7);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	const { weights, Glam, horizon } = useMemo(() => {
		// geometric weights over n = 1..H, tail absorbs the remainder
		const weights = new Array(H + 1).fill(0); // index by n (1..H)
		for (let n = 1; n < H; n++) weights[n] = (1 - lam) * Math.pow(lam, n - 1);
		weights[H] = Math.pow(lam, H - 1);
		// n-step returns and the λ-return
		let Glam = 0;
		let horizon = 0;
		for (let n = 1; n <= H; n++) {
			let g = 0;
			for (let k = 0; k < n; k++) g += Math.pow(GAMMA, k) * R[k];
			if (n < H) g += Math.pow(GAMMA, n) * VHAT[n]; // bootstrap (terminal at H)
			Glam += weights[n] * g;
			horizon += weights[n] * n;
		}
		return { weights, Glam, horizon };
	}, [lam]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes
		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		const x0 = 30,
			x1 = W - 24;
		const nodeX = (t: number) => x0 + (t / H) * (x1 - x0);
		const chainY = 70;

		// ---- the imagined rollout chain ------------------------------------
		ctx.strokeStyle = 'rgba(140,140,150,0.5)';
		ctx.lineWidth = 1.4;
		for (let t = 0; t < H; t++) {
			ctx.beginPath();
			ctx.moveTo(nodeX(t) + 9, chainY);
			ctx.lineTo(nodeX(t + 1) - 9, chainY);
			ctx.stroke();
			// reward on the edge
			ctx.fillStyle = 'rgba(196,196,203,0.85)';
			ctx.font = '9px ui-monospace, monospace';
			ctx.textAlign = 'center';
			ctx.fillText(`r=${R[t].toFixed(1)}`, (nodeX(t) + nodeX(t + 1)) / 2, chainY - 7);
		}
		const wmax = Math.max(...weights);
		for (let t = 0; t <= H; t++) {
			// shade node by how much the λ-return bootstraps there (weight at n=t)
			const wn = t >= 1 ? weights[t] / wmax : 0;
			ctx.beginPath();
			ctx.arc(nodeX(t), chainY, 9, 0, 2 * Math.PI);
			ctx.fillStyle = t === H ? 'rgba(120,120,130,0.5)' : `rgba(45,212,191,${0.12 + 0.8 * wn})`;
			ctx.fill();
			ctx.strokeStyle = 'rgba(150,150,160,0.7)';
			ctx.lineWidth = 1;
			ctx.stroke();
			ctx.fillStyle = fg;
			ctx.font = '9px ui-monospace, monospace';
			ctx.textAlign = 'center';
			ctx.fillText(`s${t}`, nodeX(t), chainY + 0.5);
			// critic value below
			ctx.fillStyle = 'rgba(251,146,60,0.85)';
			ctx.fillText(`v=${VHAT[t].toFixed(1)}`, nodeX(t), chainY + 22);
		}
		ctx.fillStyle = fg;
		ctx.textAlign = 'left';
		ctx.font = '10px ui-monospace, monospace';
		ctx.fillText('imagined rollout  (teal = bootstrap weight, amber = critic v)', x0, 28);

		// effective-horizon marker
		const hx = nodeX(horizon);
		ctx.strokeStyle = 'rgba(251,191,36,0.8)';
		ctx.setLineDash([4, 3]);
		ctx.lineWidth = 1.5;
		ctx.beginPath();
		ctx.moveTo(hx, chainY + 30);
		ctx.lineTo(hx, chainY - 26);
		ctx.stroke();
		ctx.setLineDash([]);
		ctx.fillStyle = '#fbbf24';
		ctx.textAlign = 'center';
		ctx.font = '9px ui-monospace, monospace';
		ctx.fillText(`effective horizon ≈ ${horizon.toFixed(1)}`, hx, chainY - 30);

		// ---- weight bars w_n -----------------------------------------------
		const by0 = 150,
			by1 = HGT - 30;
		ctx.fillStyle = fg;
		ctx.font = '10px ui-monospace, monospace';
		ctx.textAlign = 'left';
		ctx.fillText('weight on each n-step return   wₙ = (1−λ)λⁿ⁻¹', x0, by0 - 10);
		const bw = ((x1 - x0) / H) * 0.6;
		for (let n = 1; n <= H; n++) {
			const bx = nodeX(n) - bw / 2;
			const bh = (weights[n] / wmax) * (by1 - by0);
			ctx.fillStyle = n === H ? 'rgba(96,211,197,0.6)' : 'rgba(45,212,191,0.85)';
			ctx.fillRect(bx, by1 - bh, bw, bh);
			ctx.fillStyle = 'rgba(196,196,203,0.7)';
			ctx.font = '9px ui-monospace, monospace';
			ctx.textAlign = 'center';
			ctx.fillText(`${n}`, nodeX(n), by1 + 12);
		}
		ctx.strokeStyle = 'rgba(140,140,150,0.3)';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(x0, by1);
		ctx.lineTo(x1, by1);
		ctx.stroke();
		ctx.fillStyle = 'rgba(196,196,203,0.6)';
		ctx.textAlign = 'right';
		ctx.fillText('n →', x1, by1 + 12);
	}, [lam, weights, horizon]);

	const regime = lam < 0.05 ? 'TD(0) — pure bootstrap' : lam > 0.95 ? 'Monte Carlo — full rollout' : 'mixture of n-step returns';

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
						aria-label="The lambda-return as a geometric mixture of n-step returns"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							λ <b>{lam.toFixed(2)}</b>
						</span>
						<input
							type="range"
							min={0}
							max={1}
							step={0.01}
							value={lam}
							onChange={(e) => setLam(parseFloat(e.target.value))}
						/>
					</label>

					<div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
						<button type="button" className="viz__btn" onClick={() => setLam(0)}>
							TD(0)
						</button>
						<button type="button" className="viz__btn" onClick={() => setLam(0.95)}>
							λ=0.95
						</button>
						<button type="button" className="viz__btn" onClick={() => setLam(1)}>
							MC
						</button>
					</div>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						{regime}
						<br />
						target Gλ <b>{Glam.toFixed(2)}</b>
						<br />
						<span style={{ opacity: 0.85 }}>
							true V*(s₀) = {VTRUE[0].toFixed(2)} · the biased critic drags small-λ
							targets low; λ→1 removes the bias but adds variance.
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">
				Gλ₀ = Σₙ (1−λ)λⁿ⁻¹ G⁽ⁿ⁾,   G⁽ⁿ⁾ = Σₖ₌₀ⁿ⁻¹ γᵏ rₖ + γⁿ v(sₙ)
			</code>
		</div>
	);
}
