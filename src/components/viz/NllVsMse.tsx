import { useState } from 'react';

/**
 * Why classification uses negative log-likelihood (cross-entropy), not squared
 * error. Left: the two losses as a function of the predicted probability.
 * Right — the punchline — the gradient w.r.t. the logit z (p = σ(z)):
 *
 *   NLL:  dL/dz = p − y                  (clean, large when wrong)
 *   MSE:  dL/dz = 2(p − y)·p(1 − p)       (vanishes when p → 0 or 1)
 *
 * So a confidently-wrong sigmoid unit barely learns under MSE, but learns fast
 * under NLL.
 */

const TEAL = '#60d3c5'; // NLL
const AMBER = '#e0b15e'; // MSE
const W = 340;
const H = 200;
const PAD = 34;
const EPS = 1e-3;

const nll = (p: number, y: number) => -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
const mse = (p: number, y: number) => (p - y) * (p - y);
const nllGrad = (p: number, y: number) => Math.abs(p - y);
const mseGrad = (p: number, y: number) => Math.abs(2 * (p - y) * p * (1 - p));

function Panel({
	title,
	y,
	yMax,
	fns,
	p,
	ann,
}: {
	title: string;
	y: number;
	yMax: number;
	fns: { color: string; f: (p: number) => number }[];
	p: number;
	ann?: string;
}) {
	const xToPx = (px: number) => PAD + px * (W - 2 * PAD);
	const yToPx = (v: number) => H - PAD - (Math.min(v, yMax) / yMax) * (H - 2 * PAD);
	const N = 200;
	const pathOf = (f: (p: number) => number) => {
		let d = '';
		for (let i = 0; i <= N; i++) {
			const px = EPS + (i / N) * (1 - 2 * EPS);
			d += `${i ? 'L' : 'M'} ${xToPx(px).toFixed(1)} ${yToPx(f(px)).toFixed(1)} `;
		}
		return d;
	};
	return (
		<svg className="nll" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
			{/* axes */}
			<line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} className="nll__axis" />
			<line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} className="nll__axis" />
			<text x={W / 2} y={H - 6} className="nll__axislbl" textAnchor="middle">predicted probability p</text>
			<text x={(W - 2 * PAD) / 2 + PAD} y={18} className="nll__title" textAnchor="middle">{title}</text>
			{/* current-p guide */}
			<line x1={xToPx(p)} y1={PAD} x2={xToPx(p)} y2={H - PAD} className="nll__guide" />
			{fns.map((fn, i) => (
				<path key={i} d={pathOf(fn.f)} fill="none" stroke={fn.color} strokeWidth={2} />
			))}
			{fns.map((fn, i) => (
				<circle key={`d${i}`} cx={xToPx(p)} cy={yToPx(fn.f(p))} r={4} fill={fn.color} stroke="#0a0a0b" strokeWidth={1} />
			))}
			{ann && (
				<text x={W - PAD} y={PAD + 12} className="nll__ann" textAnchor="end">{ann}</text>
			)}
		</svg>
	);
}

export default function NllVsMse() {
	const [p, setP] = useState(0.08);
	const [y, setY] = useState<0 | 1>(1);

	return (
		<div className="viz">
			<div className="nll__panels">
				<Panel
					title="loss vs prediction"
					y={y}
					yMax={5}
					p={p}
					fns={[
						{ color: TEAL, f: (pp) => nll(pp, y) },
						{ color: AMBER, f: (pp) => mse(pp, y) },
					]}
					ann="NLL → ∞"
				/>
				<Panel
					title="| gradient w.r.t. logit z |"
					y={y}
					yMax={1}
					p={p}
					fns={[
						{ color: TEAL, f: (pp) => nllGrad(pp, y) },
						{ color: AMBER, f: (pp) => mseGrad(pp, y) },
					]}
					ann="MSE → 0"
				/>
			</div>

			<div className="r3f-controls">
				<label className="viz__slider" style={{ flex: '1 1 16rem' }}>
					<span className="viz__slider-label">
						prediction p = σ(z) <b>{p.toFixed(2)}</b>
					</span>
					<input type="range" min={EPS} max={1 - EPS} step={0.005} value={p} onChange={(e) => setP(parseFloat(e.target.value))} />
				</label>
				<div className="nll__toggle">
					<span className="viz__slider-label">true label y</span>
					<div className="cg__buttons">
						<button type="button" className={`viz__btn${y === 1 ? ' viz__btn--on' : ''}`} onClick={() => setY(1)}>y = 1</button>
						<button type="button" className={`viz__btn${y === 0 ? ' viz__btn--on' : ''}`} onClick={() => setY(0)}>y = 0</button>
					</div>
				</div>
			</div>

			<div className="nll__readout">
				<span className="nll__key" style={{ color: TEAL }}>● NLL</span>
				<span>loss <b>{nll(p, y).toFixed(2)}</b></span>
				<span>|grad| <b>{nllGrad(p, y).toFixed(3)}</b></span>
				<span className="nll__key" style={{ color: AMBER }}>● MSE</span>
				<span>loss <b>{mse(p, y).toFixed(2)}</b></span>
				<span>|grad| <b>{mseGrad(p, y).toFixed(3)}</b></span>
			</div>
			<p className="cg__caption">
				Push p toward the wrong end (e.g. p≈0 with y=1): NLL's gradient stays strong while MSE's collapses toward zero — the confidently-wrong unit stops learning.
			</p>
		</div>
	);
}
