import { useEffect, useRef, useState } from 'react';

/**
 * First and second order, side by side, on the surface f(x, y) = sin x · cos y.
 *
 * Drag the point. The teal arrow is the GRADIENT ∇f — steepest ascent. The
 * cross and ellipse are the HESSIAN H = ∇²f: its eigenvectors are the principal
 * curvature directions, the eigenvalues their curvatures (teal = positive /
 * bowl, pink = negative / dome). Where the two signs disagree the point is a
 * saddle and the ellipse opens into a hyperbola (drawn as its asymptotes).
 *
 * Toggle the NEWTON step −H⁻¹∇f against the plain gradient step −∇f: Newton
 * rescales and rotates the step by the local curvature, stepping straight to the
 * critical point of the quadratic model (the dashed gradient step does not).
 */

const SIZE = 340;
const DOM = 3; // x, y ∈ [−3, 3]
const TEAL = '#60d3c5';
const PINK = '#f472b6';
const VIOLET = '#a78bfa';
const GRAY = 'rgba(170,170,180,0.65)';

// f and its exact derivatives
const f = (x: number, y: number) => Math.sin(x) * Math.cos(y);
const grad = (x: number, y: number): [number, number] => [
	Math.cos(x) * Math.cos(y),
	-Math.sin(x) * Math.sin(y),
];
// Hessian [[a, b], [b, d]]
const hess = (x: number, y: number) => {
	const a = -Math.sin(x) * Math.cos(y);
	const b = -Math.cos(x) * Math.sin(y);
	const d = -Math.sin(x) * Math.cos(y);
	return { a, b, d };
};

// symmetric 2×2 eigensystem, λ1 ≥ λ2
function eig(a: number, b: number, d: number) {
	const tr = a + d;
	const disc = Math.sqrt(Math.max(0, ((a - d) / 2) ** 2 + b * b));
	const l1 = tr / 2 + disc;
	const l2 = tr / 2 - disc;
	const vec = (l: number): [number, number] => {
		if (Math.abs(b) > 1e-9) {
			const v: [number, number] = [b, l - a];
			const n = Math.hypot(v[0], v[1]);
			return [v[0] / n, v[1] / n];
		}
		// diagonal H: axis-aligned eigenvectors
		return Math.abs(a - l) < Math.abs(d - l) ? [1, 0] : [0, 1];
	};
	const v1 = vec(l1);
	return { l1, l2, v1, v2: [-v1[1], v1[0]] as [number, number] };
}

export default function CurvatureField() {
	const canvasRef = useRef<HTMLCanvasElement | null>(null);
	const svgRef = useRef<SVGSVGElement | null>(null);
	const [p, setP] = useState<[number, number]>([-1.25, 0.5]);
	const [showHess, setShowHess] = useState(true);
	const [showNewton, setShowNewton] = useState(true);
	const drag = useRef(false);

	const xToPx = (x: number) => ((x + DOM) / (2 * DOM)) * SIZE;
	const yToPx = (y: number) => SIZE - ((y + DOM) / (2 * DOM)) * SIZE;
	const pxPer = SIZE / (2 * DOM);
	// unit data-vector → pixel-space direction (y flips)
	const dirPx = (vx: number, vy: number): [number, number] => [vx, -vy];

	// banded contour map of f, drawn once
	useEffect(() => {
		const c = canvasRef.current;
		if (!c) return;
		const ctx = c.getContext('2d');
		if (!ctx) return;
		const RES = 150;
		const img = ctx.createImageData(RES, RES);
		for (let j = 0; j < RES; j++)
			for (let i = 0; i < RES; i++) {
				const x = -DOM + (2 * DOM * i) / (RES - 1);
				const y = DOM - (2 * DOM * j) / (RES - 1);
				const t = (f(x, y) + 1) / 2; // [0,1]
				const band = Math.floor(t * 11) / 11;
				const g = Math.round(22 + band * 170);
				const k = (j * RES + i) * 4;
				img.data[k] = g;
				img.data[k + 1] = g;
				img.data[k + 2] = g + 6;
				img.data[k + 3] = 255;
			}
		const off = document.createElement('canvas');
		off.width = RES;
		off.height = RES;
		off.getContext('2d')!.putImageData(img, 0, 0);
		ctx.imageSmoothingEnabled = true;
		ctx.clearRect(0, 0, SIZE, SIZE);
		ctx.drawImage(off, 0, 0, SIZE, SIZE);
	}, []);

	const onMove = (e: React.PointerEvent) => {
		if (!drag.current || !svgRef.current) return;
		const r = svgRef.current.getBoundingClientRect();
		const px = ((e.clientX - r.left) / r.width) * SIZE;
		const py = ((e.clientY - r.top) / r.height) * SIZE;
		const x = Math.max(-DOM, Math.min(DOM, (px / SIZE) * 2 * DOM - DOM));
		const y = Math.max(-DOM, Math.min(DOM, DOM - (py / SIZE) * 2 * DOM));
		setP([x, y]);
	};

	const [x, y] = p;
	const g = grad(x, y);
	const gn = Math.hypot(g[0], g[1]);
	const { a, b, d } = hess(x, y);
	const { l1, l2, v1, v2 } = eig(a, b, d);
	const det = a * d - b * b;
	const definite = l1 * l2 > 1e-4;
	const kind =
		l1 > 0.02 && l2 > 0.02 ? 'convex · bowl'
		: l1 < -0.02 && l2 < -0.02 ? 'concave · dome'
		: l1 * l2 < -1e-4 ? 'saddle'
		: 'near-flat';
	const kappa = Math.abs(l2) > 1e-4 ? Math.abs(l1) / Math.abs(l2) : Infinity;

	// Newton step −H⁻¹∇f  (displacement to the quadratic model's critical point)
	const newton: [number, number] =
		Math.abs(det) > 1e-4
			? [-(d * g[0] - b * g[1]) / det, -(-b * g[0] + a * g[1]) / det]
			: [0, 0];

	// pixel anchors
	const P: [number, number] = [xToPx(x), yToPx(y)];
	const GA = 70; // px per unit gradient
	const gTip: [number, number] = [P[0] + dirPx(g[0], g[1])[0] * GA, P[1] + dirPx(g[0], g[1])[1] * GA];

	// eigen-cross half-lengths ∝ √|λ|, clamped
	const half = (l: number) => Math.max(9, Math.min(46, 30 * Math.sqrt(Math.abs(l))));
	const crossSeg = (v: [number, number], l: number) => {
		const dpx = dirPx(v[0], v[1]);
		const L = half(l);
		return {
			x1: P[0] - dpx[0] * L, y1: P[1] - dpx[1] * L,
			x2: P[0] + dpx[0] * L, y2: P[1] + dpx[1] * L,
			color: l >= 0 ? TEAL : PINK,
		};
	};
	const seg1 = crossSeg(v1, l1);
	const seg2 = crossSeg(v2, l2);

	// osculating ellipse (definite case): semi-axes ∝ 1/√|λ|
	const ax1 = Math.max(10, Math.min(70, 24 / Math.sqrt(Math.abs(l1))));
	const ax2 = Math.max(10, Math.min(70, 24 / Math.sqrt(Math.abs(l2))));
	const ellRot = (Math.atan2(-v1[1], v1[0]) * 180) / Math.PI;

	// Newton + gradient-descent step tips (Newton at true scale, capped; descent for contrast)
	const cap = (v: [number, number], maxpx: number): [number, number] => {
		const lpx = Math.hypot(v[0] * pxPer, v[1] * pxPer);
		const s = lpx > maxpx ? maxpx / lpx : 1;
		return [P[0] + dirPx(v[0], v[1])[0] * pxPer * s, P[1] + dirPx(v[0], v[1])[1] * pxPer * s];
	};
	const newtonTip = cap(newton, 150);
	const descTip = cap([-g[0], -g[1]], 150);

	const fmt = (v: number, n = 2) => (v < 0 ? '−' : '') + Math.abs(v).toFixed(n);
	const arrow = (x1: number, y1: number, x2: number, y2: number, color: string, dash = false, w = 2) => {
		const ang = Math.atan2(y2 - y1, x2 - x1);
		const h = 7;
		const a1 = ang + Math.PI - 0.42, a2 = ang + Math.PI + 0.42;
		return (
			<g>
				<line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={w}
					strokeDasharray={dash ? '4 3' : undefined} />
				<polygon
					points={`${x2},${y2} ${x2 + h * Math.cos(a1)},${y2 + h * Math.sin(a1)} ${x2 + h * Math.cos(a2)},${y2 + h * Math.sin(a2)}`}
					fill={color} />
			</g>
		);
	};

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: SIZE }}>
					<canvas ref={canvasRef} width={SIZE} height={SIZE} className="viz__canvas" />
					<svg
						ref={svgRef}
						className="opt-overlay"
						viewBox={`0 0 ${SIZE} ${SIZE}`}
						width={SIZE}
						height={SIZE}
						style={{ touchAction: 'none', cursor: drag.current ? 'grabbing' : 'grab' }}
						onPointerDown={(e) => { drag.current = true; (e.target as Element).setPointerCapture?.(e.pointerId); onMove(e); }}
						onPointerMove={onMove}
						onPointerUp={(e) => { drag.current = false; (e.target as Element).releasePointerCapture?.(e.pointerId); }}
					>
						{/* osculating quadric */}
						{showHess && definite && (
							<ellipse
								cx={P[0]} cy={P[1]} rx={ax1} ry={ax2}
								transform={`rotate(${ellRot} ${P[0]} ${P[1]})`}
								fill={l1 > 0 ? 'rgba(96,211,197,0.10)' : 'rgba(244,114,182,0.10)'}
								stroke={l1 > 0 ? TEAL : PINK} strokeWidth={1.4} opacity={0.9}
							/>
						)}
						{/* Hessian eigen-cross */}
						{showHess && (
							<>
								<line x1={seg1.x1} y1={seg1.y1} x2={seg1.x2} y2={seg1.y2} stroke={seg1.color} strokeWidth={2.5} />
								<line x1={seg2.x1} y1={seg2.y1} x2={seg2.x2} y2={seg2.y2} stroke={seg2.color} strokeWidth={2.5} />
							</>
						)}
						{/* Newton vs gradient-descent step */}
						{showNewton && arrow(P[0], P[1], descTip[0], descTip[1], GRAY, true, 2)}
						{showNewton && arrow(P[0], P[1], newtonTip[0], newtonTip[1], VIOLET, false, 2.5)}
						{/* gradient (steepest ascent) */}
						{arrow(P[0], P[1], gTip[0], gTip[1], TEAL, false, 2.5)}
						{/* the point */}
						<circle cx={P[0]} cy={P[1]} r={5} fill="#fff" stroke="#0a0a0b" strokeWidth={1.5} />
					</svg>
					<span className="viz__axis viz__axis--x">x →</span>
					<span className="viz__axis viz__axis--y">y →</span>
				</div>

				<div className="viz__controls">
					<p className="cg__caption" style={{ marginTop: 0 }}>
						Drag the point. <span style={{ color: TEAL }}>▬</span> gradient ∇f ·{' '}
						<span style={{ color: TEAL }}>+</span>/<span style={{ color: PINK }}>+</span> Hessian eigen-curvatures.
					</p>

					<div className="cf__readout">
						<div className="cf__line"><span>f(x,y)</span><b>{fmt(f(x, y), 3)}</b></div>
						<div className="cf__line"><span>∇f</span><b>[ {fmt(g[0])}, {fmt(g[1])} ]</b></div>
						<div className="cf__line"><span>‖∇f‖</span><b>{fmt(gn, 3)}</b></div>
						<div className="cf__matwrap">
							<span className="cf__matlabel">H = ∇²f</span>
							<div className="cf__mat">
								<span>{fmt(a)}</span><span>{fmt(b)}</span>
								<span>{fmt(b)}</span><span>{fmt(d)}</span>
							</div>
						</div>
						<div className="cf__line"><span>eigenvalues</span><b><span style={{ color: l1 >= 0 ? TEAL : PINK }}>{fmt(l1)}</span>, <span style={{ color: l2 >= 0 ? TEAL : PINK }}>{fmt(l2)}</span></b></div>
						<div className="cf__line"><span>condition κ</span><b>{Number.isFinite(kappa) ? fmt(kappa, 2) : '∞'}</b></div>
						<div className="cf__line"><span>type</span><b className="cf__kind">{kind}</b></div>
						{showNewton && (
							<div className="cf__line"><span style={{ color: VIOLET }}>Newton −H⁻¹∇f</span><b>[ {fmt(newton[0])}, {fmt(newton[1])} ]</b></div>
						)}
					</div>

					<div className="cg__buttons">
						<button type="button" className={`viz__btn${showHess ? ' viz__btn--on' : ''}`} onClick={() => setShowHess((s) => !s)}>Hessian</button>
						<button type="button" className={`viz__btn${showNewton ? ' viz__btn--on' : ''}`} onClick={() => setShowNewton((s) => !s)}>Newton step</button>
						<button type="button" className="viz__btn" onClick={() => setP([-1.25, 0.5])}>reset</button>
					</div>
				</div>
			</div>
			<code className="viz__formula">
				f(p+Δ) ≈ f(p) + ∇f·Δ + ½ Δᵀ H Δ &nbsp;·&nbsp; H = ∇²f &nbsp;·&nbsp; Newton: Δ* = −H⁻¹∇f
			</code>
		</div>
	);
}
