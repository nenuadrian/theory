import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Rotary position embedding (RoPE) acting on a single 2-D coordinate pair.
 *
 * A query and a key sit at fixed "content" directions φ_q, φ_k. RoPE rotates the
 * query by m·θ and the key by n·θ, where m, n are the two tokens' positions and θ
 * is the pair's angular frequency. The angle BETWEEN the rotated vectors is
 *
 *     Δ = (φ_k − φ_q) + (n − m)·θ,
 *
 * so the score q·k = cos Δ (unit vectors) depends on position only through the
 * relative offset n − m. Sliding both positions together spins the two arrows in
 * lockstep and leaves Δ — hence the score — untouched; sliding the offset opens
 * the angle and moves the score. Relative position is thus realized exactly,
 * inside the dot product, with no parameters.
 *
 * Plain 2-D canvas on the hardcoded dark .viz__canvas (#111): all text/axes are
 * drawn in fixed light colors, never getComputedStyle().
 */

const DISP = 320;
const CX = DISP / 2;
const CY = DISP / 2;
const R = 116;

// Fixed illustrative content directions (radians), before any rotation.
const PHI_Q = (104 * Math.PI) / 180;
const PHI_K = (40 * Math.PI) / 180;

const TEAL = '#60d3c5';
const ROSE = '#e0607a';
const INK = '#c4c4cb';

// math angle (counter-clockwise, +x = 0) -> screen point (canvas y points down)
const pt = (ang: number, r: number) => [CX + r * Math.cos(ang), CY - r * Math.sin(ang)] as const;

function arrow(ctx: CanvasRenderingContext2D, ang: number, color: string, label: string) {
	const [x, y] = pt(ang, R);
	ctx.strokeStyle = color;
	ctx.fillStyle = color;
	ctx.lineWidth = 2.5;
	ctx.beginPath();
	ctx.moveTo(CX, CY);
	ctx.lineTo(x, y);
	ctx.stroke();
	// arrowhead (screen y is flipped, so the perpendicular spread adds on y)
	const spread = 0.42;
	const hl = 13;
	ctx.beginPath();
	ctx.moveTo(x, y);
	ctx.lineTo(x - hl * Math.cos(ang - spread), y + hl * Math.sin(ang - spread));
	ctx.lineTo(x - hl * Math.cos(ang + spread), y + hl * Math.sin(ang + spread));
	ctx.closePath();
	ctx.fill();
	const [lx, ly] = pt(ang, R + 17);
	ctx.font = '600 14px ui-monospace, monospace';
	ctx.textAlign = 'center';
	ctx.textBaseline = 'middle';
	ctx.fillText(label, lx, ly);
}

export default function RotaryEncoding() {
	const [pos, setPos] = useState(3); // base position m
	const [offset, setOffset] = useState(4); // relative offset n − m
	const [theta, setTheta] = useState(0.5); // angular frequency, rad per position
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	const m = pos;
	const n = pos + offset;
	const angQ = PHI_Q + m * theta;
	const angK = PHI_K + n * theta;
	const delta = PHI_K - PHI_Q + offset * theta; // = angK − angQ, independent of m
	const score = Math.cos(delta);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		ctx.fillStyle = '#111';
		ctx.fillRect(0, 0, DISP, DISP);

		// faint axes + unit circle
		ctx.strokeStyle = '#33343a';
		ctx.lineWidth = 1;
		ctx.beginPath();
		ctx.moveTo(CX - R - 20, CY);
		ctx.lineTo(CX + R + 20, CY);
		ctx.moveTo(CX, CY - R - 20);
		ctx.lineTo(CX, CY + R + 20);
		ctx.stroke();
		ctx.strokeStyle = '#2a2b30';
		ctx.beginPath();
		ctx.arc(CX, CY, R, 0, Math.PI * 2);
		ctx.stroke();

		// minor arc spanning the angle between the two arrows
		const sQ = -angQ; // screen angles (clockwise positive)
		let d = -angK - sQ;
		while (d <= -Math.PI) d += 2 * Math.PI;
		while (d > Math.PI) d -= 2 * Math.PI;
		ctx.strokeStyle = INK;
		ctx.lineWidth = 1.6;
		ctx.beginPath();
		ctx.arc(CX, CY, 38, sQ, sQ + d, d < 0);
		ctx.stroke();

		arrow(ctx, angQ, TEAL, 'q');
		arrow(ctx, angK, ROSE, 'k');
	}, [angQ, angK, pos, offset, theta]);

	// Score as a function of offset, at the current frequency — the curve the
	// dot product traces out. The score depends on offset alone, never on m.
	const curve = useMemo(() => {
		const W = 300;
		const H = 84;
		const OMAX = 12;
		const xs = (o: number) => ((o + OMAX) / (2 * OMAX)) * W;
		const ys = (s: number) => H / 2 - s * (H / 2 - 6);
		const pts: string[] = [];
		for (let o = -OMAX; o <= OMAX; o += 0.2) {
			pts.push(`${xs(o).toFixed(1)},${ys(Math.cos(PHI_K - PHI_Q + o * theta)).toFixed(1)}`);
		}
		return { line: pts.join(' '), W, H, mx: xs(offset), my: ys(score), zx: xs(0) };
	}, [theta, offset, score]);

	const barFill = Math.round(((score + 1) / 2) * 100);

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: DISP }}>
					<canvas
						ref={canvasRef}
						width={DISP}
						height={DISP}
						className="viz__canvas"
						aria-label="Query and key vectors rotated by their positions"
					/>
					<div className="viz__axis viz__axis--x">x →</div>
					<div className="viz__axis viz__axis--y">y →</div>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							position <b>m = {m}</b>
						</span>
						<input
							type="range"
							min={0}
							max={24}
							step={1}
							value={pos}
							onChange={(e) => setPos(parseInt(e.target.value))}
						/>
					</label>
					<label className="viz__slider">
						<span className="viz__slider-label">
							offset <b>n − m = {offset}</b>
						</span>
						<input
							type="range"
							min={-10}
							max={10}
							step={1}
							value={offset}
							onChange={(e) => setOffset(parseInt(e.target.value))}
						/>
					</label>
					<label className="viz__slider">
						<span className="viz__slider-label">
							frequency <b>θ = {theta.toFixed(2)}</b>
						</span>
						<input
							type="range"
							min={0.1}
							max={1.0}
							step={0.05}
							value={theta}
							onChange={(e) => setTheta(parseFloat(e.target.value))}
						/>
					</label>

					<div style={{ fontSize: '0.78rem', color: 'var(--sl-color-gray-3)', lineHeight: 1.7 }}>
						<div>
							positions{' '}
							<b style={{ color: TEAL }}>m = {m}</b>,{' '}
							<b style={{ color: ROSE }}>n = {n}</b>
						</div>
						<div>
							score q·k = cos Δ ={' '}
							<b style={{ color: 'var(--sl-color-text-accent)' }}>{score.toFixed(3)}</b>
						</div>
					</div>
					<div
						style={{
							position: 'relative',
							height: 6,
							background: 'var(--sl-color-bg)',
							border: '1px solid var(--sl-color-hairline)',
						}}
						aria-hidden="true"
					>
						<div
							style={{
								position: 'absolute',
								left: 0,
								top: 0,
								bottom: 0,
								width: `${barFill}%`,
								background: score >= 0 ? TEAL : ROSE,
							}}
						/>
						<div
							style={{
								position: 'absolute',
								left: '50%',
								top: -2,
								bottom: -2,
								width: 1,
								background: 'var(--sl-color-gray-4)',
							}}
						/>
					</div>

					<button
						type="button"
						className="viz__btn"
						onClick={() => {
							setPos(3);
							setOffset(4);
							setTheta(0.5);
						}}
					>
						reset
					</button>

					<svg
						className="pe-waves"
						viewBox={`0 0 ${curve.W} ${curve.H}`}
						role="img"
						aria-label="Attention score as a function of relative offset"
					>
						<line x1="0" y1={curve.H / 2} x2={curve.W} y2={curve.H / 2} className="pe-waves__axis" />
						<line x1={curve.zx} y1="0" x2={curve.zx} y2={curve.H} className="pe-waves__axis" />
						<polyline points={curve.line} fill="none" stroke={TEAL} strokeWidth={1.6} />
						<circle cx={curve.mx} cy={curve.my} r={3.2} fill={ROSE} />
					</svg>
					<p className="cg__caption">
						score vs. offset n − m. Slide <b>position</b> and both arrows co-rotate — the score is
						unmoved. Slide <b>offset</b> and the angle opens, tracing this curve.
					</p>
				</div>
			</div>
		</div>
	);
}
