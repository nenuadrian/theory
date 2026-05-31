import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * The latent space of a variational autoencoder, made navigable.
 *
 * A decoder maps a 2-D latent z to an image o = decode(z). Here the decoder is a
 * fixed, smooth generator (an oriented Gabor patch: z1 rotates the grating, z2
 * sets its frequency) so the *geometry* the VAE imposes is visible without
 * training: nearby latents decode to similar images (a smooth manifold), and the
 * prior N(0, I) is the region from which plausible samples are drawn. Drag the
 * sliders to traverse the manifold; sample from the prior to generate.
 */

const W = 560;
const HGT = 320;
const SCALE = 2;
const RES = 24; // decoded image resolution
const THUMB = 7; // manifold lattice is THUMB x THUMB
const LIM = 2.6; // latent plane shows z in [-LIM, LIM]

// decode(z) -> RES x RES image in [0,1]. An oriented Gabor patch.
function decode(z1: number, z2: number): Float32Array {
	const img = new Float32Array(RES * RES);
	const theta = z1 * (Math.PI / 3); // orientation from z1
	const freq = Math.max(0.4, 2.0 + 0.7 * z2); // spatial frequency from z2
	const sig = 0.55;
	const ct = Math.cos(theta),
		st = Math.sin(theta);
	for (let j = 0; j < RES; j++) {
		for (let i = 0; i < RES; i++) {
			const u = (i / (RES - 1)) * 2 - 1;
			const v = (j / (RES - 1)) * 2 - 1;
			const env = Math.exp(-(u * u + v * v) / (2 * sig * sig));
			const g = env * Math.cos(2 * Math.PI * freq * (u * ct + v * st));
			img[j * RES + i] = (g + 1) / 2; // -> [0,1]
		}
	}
	return img;
}

function gauss(): number {
	// Box–Muller; fresh randomness per call is fine in component code.
	let u = 0,
		v = 0;
	while (u === 0) u = Math.random();
	while (v === 0) v = Math.random();
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export default function VAELatent() {
	const [z1, setZ1] = useState(0.4);
	const [z2, setZ2] = useState(-0.3);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	// The manifold: a fixed lattice of decoded thumbnails. Computed once.
	const thumbs = useMemo(() => {
		const out: { z1: number; z2: number; img: Float32Array }[] = [];
		for (let r = 0; r < THUMB; r++) {
			for (let c = 0; c < THUMB; c++) {
				const a = -LIM + (2 * LIM * c) / (THUMB - 1);
				const b = LIM - (2 * LIM * r) / (THUMB - 1);
				out.push({ z1: a, z2: b, img: decode(a, b) });
			}
		}
		return out;
	}, []);

	const current = useMemo(() => decode(z1, z2), [z1, z2]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const fg = '#c4c4cb'; // canvas bg is a fixed dark (#111) in both themes
		ctx.setTransform(SCALE, 0, 0, SCALE, 0, 0);
		ctx.clearRect(0, 0, W, HGT);

		// ---- left: the latent plane ----------------------------------------
		const L = 12,
			T = 16,
			S = HGT - 40; // square plane of side S
		const mapX = (z: number) => L + ((z + LIM) / (2 * LIM)) * S;
		const mapY = (z: number) => T + ((LIM - z) / (2 * LIM)) * S;

		// prior N(0, I): faint concentric density rings
		for (let k = 3; k >= 1; k--) {
			ctx.beginPath();
			ctx.arc(mapX(0), mapY(0), (k / 3) * (S / 2) * 0.92, 0, 2 * Math.PI);
			ctx.fillStyle = `rgba(96,211,197,${0.05 + 0.05 * (3 - k)})`;
			ctx.fill();
		}

		// the manifold of thumbnails
		const ts = (S / THUMB) * 0.62; // thumbnail draw size
		for (const t of thumbs) {
			const cx = mapX(t.z1) - ts / 2;
			const cy = mapY(t.z2) - ts / 2;
			const px = ts / RES;
			for (let j = 0; j < RES; j++)
				for (let i = 0; i < RES; i++) {
					const val = t.img[j * RES + i];
					const g = Math.round(val * 235);
					ctx.fillStyle = `rgb(${g},${g},${g})`;
					ctx.fillRect(cx + i * px, cy + j * px, px + 0.5, px + 0.5);
				}
		}

		// current z marker
		ctx.beginPath();
		ctx.arc(mapX(z1), mapY(z2), 6, 0, 2 * Math.PI);
		ctx.strokeStyle = '#fb923c';
		ctx.lineWidth = 2.5;
		ctx.stroke();
		ctx.fillStyle = 'rgba(251,146,60,0.35)';
		ctx.fill();

		// axes labels
		ctx.fillStyle = fg;
		ctx.font = '11px ui-monospace, monospace';
		ctx.textAlign = 'center';
		ctx.fillText('latent space  z ∈ ℝ²   (prior N(0, I))', L + S / 2, T + S + 18);
		ctx.save();
		ctx.translate(L - 4, T + S / 2);
		ctx.rotate(-Math.PI / 2);
		ctx.fillText('z₂  (frequency)', 0, 0);
		ctx.restore();
		ctx.fillText('z₁  (orientation)', L + S / 2, T - 6);

		// ---- right: decoded image at current z -----------------------------
		const bx = L + S + 34,
			by = T + 16,
			bs = Math.min(W - bx - 16, S - 40);
		const px = bs / RES;
		for (let j = 0; j < RES; j++)
			for (let i = 0; i < RES; i++) {
				const val = current[j * RES + i];
				const g = Math.round(val * 235);
				ctx.fillStyle = `rgb(${g},${g},${g})`;
				ctx.fillRect(bx + i * px, by + j * px, px + 0.5, px + 0.5);
			}
		ctx.strokeStyle = 'rgba(251,146,60,0.7)';
		ctx.lineWidth = 1.5;
		ctx.strokeRect(bx, by, bs, bs);
		ctx.fillStyle = fg;
		ctx.fillText('decode(z)', bx + bs / 2, by - 8);
		ctx.font = '10px ui-monospace, monospace';
		ctx.fillStyle = 'rgba(196,196,203,0.8)';
		ctx.fillText(`z = (${z1.toFixed(2)}, ${z2.toFixed(2)})`, bx + bs / 2, by + bs + 16);
	}, [z1, z2, thumbs, current]);

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
						aria-label="A 2-D VAE latent space with a decoded-image manifold"
					/>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							z₁ <b>{z1.toFixed(2)}</b>
						</span>
						<input
							type="range"
							min={-LIM}
							max={LIM}
							step={0.02}
							value={z1}
							onChange={(e) => setZ1(parseFloat(e.target.value))}
						/>
					</label>
					<label className="viz__slider">
						<span className="viz__slider-label">
							z₂ <b>{z2.toFixed(2)}</b>
						</span>
						<input
							type="range"
							min={-LIM}
							max={LIM}
							step={0.02}
							value={z2}
							onChange={(e) => setZ2(parseFloat(e.target.value))}
						/>
					</label>

					<div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								setZ1(Math.max(-LIM, Math.min(LIM, gauss())));
								setZ2(Math.max(-LIM, Math.min(LIM, gauss())));
							}}
						>
							▸ sample z ~ N(0, I)
						</button>
						<button
							type="button"
							className="viz__btn"
							onClick={() => {
								setZ1(0);
								setZ2(0);
							}}
						>
							z = 0
						</button>
					</div>

					<p className="viz__slider-label" style={{ lineHeight: 1.5 }}>
						<span style={{ opacity: 0.85 }}>
							Nearby z decode to similar images — a smooth manifold. The prior
							N(0, I) is where samples are drawn to generate.
						</span>
					</p>
				</div>
			</div>
			<code className="viz__formula">
				o = decode(z),  z ~ N(0, I);   ELBO = E_q[ log p(o|z) ] − KL( q(z|o) ‖ p(z) )
			</code>
		</div>
	);
}
