import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * The sinusoidal positional-encoding matrix, drawn as a heatmap.
 *
 *   PE[pos, 2k]   = sin( pos / 10000^(2k/d) )
 *   PE[pos, 2k+1] = cos( pos / 10000^(2k/d) )
 *
 * Every column is a sinusoid; their wavelengths grow geometrically from ~2π (low
 * dimensions, the fast stripes on the left) to ~10000·2π (high dimensions, the
 * slow gradients on the right). That spread is what lets a fixed-size vector name
 * every position uniquely — like the digits of a multi-resolution clock.
 */

const DISP_W = 460;
const DISP_H = 300;

const peValue = (pos: number, i: number, d: number) => {
	const k = Math.floor(i / 2);
	const angle = pos / Math.pow(10000, (2 * k) / d);
	return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
};

export default function PositionalEncoding() {
	const [L, setL] = useState(40);
	const [d, setD] = useState(48);
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	// Heatmap: positive values teal, negative values rose, alpha = magnitude.
	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const img = ctx.createImageData(d, L);
		for (let p = 0; p < L; p++) {
			for (let i = 0; i < d; i++) {
				const v = peValue(p, i, d); // [-1, 1]
				const idx = (p * d + i) * 4;
				if (v >= 0) {
					img.data[idx] = 96;
					img.data[idx + 1] = 211;
					img.data[idx + 2] = 197;
				} else {
					img.data[idx] = 224;
					img.data[idx + 1] = 96;
					img.data[idx + 2] = 122;
				}
				img.data[idx + 3] = Math.round(Math.abs(v) * 255);
			}
		}
		const off = document.createElement('canvas');
		off.width = d;
		off.height = L;
		off.getContext('2d')!.putImageData(img, 0, 0);

		ctx.fillStyle = '#0a0a0b';
		ctx.fillRect(0, 0, DISP_W, DISP_H);
		ctx.imageSmoothingEnabled = false;
		ctx.drawImage(off, 0, 0, DISP_W, DISP_H);
	}, [L, d]);

	// Three columns of contrasting wavelength, plotted as curves across position.
	const waves = useMemo(() => {
		const dims = [0, Math.max(2, Math.round(d / 8)) * 2, Math.max(2, Math.round(d / 3)) * 2].map(
			(i) => Math.min(i, d - 1),
		);
		const W = 460;
		const H = 90;
		return dims.map((dim) => {
			const pts = Array.from({ length: L }, (_, p) => {
				const x = (p / Math.max(1, L - 1)) * W;
				const y = H / 2 - (peValue(p, dim, d) * (H / 2 - 4));
				return `${x.toFixed(1)},${y.toFixed(1)}`;
			}).join(' ');
			return { dim, pts };
		});
	}, [L, d]);

	const waveColors = ['#60d3c5', '#9a9aa4', '#e0607a'];

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: DISP_W }}>
					<canvas
						ref={canvasRef}
						width={DISP_W}
						height={DISP_H}
						className="viz__canvas"
						aria-label="Sinusoidal positional-encoding matrix"
					/>
					<div className="viz__axis viz__axis--x">dimension →</div>
					<div className="viz__axis viz__axis--y">position →</div>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							sequence length <b>{L}</b>
						</span>
						<input
							type="range"
							min={8}
							max={64}
							step={1}
							value={L}
							onChange={(e) => setL(parseInt(e.target.value))}
						/>
					</label>
					<label className="viz__slider">
						<span className="viz__slider-label">
							model dim d <b>{d}</b>
						</span>
						<input
							type="range"
							min={8}
							max={64}
							step={2}
							value={d}
							onChange={(e) => setD(parseInt(e.target.value))}
						/>
					</label>
					<button
						type="button"
						className="viz__btn"
						onClick={() => {
							setL(40);
							setD(48);
						}}
					>
						reset
					</button>

					<svg className="pe-waves" viewBox="0 0 460 90" role="img" aria-label="A few PE dimensions plotted across position">
						<line x1="0" y1="45" x2="460" y2="45" className="pe-waves__axis" />
						{waves.map((w, k) => (
							<polyline key={k} points={w.pts} fill="none" stroke={waveColors[k]} strokeWidth={1.6} />
						))}
					</svg>
					<p className="cg__caption">
						columns <b style={{ color: waveColors[0] }}>{waves[0]?.dim}</b>,{' '}
						<b style={{ color: waveColors[1] }}>{waves[1]?.dim}</b>,{' '}
						<b style={{ color: waveColors[2] }}>{waves[2]?.dim}</b> across position — low dims
						oscillate fast, high dims slow.
					</p>
				</div>
			</div>
		</div>
	);
}
