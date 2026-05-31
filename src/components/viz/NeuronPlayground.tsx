import { useEffect, useMemo, useRef, useState } from 'react';

/**
 * Interactive single neuron. Drag the weights and bias and watch the neuron's
 * output across the whole 2-D input plane — the "decision surface" — rotate and
 * shift in real time. The bright contour is where the output crosses its middle
 * value: the decision boundary.
 */

type Act = 'sigmoid' | 'tanh' | 'relu';

const ACT: Record<Act, { fn: (z: number) => number; norm: (a: number) => number; mid: number; label: string }> = {
	sigmoid: { fn: (z) => 1 / (1 + Math.exp(-z)), norm: (a) => a, mid: 0.5, label: 'σ' },
	tanh: { fn: (z) => Math.tanh(z), norm: (a) => (a + 1) / 2, mid: 0.5, label: 'tanh' },
	relu: { fn: (z) => Math.max(0, z), norm: (a) => Math.min(1, a / 4), mid: 0.5 / 4, label: 'relu' },
};

const SIZE = 240; // display px
const RES = 120; // compute grid
const RANGE = 4; // input domain [-RANGE, RANGE]

function Slider({
	label,
	value,
	onChange,
}: {
	label: string;
	value: number;
	onChange: (v: number) => void;
}) {
	return (
		<label className="viz__slider">
			<span className="viz__slider-label">
				{label} <b>{value.toFixed(2)}</b>
			</span>
			<input
				type="range"
				min={-3}
				max={3}
				step={0.01}
				value={value}
				onChange={(e) => onChange(parseFloat(e.target.value))}
			/>
		</label>
	);
}

export default function NeuronPlayground() {
	const [w1, setW1] = useState(1.2);
	const [w2, setW2] = useState(-0.8);
	const [b, setB] = useState(0.3);
	const [act, setAct] = useState<Act>('sigmoid');
	const canvasRef = useRef<HTMLCanvasElement | null>(null);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;

		const { fn, norm, mid } = ACT[act];
		const img = ctx.createImageData(RES, RES);
		for (let j = 0; j < RES; j++) {
			const x2 = RANGE - (j / (RES - 1)) * 2 * RANGE; // top = +RANGE
			for (let i = 0; i < RES; i++) {
				const x1 = -RANGE + (i / (RES - 1)) * 2 * RANGE;
				const a = norm(fn(w1 * x1 + w2 * x2 + b));
				const t = Math.max(0, Math.min(1, a));
				// grayscale fill
				let g = Math.round(18 + t * (232 - 18));
				let r = g,
					bl = g,
					gg = g;
				// crisp decision-boundary contour in a restrained accent
				if (Math.abs(t - mid) < 0.012) {
					r = 96;
					gg = 211;
					bl = 197;
				}
				const idx = (j * RES + i) * 4;
				img.data[idx] = r;
				img.data[idx + 1] = gg;
				img.data[idx + 2] = bl;
				img.data[idx + 3] = 255;
			}
		}
		// draw at native grid res, then scale up with smoothing off for crispness
		const off = document.createElement('canvas');
		off.width = RES;
		off.height = RES;
		off.getContext('2d')!.putImageData(img, 0, 0);
		ctx.imageSmoothingEnabled = true;
		ctx.clearRect(0, 0, SIZE, SIZE);
		ctx.drawImage(off, 0, 0, SIZE, SIZE);
	}, [w1, w2, b, act]);

	const formula = useMemo(() => {
		const term = (c: number, v: string) =>
			`${c < 0 ? '−' : '+'} ${Math.abs(c).toFixed(2)} ${v}`;
		return `a = ${ACT[act].label}( ${w1.toFixed(2)} x₁ ${term(w2, 'x₂')} ${
			b < 0 ? '−' : '+'
		} ${Math.abs(b).toFixed(2)} )`;
	}, [w1, w2, b, act]);

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="viz__canvas-wrap" style={{ width: SIZE }}>
					<canvas
						ref={canvasRef}
						width={SIZE}
						height={SIZE}
						className="viz__canvas"
						aria-label="Neuron output over the 2-D input plane"
					/>
					<div className="viz__axis viz__axis--x">x₁ →</div>
					<div className="viz__axis viz__axis--y">x₂ →</div>
				</div>

				<div className="viz__controls">
					<label className="viz__select">
						<span className="viz__slider-label">activation</span>
						<select value={act} onChange={(e) => setAct(e.target.value as Act)}>
							<option value="sigmoid">sigmoid</option>
							<option value="tanh">tanh</option>
							<option value="relu">ReLU</option>
						</select>
					</label>
					<Slider label="w₁" value={w1} onChange={setW1} />
					<Slider label="w₂" value={w2} onChange={setW2} />
					<Slider label="b " value={b} onChange={setB} />
					<button
						type="button"
						className="viz__btn"
						onClick={() => {
							setW1(1.2);
							setW2(-0.8);
							setB(0.3);
							setAct('sigmoid');
						}}
					>
						reset
					</button>
				</div>
			</div>
			<code className="viz__formula">{formula}</code>
		</div>
	);
}
