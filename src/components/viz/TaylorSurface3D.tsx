import { useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

/**
 * The Taylor tower in 3-D. A surface z = f(x, y), and over a local patch the
 * successive Taylor approximations about a chosen point p:
 *
 *   order 1   tangent plane      f(p) + ∇f·Δ                     (the GRADIENT)
 *   order 2   tangent paraboloid + ½ Δᵀ H Δ                      (the HESSIAN)
 *   order 3   cubic correction   + (1/6) Σ ∂³f Δᵢ Δⱼ Δₖ          (THIRD order)
 *
 * Grow the patch radius and watch the plane peel away first, the paraboloid
 * next, the cubic last — the remainder of a k-th order model is O(rᵏ⁺¹), printed
 * live as the RMS error. The monkey saddle x³ − 3xy² is the extreme case: at the
 * origin ∇f = 0 AND H = 0, so the plane and paraboloid are both flat — only the
 * third-order term reconstructs the shape.
 */

interface Fn {
	name: string;
	R: number;
	p0: [number, number];
	f: (x: number, y: number) => number;
	grad: (x: number, y: number) => [number, number];
	hess: (x: number, y: number) => [number, number, number]; // a, b, d
	cubic: (x: number, y: number, dx: number, dy: number) => number; // (1/6) Σ ∂³f Δ³
}

const FUNCS: Record<string, Fn> = {
	wave: {
		name: 'sin x · cos y',
		R: 3,
		p0: [0.9, 0.6],
		f: (x, y) => Math.sin(x) * Math.cos(y),
		grad: (x, y) => [Math.cos(x) * Math.cos(y), -Math.sin(x) * Math.sin(y)],
		hess: (x, y) => [-Math.sin(x) * Math.cos(y), -Math.cos(x) * Math.sin(y), -Math.sin(x) * Math.cos(y)],
		cubic: (x, y, dx, dy) => {
			const fxxx = -Math.cos(x) * Math.cos(y);
			const fxxy = Math.sin(x) * Math.sin(y);
			const fxyy = -Math.cos(x) * Math.cos(y);
			const fyyy = Math.sin(x) * Math.sin(y);
			return (fxxx * dx ** 3 + 3 * fxxy * dx ** 2 * dy + 3 * fxyy * dx * dy ** 2 + fyyy * dy ** 3) / 6;
		},
	},
	monkey: {
		name: 'x³ − 3xy²  (monkey saddle)',
		R: 1.6,
		p0: [0, 0],
		// scaled so the surface height is comparable to the wave
		f: (x, y) => 0.18 * (x ** 3 - 3 * x * y ** 2),
		grad: (x, y) => [0.18 * (3 * x ** 2 - 3 * y ** 2), 0.18 * (-6 * x * y)],
		hess: (x, y) => [0.18 * 6 * x, 0.18 * -6 * y, 0.18 * -6 * x],
		cubic: (_x, _y, dx, dy) => 0.18 * (dx ** 3 - 3 * dx * dy ** 2),
	},
};

const ORDERS = [
	{ key: 1, label: '1 · plane (∇f)', color: '#60d3c5' },
	{ key: 2, label: '2 · paraboloid (H)', color: '#e0b15e' },
	{ key: 3, label: '3 · cubic (∇³f)', color: '#a78bfa' },
];

// the k-th order Taylor model at p, as a function of the offset (dx, dy)
function model(fn: Fn, p: [number, number], k: number) {
	const f0 = fn.f(p[0], p[1]);
	const [gx, gy] = fn.grad(p[0], p[1]);
	const [a, b, d] = fn.hess(p[0], p[1]);
	return (dx: number, dy: number) => {
		let z = f0;
		if (k >= 1) z += gx * dx + gy * dy;
		if (k >= 2) z += 0.5 * (a * dx * dx + 2 * b * dx * dy + d * dy * dy);
		if (k >= 3) z += fn.cubic(p[0], p[1], dx, dy);
		return z;
	};
}

function surfaceGeometry(fn: Fn) {
	const N = 72;
	const R = fn.R;
	const pos: number[] = [];
	const col: number[] = [];
	const idx: number[] = [];
	const zs: number[] = [];
	let lo = Infinity, hi = -Infinity;
	for (let i = 0; i < N; i++)
		for (let j = 0; j < N; j++) {
			const x = -R + (2 * R * i) / (N - 1);
			const y = -R + (2 * R * j) / (N - 1);
			const z = fn.f(x, y);
			zs.push(z);
			lo = Math.min(lo, z);
			hi = Math.max(hi, z);
			pos.push(x, z, y);
		}
	for (let k = 0; k < zs.length; k++) {
		const t = (zs[k] - lo) / (hi - lo || 1);
		const gg = 0.16 + t * 0.7;
		col.push(gg, gg, gg);
	}
	for (let i = 0; i < N - 1; i++)
		for (let j = 0; j < N - 1; j++) {
			const A = i * N + j, B = i * N + j + 1, C = (i + 1) * N + j, D = (i + 1) * N + j + 1;
			idx.push(A, B, D, A, D, C);
		}
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	geo.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
	geo.setIndex(idx);
	geo.computeVertexNormals();
	return geo;
}

function patchGeometry(fn: Fn, p: [number, number], r: number, k: number) {
	const N = 30;
	const m = model(fn, p, k);
	const pos: number[] = [];
	const idx: number[] = [];
	for (let i = 0; i < N; i++)
		for (let j = 0; j < N; j++) {
			const dx = -r + (2 * r * i) / (N - 1);
			const dy = -r + (2 * r * j) / (N - 1);
			pos.push(p[0] + dx, m(dx, dy), p[1] + dy);
		}
	for (let i = 0; i < N - 1; i++)
		for (let j = 0; j < N - 1; j++) {
			const A = i * N + j, B = i * N + j + 1, C = (i + 1) * N + j, D = (i + 1) * N + j + 1;
			idx.push(A, B, D, A, D, C);
		}
	const geo = new THREE.BufferGeometry();
	geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
	geo.setIndex(idx);
	geo.computeVertexNormals();
	return geo;
}

function Scene({ fn, p, r, shown }: { fn: Fn; p: [number, number]; r: number; shown: Set<number> }) {
	const surf = useMemo(() => surfaceGeometry(fn), [fn]);
	const patches = useMemo(
		() => ORDERS.filter((o) => shown.has(o.key)).map((o) => ({ ...o, geo: patchGeometry(fn, p, r, o.key) })),
		[fn, p, r, shown],
	);
	const pz = fn.f(p[0], p[1]);
	return (
		<>
			<ambientLight intensity={0.65} />
			<directionalLight position={[6, 10, 4]} intensity={1.05} />
			<directionalLight position={[-6, 6, -4]} intensity={0.4} />
			<mesh geometry={surf}>
				<meshStandardMaterial vertexColors roughness={0.92} metalness={0.04} side={THREE.DoubleSide} />
			</mesh>
			<mesh geometry={surf}>
				<meshBasicMaterial wireframe color="#ffffff" transparent opacity={0.05} />
			</mesh>
			{patches.map((pt) => (
				<mesh key={pt.key} geometry={pt.geo} renderOrder={pt.key}>
					<meshStandardMaterial
						color={pt.color}
						emissive={pt.color}
						emissiveIntensity={0.35}
						roughness={0.6}
						transparent
						opacity={0.5}
						depthWrite={false}
						side={THREE.DoubleSide}
					/>
				</mesh>
			))}
			<mesh position={[p[0], pz, p[1]]}>
				<sphereGeometry args={[fn.R * 0.03, 20, 20]} />
				<meshStandardMaterial color="#ffffff" emissive="#444" emissiveIntensity={0.6} />
			</mesh>
			<OrbitControls enablePan={false} minDistance={fn.R * 1.6} maxDistance={fn.R * 6} target={[0, 0, 0]} />
		</>
	);
}

export default function TaylorSurface3D() {
	const [fnKey, setFnKey] = useState<keyof typeof FUNCS>('wave');
	const fn = FUNCS[fnKey];
	const [p, setP] = useState<[number, number]>(fn.p0);
	const [r, setR] = useState(1.2);
	const [shown, setShown] = useState<Set<number>>(new Set([1, 2, 3]));

	const switchFn = (key: keyof typeof FUNCS) => {
		setFnKey(key);
		setP(FUNCS[key].p0);
		setR(key === 'monkey' ? 1.1 : 1.2);
	};
	const toggle = (k: number) =>
		setShown((s) => {
			const n = new Set(s);
			n.has(k) ? n.delete(k) : n.add(k);
			return n;
		});

	// RMS error of each order over the patch (live)
	const errs = useMemo(() => {
		const out: Record<number, number> = {};
		for (const o of ORDERS) {
			const m = model(fn, p, o.key);
			let e = 0, n = 0;
			for (let i = 0; i < 22; i++)
				for (let j = 0; j < 22; j++) {
					const dx = -r + (2 * r * i) / 21;
					const dy = -r + (2 * r * j) / 21;
					const diff = fn.f(p[0] + dx, p[1] + dy) - m(dx, dy);
					e += diff * diff;
					n++;
				}
			out[o.key] = Math.sqrt(e / n);
		}
		return out;
	}, [fn, p, r]);

	return (
		<div className="viz">
			<div className="r3f-stage">
				<Canvas key={fnKey} camera={{ position: [fn.R * 2.3, fn.R * 2.3, fn.R * 3], fov: 45 }} dpr={[1, 2]}>
					<color attach="background" args={['#0a0a0b']} />
					<Scene fn={fn} p={p} r={r} shown={shown} />
				</Canvas>
				<span className="r3f-hint">drag to rotate · scroll to zoom</span>
			</div>

			<div className="r3f-controls">
				<label className="viz__select" style={{ flex: '1 1 12rem' }}>
					<span className="viz__slider-label">function</span>
					<select value={fnKey} onChange={(e) => switchFn(e.target.value as keyof typeof FUNCS)}>
						{Object.entries(FUNCS).map(([k, v]) => (
							<option key={k} value={k}>{v.name}</option>
						))}
					</select>
				</label>
				<label className="viz__slider" style={{ flex: '1 1 11rem' }}>
					<span className="viz__slider-label">patch radius r <b>{r.toFixed(2)}</b></span>
					<input type="range" min={0.3} max={fn.R * 0.7} step={0.05} value={r} onChange={(e) => setR(parseFloat(e.target.value))} />
				</label>
				<label className="viz__slider" style={{ flex: '1 1 9rem' }}>
					<span className="viz__slider-label">point x <b>{p[0].toFixed(2)}</b></span>
					<input type="range" min={-fn.R * 0.8} max={fn.R * 0.8} step={0.05} value={p[0]} onChange={(e) => setP([parseFloat(e.target.value), p[1]])} />
				</label>
				<label className="viz__slider" style={{ flex: '1 1 9rem' }}>
					<span className="viz__slider-label">point y <b>{p[1].toFixed(2)}</b></span>
					<input type="range" min={-fn.R * 0.8} max={fn.R * 0.8} step={0.05} value={p[1]} onChange={(e) => setP([p[0], parseFloat(e.target.value)])} />
				</label>
			</div>

			<div className="cf__orders">
				{ORDERS.map((o) => (
					<button
						key={o.key}
						type="button"
						className={`viz__btn${shown.has(o.key) ? ' viz__btn--on' : ''}`}
						style={shown.has(o.key) ? { borderColor: o.color, color: o.color } : undefined}
						onClick={() => toggle(o.key)}
					>
						{o.label}
						<span className="cf__err">RMS {errs[o.key] < 0.001 ? errs[o.key].toExponential(1) : errs[o.key].toFixed(3)}</span>
					</button>
				))}
			</div>

			<code className="viz__formula">
				f(p+Δ) = f(p) + ∇f·Δ + ½ ΔᵀHΔ + (1/6) Σ ∂³f ΔᵢΔⱼΔₖ + O(‖Δ‖⁴)
			</code>
		</div>
	);
}
