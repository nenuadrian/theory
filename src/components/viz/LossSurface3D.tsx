import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';
import * as THREE from 'three';

/**
 * A 3-D loss landscape with a ball that rolls downhill under gradient descent.
 * Rotate with the mouse; change the learning rate and watch the ball converge,
 * crawl, or bounce out of a basin entirely.
 *
 *   L(x, y) = 0.10 (x² + y²) + 0.55 · sin(x) · cos(y)
 *
 * The quadratic bowl pulls toward the origin; the ripple carves local minima, so
 * where the ball ends up depends on where it starts and how big its steps are.
 */

const R = 4;
const START = { x: 2.7, y: 3.1 };

const loss = (x: number, y: number) =>
	0.1 * (x * x + y * y) + 0.55 * Math.sin(x) * Math.cos(y);
const gradX = (x: number, y: number) => 0.2 * x + 0.55 * Math.cos(x) * Math.cos(y);
const gradY = (x: number, y: number) => 0.2 * y - 0.55 * Math.sin(x) * Math.sin(y);

type Pt = [number, number, number];

function Scene({
	lr,
	startNonce,
	resetNonce,
	onStop,
}: {
	lr: number;
	startNonce: number;
	resetNonce: number;
	onStop: (steps: number, lossVal: number) => void;
}) {
	const ballRef = useRef<THREE.Mesh>(null);
	const posRef = useRef({ ...START });
	const lrRef = useRef(lr);
	const accRef = useRef(0);
	const stepRef = useRef(0);
	const [running, setRunning] = useState(false);
	const [trail, setTrail] = useState<Pt[]>([[START.x, loss(START.x, START.y) + 0.06, START.y]]);

	useEffect(() => {
		lrRef.current = lr;
	}, [lr]);

	// Start (re)runs gradient descent from START.
	useEffect(() => {
		if (startNonce === 0) return;
		posRef.current = { ...START };
		stepRef.current = 0;
		accRef.current = 0;
		setTrail([[START.x, loss(START.x, START.y) + 0.06, START.y]]);
		setRunning(true);
	}, [startNonce]);

	// Reset stops and parks the ball back at the start.
	useEffect(() => {
		setRunning(false);
		posRef.current = { ...START };
		setTrail([[START.x, loss(START.x, START.y) + 0.06, START.y]]);
	}, [resetNonce]);

	// Build the surface geometry once: grayscale by height + computed normals.
	const geometry = useMemo(() => {
		const N = 70;
		const positions: number[] = [];
		const colors: number[] = [];
		const indices: number[] = [];
		const zs: number[] = [];
		let lo = Infinity;
		let hi = -Infinity;
		for (let i = 0; i < N; i++) {
			for (let j = 0; j < N; j++) {
				const x = -R + (2 * R * i) / (N - 1);
				const y = -R + (2 * R * j) / (N - 1);
				const z = loss(x, y);
				zs.push(z);
				lo = Math.min(lo, z);
				hi = Math.max(hi, z);
				positions.push(x, z, y);
			}
		}
		for (let k = 0; k < zs.length; k++) {
			const t = (zs[k] - lo) / (hi - lo);
			const g = 0.16 + t * 0.74;
			colors.push(g, g, g);
		}
		for (let i = 0; i < N - 1; i++) {
			for (let j = 0; j < N - 1; j++) {
				const a = i * N + j;
				const b = i * N + j + 1;
				const c = (i + 1) * N + j;
				const d = (i + 1) * N + j + 1;
				indices.push(a, b, d, a, d, c);
			}
		}
		const geo = new THREE.BufferGeometry();
		geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
		geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
		geo.setIndex(indices);
		geo.computeVertexNormals();
		return geo;
	}, []);

	useFrame((_, dt) => {
		const { x, y } = posRef.current;
		if (ballRef.current) ballRef.current.position.set(x, loss(x, y) + 0.13, y);
		if (!running) return;
		accRef.current += dt;
		if (accRef.current < 0.05) return;
		accRef.current = 0;
		const gx = gradX(x, y);
		const gy = gradY(x, y);
		const nx = x - lrRef.current * gx;
		const ny = y - lrRef.current * gy;
		posRef.current = { x: nx, y: ny };
		setTrail((t) => [...t, [nx, loss(nx, ny) + 0.06, ny]]);
		stepRef.current += 1;
		const gnorm = Math.hypot(gx, gy);
		if (stepRef.current > 240 || gnorm < 1.5e-3 || Math.abs(nx) > 8 || Math.abs(ny) > 8) {
			setRunning(false);
			onStop(stepRef.current, loss(nx, ny));
		}
	});

	return (
		<>
			<ambientLight intensity={0.6} />
			<directionalLight position={[6, 10, 4]} intensity={1.1} />
			<directionalLight position={[-6, 6, -4]} intensity={0.4} />
			<mesh geometry={geometry}>
				<meshStandardMaterial vertexColors roughness={0.9} metalness={0.05} side={THREE.DoubleSide} />
			</mesh>
			<mesh geometry={geometry}>
				<meshBasicMaterial wireframe color="#ffffff" transparent opacity={0.05} />
			</mesh>
			{trail.length > 1 && <Line points={trail} color="#60d3c5" lineWidth={2} />}
			<mesh ref={ballRef}>
				<sphereGeometry args={[0.14, 24, 24]} />
				<meshStandardMaterial color="#60d3c5" emissive="#16403b" emissiveIntensity={0.8} />
			</mesh>
			<OrbitControls enablePan={false} minDistance={6} maxDistance={20} target={[0, 0, 0]} />
		</>
	);
}

export default function LossSurface3D() {
	const [lr, setLr] = useState(0.15);
	const [startNonce, setStartNonce] = useState(0);
	const [resetNonce, setResetNonce] = useState(0);
	const [status, setStatus] = useState('Press “drop” to release the ball.');

	return (
		<div className="viz">
			<div className="r3f-stage">
				<Canvas camera={{ position: [7, 7, 9], fov: 45 }} dpr={[1, 2]}>
					<color attach="background" args={['#0a0a0b']} />
					<Scene
						lr={lr}
						startNonce={startNonce}
						resetNonce={resetNonce}
						onStop={(steps, l) =>
							setStatus(`Settled after ${steps} steps · loss ≈ ${l.toFixed(3)}`)
						}
					/>
				</Canvas>
				<span className="r3f-hint">drag to rotate · scroll to zoom</span>
			</div>

			<div className="r3f-controls">
				<label className="viz__slider" style={{ flex: '1 1 14rem' }}>
					<span className="viz__slider-label">
						learning rate η <b>{lr.toFixed(2)}</b>
					</span>
					<input
						type="range"
						min={0.02}
						max={1.4}
						step={0.01}
						value={lr}
						onChange={(e) => setLr(parseFloat(e.target.value))}
					/>
				</label>
				<button
					type="button"
					className="viz__btn"
					onClick={() => {
						setStatus('Rolling downhill…');
						setStartNonce((n) => n + 1);
					}}
				>
					▸ drop
				</button>
				<button type="button" className="viz__btn" onClick={() => setResetNonce((n) => n + 1)}>
					reset
				</button>
			</div>
			<p className="cg__caption">{status}</p>
		</div>
	);
}
