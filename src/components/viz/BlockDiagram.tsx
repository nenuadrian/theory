import { useState } from 'react';

/**
 * The transformer block as a residual stream.
 *
 * A vertical "highway" carries each token's vector from top to bottom. Two
 * sublayers — multi-head self-attention and a position-wise MLP — each read a
 * normalized copy of the stream, compute an update, and add it back. The toggle
 * swaps where LayerNorm sits: inside the branch *before* each sublayer (Pre-LN,
 * what modern transformers use) versus on the stream *after* the add (Post-LN,
 * the original 2017 design).
 */

const STREAM_X = 70;
const RET_X = 330;

function Sublayer({
	yc,
	label,
	sub,
	preLN,
}: {
	yc: number;
	label: string;
	sub: string;
	preLN: boolean;
}) {
	const branchY = yc - 48;
	const boxY = branchY - 17; // boxes are 34 tall, centred on branchY
	const subX = preLN ? 150 : 116;
	const subW = 150;
	const branchStart = preLN ? 112 : subX;

	return (
		<g>
			{/* branch out of the stream, into the box chain */}
			<path
				className="cg__edge"
				d={`M ${STREAM_X} ${branchY} L ${branchStart} ${branchY}`}
				fill="none"
				markerEnd="url(#bd-arr)"
			/>
			{/* return: box chain → down → into the ⊕ from the right */}
			<path
				className="cg__edge"
				d={`M ${subX + subW} ${branchY} L ${RET_X} ${branchY} L ${RET_X} ${yc} L ${STREAM_X + 10} ${yc}`}
				fill="none"
				markerEnd="url(#bd-arr)"
			/>

			{/* Pre-LN: LayerNorm sits in the branch, before the sublayer */}
			{preLN && (
				<>
					<g transform={`translate(112 ${boxY})`}>
						<rect className="cg__node" width="30" height="34" />
						<text className="cg__val" x="15" y="21" textAnchor="middle">
							LN
						</text>
					</g>
					<path className="cg__edge" d={`M 142 ${branchY} L ${subX} ${branchY}`} fill="none" />
				</>
			)}

			{/* the sublayer itself */}
			<g transform={`translate(${subX} ${boxY})`}>
				<rect className="cg__node cg__node--sub" width={subW} height="34" />
				<text className="cg__label" x={subW / 2} y="21" textAnchor="middle">
					{sub}
				</text>
			</g>

			{/* the residual add on the stream */}
			<circle className="cg__add" cx={STREAM_X} cy={yc} r="9" />
			<text className="cg__addsign" x={STREAM_X} y={yc + 4} textAnchor="middle">
				+
			</text>

			{/* Post-LN: LayerNorm sits on the stream, after the add */}
			{!preLN && (
				<g transform={`translate(${STREAM_X - 15} ${yc + 16})`}>
					<rect className="cg__node" width="30" height="24" />
					<text className="cg__val" x="15" y="16" textAnchor="middle">
						LN
					</text>
				</g>
			)}

			<text className="cg__sublabel" x={RET_X} y={branchY - 13} textAnchor="end">
				{label}
			</text>
		</g>
	);
}

export default function BlockDiagram() {
	const [preLN, setPreLN] = useState(true);

	return (
		<div className="viz">
			<div className="viz__row" style={{ alignItems: 'center' }}>
				<svg className="cg bd" viewBox="0 0 380 470" role="img" aria-label="A transformer block drawn as a residual stream">
					<defs>
						<marker id="bd-arr" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto">
							<path d="M0,0 L6,3 L0,6 Z" className="bd__arrhead" />
						</marker>
					</defs>

					{/* the residual highway */}
					<path className="cg__stream" d={`M ${STREAM_X} 28 L ${STREAM_X} 446`} fill="none" markerEnd="url(#bd-arr)" />

					<text className="cg__label" x={STREAM_X} y="20" textAnchor="middle">
						x
					</text>
					<text className="cg__sublabel" x={STREAM_X + 14} y="20" textAnchor="start">
						input + positional encoding
					</text>

					<Sublayer yc={150} label="mixes information across positions" sub="Multi-Head Self-Attention" preLN={preLN} />
					<Sublayer yc={330} label="processes each position on its own" sub="Feed-Forward (MLP)" preLN={preLN} />

					<text className="cg__sublabel" x={STREAM_X + 14} y="462" textAnchor="start">
						to the next block →
					</text>
				</svg>

				<div className="viz__controls">
					<div className="bd__toggle">
						<button
							type="button"
							className={`viz__btn${preLN ? ' viz__btn--on' : ''}`}
							onClick={() => setPreLN(true)}
						>
							Pre-LN
						</button>
						<button
							type="button"
							className={`viz__btn${!preLN ? ' viz__btn--on' : ''}`}
							onClick={() => setPreLN(false)}
						>
							Post-LN
						</button>
					</div>
					<p className="cg__caption">
						{preLN
							? 'Pre-LN: normalize before each sublayer, add the raw update back. The stream stays an unobstructed identity path — gradients flow cleanly, so very deep stacks train stably.'
							: 'Post-LN: add first, then normalize the sum on the stream. The original design; it needs careful warm-up because the residual path passes through every LayerNorm.'}
					</p>
					<p className="cg__caption">
						<b style={{ color: 'var(--sl-color-text-accent)' }}>x ← x + Sublayer(LN(x))</b>, twice
						— attention, then MLP. Stack N of these and you have the body of a GPT.
					</p>
				</div>
			</div>
		</div>
	);
}
