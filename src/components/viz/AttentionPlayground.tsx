import { Fragment, useMemo, useState } from 'react';

/**
 * Self-attention, made tangible.
 *
 * A short sentence; each token carries a small (illustrative, not learned)
 * feature vector. We score every query token against every key token with a
 * scaled dot product, softmax each row into a probability distribution, and show
 * the result as an attention matrix you can read straight off: row i tells you
 * what token i "looks at". Click a row to spotlight that query; drag the
 * temperature to sharpen or flatten the distributions.
 */

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];

// 3-D "content" vectors, hand-chosen so related tokens point in similar
// directions: the two determiners align, the two nouns align, verb and
// preposition align. Purely illustrative — in a real model the query and key
// vectors come from learned projections X·Wq and X·Wk.
const FEAT: number[][] = [
	[1.0, 0.2, 0.0], // The  — determiner
	[0.2, 1.0, 0.3], // cat  — noun
	[0.0, 0.3, 1.0], // sat  — verb
	[0.1, 0.0, 0.9], // on   — preposition
	[0.95, 0.25, 0.0], // the — determiner
	[0.25, 0.95, 0.35], // mat — noun
];
const DK = 3;

const dot = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);

function softmax(xs: number[], temp: number): number[] {
	const z = xs.map((x) => x / temp);
	const m = Math.max(...z);
	const e = z.map((v) => Math.exp(v - m));
	const s = e.reduce((a, b) => a + b, 0);
	return e.map((v) => v / s);
}

export default function AttentionPlayground() {
	const [temp, setTemp] = useState(1);
	const [query, setQuery] = useState(1); // spotlight "cat" to start

	// Full attention matrix: A[i][j] = softmax_j( q_i · k_j / √dₖ / τ ).
	const A = useMemo(() => {
		const scale = Math.sqrt(DK);
		return FEAT.map((qi) =>
			softmax(
				FEAT.map((kj) => dot(qi, kj) / scale),
				temp,
			),
		);
	}, [temp]);

	const row = A[query];

	return (
		<div className="viz">
			<div className="viz__row">
				<div className="attn">
					<div
						className="attn__grid"
						style={{ gridTemplateColumns: `3.4rem repeat(${TOKENS.length}, 1fr)` }}
					>
						<div className="attn__corner">{'q\\k'}</div>
						{TOKENS.map((t, j) => (
							<div key={j} className="attn__collabel">
								{t}
							</div>
						))}
						{A.map((r, i) => (
							<Fragment key={i}>
								<button
									type="button"
									className={`attn__rowlabel${i === query ? ' attn__rowlabel--active' : ''}`}
									onClick={() => setQuery(i)}
								>
									{TOKENS[i]}
								</button>
								{r.map((a, j) => (
									<button
										type="button"
										key={j}
										className={`attn__cell${i === query ? ' attn__cell--inrow' : ''}`}
										style={{
											background: `rgba(96,211,197,${a.toFixed(3)})`,
											color: a > 0.45 ? '#06201d' : 'var(--sl-color-gray-3)',
										}}
										onClick={() => setQuery(i)}
										title={`${TOKENS[i]} → ${TOKENS[j]}: ${a.toFixed(3)}`}
									>
										{a >= 0.995 ? '1.0' : a.toFixed(2).slice(1)}
									</button>
								))}
							</Fragment>
						))}
					</div>
					<div className="attn__legend">
						each <b>row</b> is a probability distribution — it sums to 1
					</div>
				</div>

				<div className="viz__controls">
					<label className="viz__slider">
						<span className="viz__slider-label">
							temperature τ <b>{temp.toFixed(2)}</b>
						</span>
						<input
							type="range"
							min={0.2}
							max={3}
							step={0.05}
							value={temp}
							onChange={(e) => setTemp(parseFloat(e.target.value))}
						/>
					</label>
					<button
						type="button"
						className="viz__btn"
						onClick={() => {
							setTemp(1);
							setQuery(1);
						}}
					>
						reset
					</button>
					<p className="cg__caption">
						“<b style={{ color: 'var(--sl-color-text-accent)' }}>{TOKENS[query]}</b>”
						attends to:
					</p>
					<div className="attn__chips">
						{TOKENS.map((t, j) => (
							<span
								key={j}
								className="attn__chip"
								style={{
									background: `rgba(96,211,197,${row[j].toFixed(3)})`,
									color: row[j] > 0.45 ? '#06201d' : 'var(--sl-color-text)',
								}}
							>
								{t}
								<i>{Math.round(row[j] * 100)}%</i>
							</span>
						))}
					</div>
				</div>
			</div>
			<code className="viz__formula">A = softmax( Q Kᵀ / √dₖ ) · V</code>
		</div>
	);
}
