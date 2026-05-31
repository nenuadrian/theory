import { useMemo, useState } from 'react';

/**
 * Why more than one attention head?
 *
 * A single head can only express one attention pattern per position. Real
 * transformers run several in parallel, and trained heads reliably specialize —
 * some track the previous token, some anchor on the first token ("attention
 * sink"), some copy the current position, some look ahead. Here are four such
 * canonical patterns over the same sentence, each a proper softmax distribution.
 * Hover a head to enlarge it.
 */

const TOKENS = ['The', 'cat', 'sat', 'on', 'the', 'mat'];

type Head = { name: string; note: string; rule: (i: number, j: number, n: number) => number };

const HEADS: Head[] = [
	{ name: 'previous token', note: 'attends one step back', rule: (i, j) => (j === i - 1 ? 4 : 0) },
	{ name: 'first token', note: 'an “attention sink”', rule: (i, j) => (j === 0 ? 3.5 : 0) },
	{ name: 'self', note: 'keeps its own value', rule: (i, j) => (j === i ? 4 : 0) },
	{
		name: 'next token',
		note: 'peeks one step ahead',
		rule: (i, j, n) => (j === Math.min(i + 1, n - 1) ? 4 : 0),
	},
];

function softmaxRow(logits: number[]): number[] {
	const m = Math.max(...logits);
	const e = logits.map((v) => Math.exp(v - m));
	const s = e.reduce((a, b) => a + b, 0);
	return e.map((v) => v / s);
}

function buildMatrix(head: Head, n: number): number[][] {
	return Array.from({ length: n }, (_, i) =>
		softmaxRow(Array.from({ length: n }, (_, j) => head.rule(i, j, n))),
	);
}

export default function MultiHeadAttention() {
	const [active, setActive] = useState<number | null>(null);
	const n = TOKENS.length;
	const matrices = useMemo(() => HEADS.map((h) => buildMatrix(h, n)), [n]);

	return (
		<div className="viz">
			<div className="heads">
				{HEADS.map((h, hi) => (
					<button
						type="button"
						key={hi}
						className={`head${active === hi ? ' head--active' : ''}`}
						onMouseEnter={() => setActive(hi)}
						onMouseLeave={() => setActive(null)}
						onFocus={() => setActive(hi)}
						onBlur={() => setActive(null)}
					>
						<div className="head__title">
							head {hi + 1} · <span>{h.name}</span>
						</div>
						<div
							className="head__grid"
							style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}
							aria-hidden="true"
						>
							{matrices[hi].map((r, i) =>
								r.map((a, j) => (
									<span
										key={`${i}-${j}`}
										className="head__cell"
										style={{ background: `rgba(96,211,197,${a.toFixed(3)})` }}
										title={`${TOKENS[i]} → ${TOKENS[j]}: ${a.toFixed(2)}`}
									/>
								)),
							)}
						</div>
						<div className="head__note">{h.note}</div>
					</button>
				))}
			</div>
			<code className="viz__formula">
				MHA(X) = [ head₁ ; head₂ ; … ; head&#8202;ₕ ] · W&#8202;ᴼ
			</code>
		</div>
	);
}
