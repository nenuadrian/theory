import { useEffect, useMemo, useState, type ReactNode } from 'react';

/**
 * A live byte-pair-encoding tokenizer — the OpenAI tokenizer playground, in
 * miniature. Type any text and watch it split into the exact tokens GPT-4o /
 * GPT-4 / GPT-3 see, via `gpt-tokenizer` (a pure-JS port of OpenAI's tiktoken).
 *
 * tiktoken itself is a Rust extension and won't load in Pyodide, so the real
 * encodings run here as a lazily-loaded JS island instead of a Python cell.
 */

type EncName = 'o200k_base' | 'cl100k_base' | 'p50k_base';

const MODELS: { enc: EncName; label: string; models: string }[] = [
	{ enc: 'o200k_base', label: 'o200k_base', models: 'GPT-4o · GPT-4.1 · o-series' },
	{ enc: 'cl100k_base', label: 'cl100k_base', models: 'GPT-4 · GPT-3.5' },
	{ enc: 'p50k_base', label: 'p50k_base', models: 'GPT-3 · Codex' },
];

interface Enc {
	encode: (s: string) => number[];
	decode: (t: number[]) => string;
}

// Each encoding's rank table is large, so load it on demand and cache it.
const cache: Partial<Record<EncName, Promise<Enc>>> = {};
function loadEnc(name: EncName): Promise<Enc> {
	if (!cache[name]) {
		const mod =
			name === 'o200k_base'
				? import('gpt-tokenizer/encoding/o200k_base')
				: name === 'cl100k_base'
					? import('gpt-tokenizer/encoding/cl100k_base')
					: import('gpt-tokenizer/encoding/p50k_base');
		cache[name] = mod.then((m) => ({ encode: m.encode, decode: m.decode }));
	}
	return cache[name]!;
}

const SAMPLE = `The cat sat on the mat.
Tokenization isn't always intuitive: "internationalization" breaks into pieces, numbers like 2025 and 3.14159 fragment, and emoji 🤖 cost several tokens.`;

const MAX = 2000;

// Low-alpha tints so adjacent tokens are distinguishable in both light and dark
// themes while the page's own text colour stays readable.
const COLORS = [
	'rgba(96, 211, 197, 0.22)',
	'rgba(130, 170, 255, 0.20)',
	'rgba(232, 140, 170, 0.20)',
	'rgba(180, 210, 120, 0.22)',
	'rgba(224, 170, 120, 0.22)',
	'rgba(176, 150, 232, 0.20)',
];

function TokenText({ t }: { t: string }): ReactNode {
	if (t.length === 0) return <span className="tok-pg__ph">∅</span>;
	if (!t.includes('\n')) return t;
	const out: ReactNode[] = [];
	t.split('\n').forEach((p, i) => {
		if (i > 0) out.push(<span key={`n${i}`} className="tok-pg__nl">↵{'\n'}</span>);
		if (p) out.push(<span key={`p${i}`}>{p}</span>);
	});
	return out;
}

export default function TokenizerPlayground() {
	const [text, setText] = useState(SAMPLE);
	const [encName, setEncName] = useState<EncName>('o200k_base');
	const [showIds, setShowIds] = useState(false);
	const [enc, setEnc] = useState<Enc | null>(null);
	const [loading, setLoading] = useState(true);

	useEffect(() => {
		let alive = true;
		setLoading(true);
		loadEnc(encName).then((e) => {
			if (alive) {
				setEnc(() => e);
				setLoading(false);
			}
		});
		return () => {
			alive = false;
		};
	}, [encName]);

	const tokens = useMemo(() => {
		if (!enc) return [] as { id: number; text: string }[];
		return enc.encode(text).map((id) => ({ id, text: enc.decode([id]) }));
	}, [enc, text]);

	const chars = [...text].length;
	const ratio = tokens.length ? chars / tokens.length : 0;
	const model = MODELS.find((m) => m.enc === encName);

	return (
		<div className="viz tok-pg">
			<div className="tok-pg__bar">
				<label className="viz__select tok-pg__enc">
					<span className="viz__slider-label">encoding</span>
					<select value={encName} onChange={(e) => setEncName(e.target.value as EncName)}>
						{MODELS.map((m) => (
							<option key={m.enc} value={m.enc}>
								{m.label} — {m.models}
							</option>
						))}
					</select>
				</label>
				<label className="tok-pg__toggle">
					<input
						type="checkbox"
						checked={showIds}
						onChange={(e) => setShowIds(e.target.checked)}
					/>
					token IDs
				</label>
			</div>

			<textarea
				className="tok-pg__input"
				value={text}
				spellCheck={false}
				rows={4}
				onChange={(e) => setText(e.target.value.slice(0, MAX))}
				aria-label="Text to tokenize"
			/>

			<div className="tok-pg__stats">
				<span>
					<b>{tokens.length}</b> tokens
				</span>
				<span>
					<b>{chars}</b> characters
				</span>
				<span>
					<b>{ratio.toFixed(2)}</b> chars / token
				</span>
				{loading && <span className="tok-pg__loading">loading {encName}…</span>}
			</div>

			<div className={`tok-pg__out${showIds ? ' tok-pg__out--ids' : ''}`}>
				{showIds
					? tokens.map((t, i) => (
							<span key={i} className="tok-pg__id">
								{t.id}
							</span>
						))
					: tokens.map((t, i) => (
							<span
								key={i}
								className="tok-pg__tok"
								style={{ background: COLORS[i % COLORS.length] }}
								title={`token #${t.id}`}
							>
								<TokenText t={t.text} />
							</span>
						))}
			</div>

			<code className="viz__formula">
				{model?.models} — byte-pair encoding via gpt-tokenizer (the BPE behind OpenAI’s tiktoken)
			</code>
		</div>
	);
}
