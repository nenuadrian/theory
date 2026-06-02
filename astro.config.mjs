// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';

// The site is served from this sub-path (GitHub Pages project page).
const BASE = '/theory/';

/**
 * Astro does not prepend `base` to hardcoded root-absolute links authored in
 * Markdown/MDX content — only to its own generated navigation. Our pages link
 * to each other with root-absolute paths (e.g. `/neural-networks/the-neuron/`),
 * which resolve against the domain root and 404 under a sub-path deploy. This
 * remark plugin prepends BASE to any in-content link/image URL that starts with
 * `/` (skipping protocol-relative `//…` and URLs already under BASE). JSX links
 * such as the Playground anchor use import.meta.env.BASE_URL and are parsed as
 * JSX nodes, so they are not matched here.
 */
function remarkBasePrefix() {
	const prefix = BASE.replace(/\/$/, '');
	/** @param {string} url */
	const withBase = (url) =>
		typeof url === 'string' &&
		url.startsWith('/') &&
		!url.startsWith('//') &&
		url !== prefix &&
		!url.startsWith(`${prefix}/`)
			? prefix + url
			: url;
	/** @param {any} node */
	const visit = (node) => {
		if (node && (node.type === 'link' || node.type === 'definition' || node.type === 'image') && node.url) {
			node.url = withBase(node.url);
		}
		if (node && node.children) node.children.forEach(visit);
	};
	return (/** @type {any} */ tree) => visit(tree);
}

// https://astro.build/config
export default defineConfig({
	site: 'https://nenuadrian.github.io',
	base: BASE,
	redirects: {
		// The EM page was merged into RL as Inference; keep old links working.
		'/reinforcement-learning/expectation-maximization': '/reinforcement-learning/rl-as-inference',
	},
	markdown: {
		remarkPlugins: [remarkGfm, remarkMath, remarkBasePrefix],
		rehypePlugins: [rehypeKatex],
	},
	vite: {
		// react-three-fiber's reconciler must share a single React instance with
		// the React DOM islands, or drei components throw "invalid hook call".
		resolve: {
			dedupe: ['react', 'react-dom', '@react-three/fiber', 'three'],
		},
		optimizeDeps: {
			include: [
				'react',
				'react-dom',
				'react/jsx-runtime',
				'@react-three/fiber',
				'@react-three/drei',
				'three',
			],
		},
	},
	integrations: [
		starlight({
			title: 'learn',
			description:
				'An interactive knowledge base — theory, visuals, and live in-browser Python.',
			customCss: [
				'@fontsource-variable/geist',
				'@fontsource-variable/geist-mono',
				'katex/dist/katex.min.css',
				'./src/styles/theme.css',
			],
			components: {
				SiteTitle: './src/components/overrides/SiteTitle.astro',
			},
			expressiveCode: {
				themes: ['github-dark', 'github-light'],
				styleOverrides: {
					borderRadius: '0px',
					borderColor: 'var(--sl-color-hairline)',
					frames: {
						shadowColor: 'transparent',
					},
				},
			},
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com' },
			],
			sidebar: [
				{
					label: 'Neural Networks',
					items: [
						{ label: 'Overview', slug: 'neural-networks' },
						{ label: 'The Neuron', slug: 'neural-networks/the-neuron' },
						{ label: 'The Forward Pass', slug: 'neural-networks/forward-pass' },
						{ label: 'Backpropagation', slug: 'neural-networks/backpropagation' },
						{ label: 'Gradient Descent', slug: 'neural-networks/gradient-descent' },
						{ label: 'Curvature & the Hessian', slug: 'neural-networks/curvature' },
						{ label: 'Optimizers', slug: 'neural-networks/optimizers' },
						{ label: 'Loss: NLL vs MSE', slug: 'neural-networks/loss-functions' },
					],
				},
				{
					label: 'Transformers',
					items: [
						{ label: 'Overview', slug: 'transformers' },
						{ label: 'Tokens & Embeddings', slug: 'transformers/tokens-and-embeddings' },
						{ label: 'Attention', slug: 'transformers/attention' },
						{ label: 'Multi-Head Attention', slug: 'transformers/multi-head-attention' },
						{ label: 'Positional Encoding', slug: 'transformers/positional-encoding' },
						{ label: 'Rotary Position Encoding', slug: 'transformers/rotary-position-encoding' },
						{ label: 'The Transformer Block', slug: 'transformers/transformer-block' },
					],
				},
				{
					label: 'Divergences',
					items: [
						{ label: 'Overview', slug: 'divergences' },
						{ label: 'Bregman Divergences', slug: 'divergences/bregman' },
						{ label: 'KL: Forward vs Reverse', slug: 'divergences/kl-forward-reverse' },
						{ label: 'Fisher Information', slug: 'divergences/fisher-information' },
					],
				},
				{
					label: 'Reinforcement Learning',
					items: [
						{ label: 'Overview', slug: 'reinforcement-learning' },
						{ label: 'The Setup', slug: 'reinforcement-learning/the-setup' },
						{
							label: 'Policy Gradients',
							items: [
								{ label: 'The Policy Gradient', slug: 'reinforcement-learning/policy-gradients' },
								{ label: 'Trust Regions', slug: 'reinforcement-learning/trust-regions' },
								{ label: 'GRPO', slug: 'reinforcement-learning/grpo' },
							],
						},
						{
							label: 'Expectation Maximization',
							items: [
								{ label: 'RL as Inference', slug: 'reinforcement-learning/rl-as-inference' },
								{ label: 'MPO', slug: 'reinforcement-learning/mpo' },
								{ label: 'V-MPO', slug: 'reinforcement-learning/vmpo' },
								{ label: 'AWR & AWAC', slug: 'reinforcement-learning/awr' },
							],
						},
						{ label: 'PG vs EM', slug: 'reinforcement-learning/pg-vs-em' },
					],
				},
				{
					label: 'World Models',
					items: [
						{ label: 'Overview', slug: 'world-models' },
						{ label: 'The Idea', slug: 'world-models/the-idea' },
						{ label: 'Perception: the VAE', slug: 'world-models/perception' },
						{ label: 'Imagination: Dynamics', slug: 'world-models/dynamics' },
						{ label: 'The RSSM & Planning', slug: 'world-models/rssm-and-planning' },
						{ label: 'Learning in Imagination', slug: 'world-models/dreamer' },
						{ label: 'Models Without Pixels', slug: 'world-models/value-equivalence' },
					],
				},
				{
					label: 'Graph Neural Networks',
					items: [
						{ label: 'Overview', slug: 'graph-neural-networks' },
						{ label: 'Graphs & Message Passing', slug: 'graph-neural-networks/graphs-and-message-passing' },
						{ label: 'Graph Convolutions', slug: 'graph-neural-networks/graph-convolutional-networks' },
						{ label: 'Attention on Graphs', slug: 'graph-neural-networks/graph-attention' },
						{ label: 'Depth & Pitfalls', slug: 'graph-neural-networks/depth-and-pitfalls' },
						{ label: 'Graph Transformers', slug: 'graph-neural-networks/graph-transformers' },
					],
				},
				{
					label: 'Lab',
					items: [{ label: 'Playground', slug: 'playground' }],
				},
				{
					label: 'Reference',
					items: [{ label: 'Bibliography', slug: 'references' }],
				},
			],
		}),
		react(),
	],
});
