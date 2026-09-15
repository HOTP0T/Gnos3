import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
	plugins: [
		sveltekit(),
		viteStaticCopy({
			targets: [
				{
					src: 'node_modules/onnxruntime-web/dist/*.jsep.*',

					dest: 'wasm'
				}
			]
		})
	],
	define: {
		APP_VERSION: JSON.stringify(process.env.npm_package_version),
		APP_BUILD_HASH: JSON.stringify(process.env.APP_BUILD_HASH || 'dev-build')
	},
	server: {
		watch: {
			// The Python venv and the backend's runtime data (vector DB, uploads,
			// SQLite) live inside this tree. Watching them exhausts inotify
			// (ENOSPC crash right after "ready") and contributes nothing to HMR.
			ignored: ['**/venv/**', '**/.venv/**', '**/backend/data/**']
		},
		proxy: {
			'/static': 'http://localhost:8080',
			'/api': 'http://localhost:8080',
			'/ollama': 'http://localhost:8080',
			'/openai': 'http://localhost:8080',
			'/ws/socket.io': {
				target: 'http://localhost:8080',
				ws: true
			}
		}
	},
	build: {
		sourcemap: true
	},
	worker: {
		format: 'es'
	},
	esbuild: {
		pure: process.env.ENV === 'dev' ? [] : ['console.log', 'console.debug', 'console.error']
	}
});
