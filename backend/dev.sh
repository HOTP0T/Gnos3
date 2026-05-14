# Default to the Vite dev origin so credentialed fetches work; override with
# CORS_ALLOW_ORIGIN=... ./backend/dev.sh if you need a different value.
export CORS_ALLOW_ORIGIN="${CORS_ALLOW_ORIGIN:-http://localhost:5173}"
PORT="${PORT:-8080}"
# Optional: set HF_TOKEN in your shell or .env to lift HuggingFace rate limits
# when the backend downloads embedding/reranker models on first boot. Never
# hardcode a real token in this file (it's committed to git).
export HF_TOKEN="${HF_TOKEN:-}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}" --reload
