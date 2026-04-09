# Default to the Vite dev origin so credentialed fetches work; override with
# CORS_ALLOW_ORIGIN=... ./backend/dev.sh if you need a different value.
export CORS_ALLOW_ORIGIN="${CORS_ALLOW_ORIGIN:-http://localhost:5173}"
PORT="${PORT:-8080}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips '*' --reload
