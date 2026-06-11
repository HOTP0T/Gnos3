#!/usr/bin/env bash
# Load Gnos3/.env if present, so values like WEBUI_SECRET_KEY (used to sign
# JWTs that the data modules verify) reach uvicorn. We parse line-by-line
# instead of `source`-ing because some values (e.g. CORS_ALLOW_ORIGIN with
# semicolon-separated origins) trip bash's command parser when sourced raw.
ENV_FILE="$(dirname "${BASH_SOURCE[0]}")/../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*|' '*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"
        val="${line#*=}"
        # Strip matching surrounding quotes
        val="${val%\"}"; val="${val#\"}"
        val="${val%\'}"; val="${val#\'}"
        # Don't clobber values already set in the shell environment
        if [ -z "${!key+x}" ]; then
            export "$key=$val"
        fi
    done < "$ENV_FILE"
fi

# Default to the Vite dev origin so credentialed fetches work; override with
# CORS_ALLOW_ORIGIN=... ./backend/dev.sh if you need a different value.
export CORS_ALLOW_ORIGIN="${CORS_ALLOW_ORIGIN:-http://localhost:5173}"
PORT="${PORT:-8080}"
# Optional: set HF_TOKEN in your shell or .env to lift HuggingFace rate limits
# when the backend downloads embedding/reranker models on first boot. Never
# hardcode a real token in this file (it's committed to git).
export HF_TOKEN="${HF_TOKEN:-}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}" --reload
