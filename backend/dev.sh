#!/usr/bin/env bash
# Load Gnos3/.env if present, so values like WEBUI_SECRET_KEY (used to sign
# JWTs that the data modules verify) reach uvicorn. We parse line-by-line
# instead of `source`-ing because some values (e.g. CORS_ALLOW_ORIGIN with
# semicolon-separated origins) trip bash's command parser when sourced raw.
# Resolve against the script's own location, not the caller's cwd — a relative
# path here silently skips the whole block when launched from elsewhere, which
# leaves the fallbacks below in force and is very hard to spot after the fact.
# Everything below stays POSIX-clean (no arrays, no ${!indirect}) so that
# `sh dev.sh` works as well as `./dev.sh`: under dash those bash-isms abort the
# expansion with "Bad substitution" and the .env load is skipped entirely.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|'#'*|' '*) continue ;; esac
        case "$line" in *=*) ;; *) continue ;; esac
        key="${line%%=*}"
        val="${line#*=}"
        # Ignore anything that isn't a plain shell identifier, so the eval
        # below can never be handed arbitrary .env content to execute.
        case "$key" in
            ''|[!A-Za-z_]*|*[!A-Za-z0-9_]*) continue ;;
        esac
        # Strip matching surrounding quotes
        val="${val%\"}"; val="${val#\"}"
        val="${val%\'}"; val="${val#\'}"
        # Don't clobber values already set in the shell environment. `eval` is
        # the portable stand-in for bash's ${!key+x} indirect expansion.
        if eval "[ -z \"\${$key+x}\" ]"; then
            export "$key=$val"
        fi
    done < "$ENV_FILE"
fi

# Default to the Vite dev origin so credentialed fetches work; override with
# CORS_ALLOW_ORIGIN=... ./backend/dev.sh if you need a different value.
export CORS_ALLOW_ORIGIN="${CORS_ALLOW_ORIGIN:-http://localhost:5173}"
# Echo it: socket.io reuses this list, so a too-narrow value shows up as chats
# that hang until refresh (the websocket is rejected, deltas never arrive)
# rather than as an obvious error.
echo "dev.sh: CORS_ALLOW_ORIGIN=$CORS_ALLOW_ORIGIN"
PORT="${PORT:-8080}"
# Optional: set HF_TOKEN in your shell or .env to lift HuggingFace rate limits
# when the backend downloads embedding/reranker models on first boot. Never
# hardcode a real token in this file (it's committed to git).
export HF_TOKEN="${HF_TOKEN:-}"
uvicorn open_webui.main:app --port $PORT --host 0.0.0.0 --forwarded-allow-ips "${FORWARDED_ALLOW_IPS:-*}" --reload
