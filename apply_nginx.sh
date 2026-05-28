#!/bin/bash
# Render nginx.conf.template with values from .env and apply to /etc/nginx/nginx.conf.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── prerequisites ──────────────────────────────────────────────────────────────
if ! command -v envsubst &>/dev/null; then
    echo "ERROR: envsubst not found. Install with: apt-get install -y gettext-base" >&2
    exit 1
fi

# ── API key ────────────────────────────────────────────────────────────────────
set -a; source "$SCRIPT_DIR/.env"; set +a

if [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY not set in .env" >&2
    exit 1
fi

TEMPLATE="$SCRIPT_DIR/nginx.conf.template"
NGINX_CONF="/etc/nginx/nginx.conf"
BACKUP="/etc/nginx/nginx.conf.bak.$(date +%Y%m%d_%H%M%S)"

# ── render template to temp file ───────────────────────────────────────────────
VLLM_TMP=$(mktemp)
API_KEY="$API_KEY" envsubst '${API_KEY}' < "$TEMPLATE" > "$VLLM_TMP"

# ── backup ─────────────────────────────────────────────────────────────────────
cp "$NGINX_CONF" "$BACKUP"
echo "[OK] Backed up to $BACKUP"

# ── update config ──────────────────────────────────────────────────────────────
VLLM_TMP="$VLLM_TMP" NGINX_CONF="$NGINX_CONF" python3 << 'PYEOF'
import os, re

nginx_conf = os.environ["NGINX_CONF"]
with open(nginx_conf) as f:
    conf = f.read()

with open(os.environ["VLLM_TMP"]) as f:
    vllm_block = f.read()

# 1. Remove stale vscode server block (conflicts on port 8001 with vLLM LLM)
conf = re.sub(
    r"^[ \t]*# vscode server\n[ \t]*server\s*\{[^}]*}\n",
    "",
    conf,
    flags=re.MULTILINE,
)

# 2. Remove any previous vLLM blocks (handles both old and new format)
#    New format has "# ─── /vLLM" end marker.
#    Old format (no end marker) ends at blank line before "map" or closing "}".
#    We match: header line, then all lines until either:
#      a) the end-marker line (new format), or
#      b) a lookahead for blank-line + map/} (old format)
conf = re.sub(
    r"^[ \t]*# ─── vLLM.*\n"
    r"(?:.*\n)*?"
    r"(?:"
    r"^[ \t]*# ─── /vLLM.*\n"
    r"|(?=\n[ \t]*(?:map |\}$))"
    r")",
    "",
    conf,
    flags=re.MULTILINE,
)

# 3. Insert vLLM blocks before the first "map" directive, or before closing http brace
if "map " in conf:
    idx = conf.index("map ")
    conf = conf[:idx] + vllm_block + "\n" + conf[idx:]
else:
    conf = conf.rstrip("\n")
    conf += "\n" + vllm_block + "\n}\n"

with open(nginx_conf, "w") as f:
    f.write(conf)
PYEOF

rm -f "$VLLM_TMP"

echo "[OK] vLLM blocks written to $NGINX_CONF"

# ── test and reload ────────────────────────────────────────────────────────────
if nginx -t 2>&1; then
    nginx -s reload
    echo "[OK] Nginx reloaded"
else
    echo "[ERROR] nginx -t failed — restoring backup" >&2
    cp "$BACKUP" "$NGINX_CONF"
    exit 1
fi
