#!/bin/bash
# Render nginx.conf.template with values from .env and apply to /etc/nginx/nginx.conf.
# The template contains ONLY the vLLM server blocks — this script inserts them
# into the existing nginx.conf (replacing any previous vLLM blocks).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Source API_KEY from .env
set -a; source "$SCRIPT_DIR/.env"; set +a

if [ -z "$API_KEY" ]; then
    echo "ERROR: API_KEY not set in .env" >&2
    exit 1
fi

TEMPLATE="$SCRIPT_DIR/nginx.conf.template"
NGINX_CONF="/etc/nginx/nginx.conf"
BACKUP="/etc/nginx/nginx.conf.bak.$(date +%Y%m%d_%H%M%S)"

# Generate the vLLM server blocks from template
VLLM_BLOCK=$(API_KEY="$API_KEY" envsubst '${API_KEY}' < "$TEMPLATE")

# Back up current config
cp "$NGINX_CONF" "$BACKUP"
echo "[OK] Backed up to $BACKUP"

# Remove any existing vLLM blocks (between "# ── vLLM" and the next "^    }" at server-block level)
# Then insert the new block before the "map" section (or at end of http block if no maps).
sed -i '/^[[:space:]]*# ── vLLM/,/^[[:space:]]*# \(Dockerless\|map\)/{ /^[[:space:]]*# \(Dockerless\|map\)/!d; }' "$NGINX_CONF"

# Insert before the first map directive or before the closing http brace
if grep -q '^[[:space:]]*map ' "$NGINX_CONF"; then
    # Insert before the map section
    INSERT_LINE=$(grep -n '^[[:space:]]*map ' "$NGINX_CONF" | head -1 | cut -d: -f1)
    head -n $((INSERT_LINE - 1)) "$NGINX_CONF" > "$NGINX_CONF.tmp"
    echo "$VLLM_BLOCK" >> "$NGINX_CONF.tmp"
    echo "" >> "$NGINX_CONF.tmp"
    tail -n +$INSERT_LINE "$NGINX_CONF" >> "$NGINX_CONF.tmp"
else
    # Insert before the closing } of http block
    sed -i '/^}/i\'$'\n'"$VLLM_BLOCK"$'\n' "$NGINX_CONF"
fi

if [ -f "$NGINX_CONF.tmp" ]; then
    mv "$NGINX_CONF.tmp" "$NGINX_CONF"
fi

echo "[OK] vLLM blocks written to $NGINX_CONF"

# Test and reload
if nginx -t 2>&1; then
    nginx -s reload
    echo "[OK] Nginx reloaded"
else
    echo "[ERROR] nginx -t failed — restoring backup" >&2
    cp "$BACKUP" "$NGINX_CONF"
    exit 1
fi
