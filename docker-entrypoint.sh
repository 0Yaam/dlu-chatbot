#!/bin/sh
set -eu

mkdir -p /app/data /app/vector_store /app/.cache
chmod -R u+rwX /app/data /app/vector_store /app/.cache 2>/dev/null || true

if [ -f /app/.env ]; then
  exec "$@"
fi

required_vars="TOKEN WEBHOOK_URL OPENROUTER_API_KEY"
missing_vars=""

for var_name in $required_vars; do
  eval "var_value=\${$var_name:-}"
  if [ -z "$var_value" ]; then
    missing_vars="$missing_vars $var_name"
  fi
done

if [ -n "$missing_vars" ]; then
  echo "Missing required environment variables:${missing_vars}" >&2
  echo "Provide them with 'env_file: .env', '--env-file .env', or mount '.env' to '/app/.env'." >&2
  exit 1
fi

exec "$@"
