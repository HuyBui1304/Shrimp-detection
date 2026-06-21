#!/usr/bin/env bash

set -euo pipefail

# Select a local, ngrok, or explicit API URL.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "${1:-}" ]; then
    echo "Usage:"
    echo "  ./scripts/run_benchmark.sh local"
    echo "  ./scripts/run_benchmark.sh ngrok"
    echo ""
    echo "Example with an explicit URL:"
    echo "  ./scripts/run_benchmark.sh https://abc123.ngrok.io"
    exit 1
fi

if [ "$1" == "local" ]; then
    API_URL="http://localhost:8000/predict"
elif [ "$1" == "ngrok" ]; then
    # Read the public URL from the local ngrok API.
    NGROK_URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -m json.tool 2>/dev/null | grep -o '"public_url":"[^"]*"' | head -1 | cut -d'"' -f4)
    if [ -z "$NGROK_URL" ]; then
        echo "Could not find an ngrok URL. Make sure ngrok is running."
        echo "Run: ngrok http 8000"
        exit 1
    fi
    API_URL="${NGROK_URL}/predict"
    echo "Found ngrok URL: $API_URL"
else
    # Accept a fully qualified URL as an alternative.
    if [[ "$1" == http* ]]; then
        API_URL="${1%/}"
        if [[ "$API_URL" != */predict ]]; then
            API_URL="$API_URL/predict"
        fi
    else
        API_URL="$1"
    fi
fi

echo "Running benchmark against: $API_URL"
echo ""

python "$SCRIPT_DIR/benchmark_api.py" \
  --api-url "$API_URL" \
  --num-requests 100 \
  --no-save-annotated
