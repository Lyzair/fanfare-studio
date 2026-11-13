#!/usr/bin/env bash
# Docker smoke test: build, start, wait until healthy, check the origin rules, synthesize German.
# First run downloads the models (about 3 GB) into the `models` volume; `down -v` removes it again
# unless KEEP_VOLUMES=1 is set (then the models stay for the next run).
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT=fanfare-smoke
PORT=${PORT:-5123}
FILES=(-f compose.yaml)
[ -n "${COMPOSE_OVERRIDE:-}" ] && FILES+=(-f "$COMPOSE_OVERRIDE")   # e.g. compose.rocm.yaml
export FANFARE_ALLOWED_ORIGINS="https://fanfare.example"
dc() { docker compose -p "$PROJECT" "${FILES[@]}" "$@"; }
OUT=$(mktemp -d)
cleanup() {
  dc logs --tail 30 fanfare-studio 2>/dev/null | tail -30 || true
  if [ "${KEEP_VOLUMES:-0}" = 1 ]; then dc down; else dc down -v; fi
  rm -rf "$OUT"
}
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; exit 1; }

t0=$SECONDS
dc build
echo "build: $((SECONDS - t0)) s; image size: $(docker image inspect --format '{{.Size}}' "$(dc images -q fanfare-studio | head -1)" | awk '{printf "%.2f GB", $1/1e9}')"
dc up -d
t1=$SECONDS
cid=$(dc ps -q fanfare-studio)
until [ "$(docker inspect --format '{{.State.Health.Status}}' "$cid")" = healthy ]; do
  [ "$(docker inspect --format '{{.State.Running}}' "$cid")" = true ] || fail "container stopped"
  [ $((SECONDS - t1)) -lt 1800 ] || fail "not healthy after 30 min"
  sleep 5
done
echo "healthy after $((SECONDS - t1)) s (includes model download on first run)"

base="http://127.0.0.1:$PORT"
check() {  # origin expected_status expected_acao
  local hdr; hdr=$(curl -s -o /dev/null -D - -H "Origin: $1" "$base/voices")
  local code; code=$(printf '%s' "$hdr" | head -1 | awk '{print $2}')
  local acao; acao=$(printf '%s' "$hdr" | tr -d '\r' | awk -F': ' 'tolower($1)=="access-control-allow-origin"{print $2}')
  [ "$code" = "$2" ] || fail "Origin $1: expected $2, got $code"
  [ "$acao" = "$3" ] || fail "Origin $1: expected ACAO '$3', got '$acao'"
  echo "ok: GET /voices with Origin $1 -> $code ${acao:+(ACAO $acao)}"
}
check "http://localhost:8080" 200 "http://localhost:8080"
check "https://fanfare.example" 200 "https://fanfare.example"   # allowed via FANFARE_ALLOWED_ORIGINS
check "https://evil.example" 403 ""

t2=$SECONDS
ctype=$(curl -s -o "$OUT/de.wav" -w '%{content_type}' -X POST "$base/synthesize" \
  -H 'Content-Type: application/json' -H 'X-Fanfare: 1' \
  -d '{"voice":"de-high","text":"Und jetzt, mit der Nummer sieben: Lea!"}')
[ "$ctype" = "audio/wav" ] || fail "synthesize returned content-type '$ctype'"
head -c 4 "$OUT/de.wav" | grep -q RIFF || fail "response is not a WAV"
echo "ok: POST /synthesize -> audio/wav, $(stat -c %s "$OUT/de.wav") bytes in $((SECONDS - t2)) s"
echo "SMOKE OK (total $((SECONDS - t0)) s)"
