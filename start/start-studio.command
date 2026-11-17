#!/bin/sh
# Fanfare Studio: start the local voice server (stop it with Ctrl+C or by closing this window).
# Options are passed on, e.g.  ./start-studio.sh --port 5124
cd "$(dirname "$0")" || exit 1
if ! command -v uv >/dev/null 2>&1; then
  echo "Fanfare Studio needs \"uv\" (free). Install it once, then start this file again:"
  echo "  https://docs.astral.sh/uv/getting-started/installation/"
  echo "Fanfare Studio braucht \"uv\" (kostenlos). Einmal installieren, dann diese Datei erneut starten:"
  echo "  https://docs.astral.sh/uv/getting-started/installation/"
  printf 'Press Enter to close / Enter zum Schliessen. '
  read -r _
  exit 1
fi
echo "Starting Fanfare Studio. The first start downloads about 1.6 GB of voice model from Hugging Face."
echo "Fanfare Studio startet. Der erste Start laedt etwa 1,6 GB Stimmenmodell von Hugging Face."
exec uv run --frozen --no-dev --extra onnx --extra cpu fanfare-studio "$@"
