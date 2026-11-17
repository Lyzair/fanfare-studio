# Fanfare Studio

Better announcer voices for [Fanfare](https://github.com/Lyzair/fanfare), made on your own computer. Fanfare
asks Studio for every announcement; you prepare the clips at home, export them, and on game day the phone plays
them without Studio. It can also speak with your own announcer's voice (with their consent).

## Download

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once (it brings Python along).
2. Get `fanfare-studio-<version>.zip` (or `.tar.gz`) from the
   [latest release](https://github.com/Lyzair/fanfare-studio/releases/latest) and unpack it.
3. Double-click `start-studio.bat` (Windows) or `start-studio.command` (macOS), or run `./start-studio.sh` (Linux).
   The first start downloads about 1.6 GB of voice model. Keep the window open while you use it.

No graphics card needed: it runs on the processor. Step-by-step help: **manual** in
[English](docs/manual/en.md) and [Deutsch](docs/manual/de.md) (also inside the download).

## Connect Fanfare

Open Fanfare in a browser on the same computer, go to step 4 → "Local voice server (advanced)" → **Connect**.
The first time, the Studio window asks whether to allow the website: type `y`. Then pick a voice under
"On this computer" and press **Prepare voices**.

## Docker

```sh
docker run -d --name fanfare-studio -p 127.0.0.1:5123:5123 -e FANFARE_ALLOWED_ORIGINS=https://lyzair.github.io \
  -v fanfare-studio-models:/models -v fanfare-studio-voices:/data -v fanfare-studio-config:/config \
  --read-only --tmpfs /tmp --tmpfs /home/fanfare:uid=10001,gid=10001 \
  --cap-drop ALL --security-opt no-new-privileges:true --restart unless-stopped ghcr.io/lyzair/fanfare-studio:1
```

CPU image for PCs and ARM computers. AMD (ROCm) and NVIDIA (CUDA) versions: build them yourself
(manual, section 8).

## Privacy and security

- Reachable from this computer only (`127.0.0.1`); nothing is uploaded.
- The first start downloads the voice model from Hugging Face (with their telemetry switched off). To run fully
  offline afterwards, start it with the environment variable `HF_HUB_OFFLINE=1`.
- Only websites you allowed may use it; the window asks the first time.
- Logs never contain names or texts.
- Every clip carries an inaudible watermark that marks it as AI-generated.

## More

- [Manual](docs/manual/en.md) ([Deutsch](docs/manual/de.md)): install, start, connect, club voice, Docker, problems.
- [Developing and running in depth](docs/development.md): backends and hardware, install from source, Docker details,
  protocol, security model, quality checks, tests, releases.
- [Why these models](docs/model-choice.md).

## Licence

MIT (`LICENSE`). The voice model (Chatterbox) is MIT; the optional Piper engine is GPL-3.0, and Piper voices have
their own licences: [details](docs/development.md#licences).
