# Fanfare Studio: manual

Fanfare Studio makes the stadium-announcer voices for Fanfare on your own computer. It sounds more
natural than the voices built into Fanfare and can even speak with your own announcer's voice.
Nothing is uploaded: names and texts stay on this computer.

You use it **at home, before the game**: Fanfare asks Studio for every announcement, you export
the result, and on game day your phone or laptop plays the prepared clips. Studio is not needed
at the pitch.

## 1. What you need

- A computer with Windows, macOS or Linux. A normal processor is enough; no graphics card needed.
- About 4 GB of free disk space and 4 GB of free memory.
- Internet for the first start (it downloads the voice model, about 1.6 GB, once).
- The free tool **uv**, which installs everything else (Python included).

## 2. Install uv (once)

Follow https://docs.astral.sh/uv/getting-started/installation/ . In short:

- **Windows:** open "PowerShell" and paste
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **macOS and Linux:** open "Terminal" and paste
  `curl -LsSf https://astral.sh/uv/install.sh | sh`

Close the window afterwards, so the next one knows about uv.

## 3. Start Fanfare Studio

Unpack the download (`fanfare-studio-<version>.zip` or `.tar.gz`). In the folder `fanfare-studio`:

- **Windows:** double-click `start-studio.bat`
- **macOS:** double-click `start-studio.command` (the first time: right-click, Open)
- **Linux:** run `./start-studio.sh`

A window opens. The **first start takes a while**: it installs the program and downloads about
1.6 GB of voice model. When it says `Fanfare Studio ready`, it is running.
**Keep the window open** while you use it. To stop, close it or press Ctrl+C.

If it says that uv is missing, do step 2 first.

## 4. Connect Fanfare

1. Open Fanfare in a browser on **the same computer**, e.g. https://lyzair.github.io/fanfare/ .
2. Step 4 "Sounds & voice" → "Local voice server (advanced)" → **Connect**
   (the address `http://127.0.0.1:5123` is filled in).
3. The first time, the Studio window asks:
   `Website https://lyzair.github.io wants to use Fanfare Studio. Allow? [y/N]`.
   Type `y` and press Enter, then press Connect again. Studio remembers this website.
   You can also allow it in advance:
   `start-studio.bat allow https://lyzair.github.io` (Windows) or
   `./start-studio.sh allow https://lyzair.github.io` (macOS/Linux).
   Using Fanfare from your club's own address? Allow that address instead.
4. Pick a voice from "On this computer", press **Prepare voices**, listen, and **Export**.
   Import the file on the phone you use at the game.

## 5. Your club's own voice

Studio can speak with the voice of your announcer, coach or a volunteer. Record 15–20 seconds
of them speaking like an announcer, in the language of the announcements, in a quiet room.
Then add it in Fanfare ("🎙 Record club voice"), or in the folder with:

    ./start-studio.sh voice add --name "Announcer Max" --lang en --file max.wav --consent
    ./start-studio.sh voice list
    ./start-studio.sh voice delete club-1a2b3c4d

(On Windows write `start-studio.bat` instead of `./start-studio.sh`.)

**Only with permission.** A voice is personal data. `--consent` confirms that the person agreed
that the recording is stored on this computer and used for announcements, and that they can ask
for it to be deleted at any time. Delete it when they ask. Never use the voice of famous people
or anyone who has not agreed. Every clip carries an inaudible watermark marking it as made by AI.

## 6. Stop, update, uninstall

- **Stop:** close the window.
- **Update:** download the new version, unpack it, start it. Your allowed websites and club
  voices are kept (they are stored in your user folder, not in the program folder).
- **Uninstall:** delete the `fanfare-studio` folder. To also remove the downloaded voice model,
  delete the folder `.cache/huggingface` in your user folder. Club voices live in
  `~/.local/share/fanfare-studio` (Linux), `~/Library/Application Support/fanfare-studio` (macOS)
  or `%LOCALAPPDATA%\fanfare-studio` (Windows).

## 7. With Docker (optional)

Instead of steps 2 and 3 you can run the ready-made image (for PCs and ARM computers such as
Apple silicon Macs):

    docker run -d --name fanfare-studio -p 127.0.0.1:5123:5123 \
      -e FANFARE_ALLOWED_ORIGINS=https://lyzair.github.io \
      -v fanfare-studio-models:/models -v fanfare-studio-voices:/data -v fanfare-studio-config:/config \
      --read-only --tmpfs /tmp --tmpfs /home/fanfare:uid=10001,gid=10001 \
      --cap-drop ALL --security-opt no-new-privileges:true --restart unless-stopped \
      ghcr.io/lyzair/fanfare-studio:1

Keep `127.0.0.1:5123:5123` exactly like this: it makes Studio reachable from this computer only.
The first start downloads about 3 GB; `docker logs -f fanfare-studio` shows the progress.

## 8. Graphics cards: build it yourself

The ready-made image and the start files use the processor (CPU), which sounded best in our
tests. For an AMD or NVIDIA graphics card you build the image yourself from the source code
(https://github.com/Lyzair/fanfare-studio); both are experimental:

- **AMD (ROCm, Linux):** the image is about 19 GB.

      export VIDEO_GID=$(getent group video | cut -d: -f3) RENDER_GID=$(getent group render | cut -d: -f3)
      docker compose -f compose.yaml -f compose.rocm.yaml up -d --build

- **NVIDIA (CUDA):** needs the NVIDIA Container Toolkit.

      docker compose -f compose.yaml -f compose.cuda.yaml up -d --build

## 9. Problems?

- **The window says port 5123 is already in use:** start with another port, e.g. `./start-studio.sh --port 5124`, and
  enter `http://127.0.0.1:5124` in Fanfare.
- **Fanfare says the website is not allowed:** see step 4.3.
- **The first start seems to hang:** the 1.6 GB download takes time on a slow connection. Wait.
