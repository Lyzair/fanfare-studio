# Fanfare Studio: Anleitung

Fanfare Studio erzeugt die Stadionsprecher-Stimmen für Fanfare auf deinem eigenen Computer. Es
klingt natürlicher als die in Fanfare eingebauten Stimmen und kann sogar mit der Stimme eures
eigenen Sprechers sprechen. Nichts wird hochgeladen: Namen und Texte bleiben auf diesem Computer.

Du benutzt es **zu Hause, vor dem Spiel**: Fanfare holt sich von Studio jede Ansage, du
exportierst das Ergebnis, und am Spieltag spielt dein Handy oder Laptop die vorbereiteten Clips.
Am Platz wird Studio nicht gebraucht.

## 1. Was du brauchst

- Einen Computer mit Windows, macOS oder Linux. Ein normaler Prozessor reicht, keine Grafikkarte nötig.
- Etwa 4 GB freien Speicherplatz und 4 GB freien Arbeitsspeicher.
- Internet beim ersten Start (das Stimmenmodell, etwa 1,6 GB, wird einmal heruntergeladen).
- Das kostenlose Werkzeug **uv**, das alles Weitere installiert (auch Python).

## 2. uv installieren (einmalig)

Anleitung: https://docs.astral.sh/uv/getting-started/installation/ . Kurz:

- **Windows:** "PowerShell" öffnen und einfügen:
  `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **macOS und Linux:** "Terminal" öffnen und einfügen:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`

Danach das Fenster schließen, damit das nächste uv kennt.

## 3. Fanfare Studio starten

Den Download (`fanfare-studio-<version>.zip` oder `.tar.gz`) entpacken. Im Ordner `fanfare-studio`:

- **Windows:** `start-studio.bat` doppelklicken
- **macOS:** `start-studio.command` doppelklicken (beim ersten Mal: Rechtsklick, Öffnen)
- **Linux:** `./start-studio.sh` ausführen

Ein Fenster öffnet sich. Der **erste Start dauert**: Das Programm wird installiert und etwa
1,6 GB Stimmenmodell geladen. Steht dort `Fanfare Studio ready`, läuft es.
**Lass das Fenster offen**, solange du es benutzt. Zum Beenden schließen oder Strg+C drücken.

Meldet es, dass uv fehlt: zuerst Schritt 2.

## 4. Fanfare verbinden

1. Fanfare im Browser **auf demselben Computer** öffnen, z. B. https://lyzair.github.io/fanfare/ .
2. Schritt 4 "Klänge & Stimme" → "Lokaler Stimmen-Server (für Fortgeschrittene)" → **Verbinden**
   (die Adresse `http://127.0.0.1:5123` ist schon eingetragen).
3. Beim ersten Mal fragt das Studio-Fenster:
   `Website https://lyzair.github.io wants to use Fanfare Studio. Allow? [y/N]`.
   `y` tippen, Enter drücken, dann in Fanfare erneut Verbinden. Studio merkt sich die Website.
   Du kannst sie auch vorher erlauben:
   `start-studio.bat allow https://lyzair.github.io` (Windows) oder
   `./start-studio.sh allow https://lyzair.github.io` (macOS/Linux).
   Nutzt ihr Fanfare unter einer eigenen Adresse des Vereins? Dann diese Adresse erlauben.
4. Unter "Auf diesem Computer" eine Stimme wählen, **Stimmen vorbereiten**, anhören, **Exportieren**.
   Die Datei auf dem Handy importieren, das ihr beim Spiel benutzt.

## 5. Die eigene Vereinsstimme

Studio kann mit der Stimme eures Sprechers, Trainers oder eines Ehrenamtlichen sprechen. Nimm
15–20 Sekunden auf, gesprochen wie ein Stadionsprecher, in der Sprache der Ansagen, in einem
ruhigen Raum. Dann in Fanfare hinzufügen ("🎙 Vereinsstimme aufnehmen") oder im Ordner mit:

    ./start-studio.sh voice add --name "Stadionsprecher Max" --lang de --file max.wav --consent
    ./start-studio.sh voice list
    ./start-studio.sh voice delete club-1a2b3c4d

(Unter Windows `start-studio.bat` statt `./start-studio.sh` schreiben.)

**Nur mit Einverständnis.** Eine Stimme ist ein personenbezogenes Datum (DSGVO). `--consent`
bestätigt, dass die Person zugestimmt hat, dass die Aufnahme auf diesem Computer gespeichert und
für Ansagen genutzt wird, und dass sie jederzeit die Löschung verlangen kann. Lösche sie, wenn
sie darum bittet. Nie die Stimme bekannter Personen oder von jemandem ohne Zustimmung verwenden.
Jeder Clip trägt ein unhörbares Wasserzeichen, das ihn als KI-erzeugt kennzeichnet.

## 6. Beenden, aktualisieren, deinstallieren

- **Beenden:** das Fenster schließen.
- **Aktualisieren:** neue Version herunterladen, entpacken, starten. Erlaubte Websites und
  Vereinsstimmen bleiben erhalten (sie liegen in deinem Benutzerordner, nicht im Programmordner).
- **Deinstallieren:** den Ordner `fanfare-studio` löschen. Um auch das Stimmenmodell zu entfernen,
  den Ordner `.cache/huggingface` in deinem Benutzerordner löschen. Vereinsstimmen liegen in
  `~/.local/share/fanfare-studio` (Linux), `~/Library/Application Support/fanfare-studio` (macOS)
  oder `%LOCALAPPDATA%\fanfare-studio` (Windows).

## 7. Mit Docker (optional)

Statt Schritt 2 und 3 kannst du das fertige Image benutzen (für PCs und ARM-Computer wie Macs
mit Apple-Chip):

    docker run -d --name fanfare-studio -p 127.0.0.1:5123:5123 \
      -e FANFARE_ALLOWED_ORIGINS=https://lyzair.github.io \
      -v fanfare-studio-models:/models -v fanfare-studio-voices:/data -v fanfare-studio-config:/config \
      --read-only --tmpfs /tmp --tmpfs /home/fanfare:uid=10001,gid=10001 \
      --cap-drop ALL --security-opt no-new-privileges:true --restart unless-stopped \
      ghcr.io/lyzair/fanfare-studio:1

`127.0.0.1:5123:5123` genau so lassen: Dann ist Studio nur von diesem Computer aus erreichbar.
Der erste Start lädt etwa 3 GB; `docker logs -f fanfare-studio` zeigt den Fortschritt.

## 8. Grafikkarten: selbst bauen

Das fertige Image und die Startdateien nutzen den Prozessor (CPU), der in unseren Tests am
besten klang. Für eine AMD- oder NVIDIA-Grafikkarte baust du das Image selbst aus dem Quellcode
(https://github.com/Lyzair/fanfare-studio); beides ist experimentell:

- **AMD (ROCm, Linux):** Das Image ist etwa 19 GB groß.

      export VIDEO_GID=$(getent group video | cut -d: -f3) RENDER_GID=$(getent group render | cut -d: -f3)
      docker compose -f compose.yaml -f compose.rocm.yaml up -d --build

- **NVIDIA (CUDA):** braucht das NVIDIA Container Toolkit.

      docker compose -f compose.yaml -f compose.cuda.yaml up -d --build

## 9. Probleme?

- **Das Fenster meldet, dass Port 5123 schon belegt ist:** mit einem anderen Port starten, z. B.
  `./start-studio.sh --port 5124`, und in Fanfare `http://127.0.0.1:5124` eintragen.
- **Fanfare meldet, die Website sei nicht erlaubt:** siehe Schritt 4.3.
- **Der erste Start scheint zu hängen:** Die 1,6 GB brauchen bei langsamer Verbindung Zeit. Warten.
