[← Back to LuminRAG](../README.md)

# Getting started

Python 3.12+ and Node 22+ with pnpm 10 are required. Use a virtual environment.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip --isolated install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r requirements-bootstrap.lock
.venv/Scripts/python.exe -m pip --isolated install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r requirements-hosted.lock
.venv/Scripts/python.exe -m pip check
pnpm --dir frontend install --frozen-lockfile
```

Start the backend and frontend in separate terminals:

```powershell
# Public-mode development: isolated temporary workspaces and browser BYOK.
$env:LUMIN_HOSTED = '1'
.venv/Scripts/python.exe -m backend
```

```powershell
pnpm --dir frontend dev --host 127.0.0.1
```

Open http://127.0.0.1:5173. Connect a provider in **AI settings**, choose fast and smart models, and upload material. The default frontend proxy sends API requests to localhost:8000; frontend/.env.example documents a separate API origin. First model startup downloads the pinned ONNX snapshots; every runtime load verifies their SHA-256 hashes. Matching cached files are reused. The hosted image verifies them during its build and runs offline.

## macOS and Linux

Use the same Node.js and pnpm requirements. Create and install the Python environment with POSIX paths:

```bash
python3 -m venv .venv
.venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r requirements-bootstrap.lock
.venv/bin/python -m pip --isolated install --index-url https://pypi.org/simple --require-hashes --only-binary=:all: -r requirements-hosted.lock
.venv/bin/python -m pip check
pnpm --dir frontend install --frozen-lockfile
LUMIN_HOSTED=1 .venv/bin/python -m backend
```

In a second terminal, run `pnpm --dir frontend dev --host 127.0.0.1`. The PowerShell launcher below is for Windows.

## Installation notes

Keep the virtual environment isolated: `.venv/pyvenv.cfg` must say `include-system-site-packages = false`. Use `requirements.lock` instead of `requirements-hosted.lock` for self-hosted audio/video and the test dependencies. The optional `requirements-av.lock` and `requirements-ner.lock` include their complete dependency sets. All locks include hashes and platform markers for supported Windows/Linux Python versions; the adjacent `.txt` files are their maintained inputs. Container bases use immutable digests. Retrieval models are allowlisted in `backend/utils/model_manifest.json`; adding a different model requires recording its immutable revision and verified file hashes there. ONNX Runtime telemetry is disabled before model or parser initialization.

After installing dependencies, Windows users can start both services with one command:

```powershell
.\scripts\run_local.ps1
```

Use `-SelfHosted` for persistent local mode, or `-BackendPort 8010 -FrontendPort 5174` if the default ports are busy. Ctrl+C stops the services started by the script.

For a persistent self-hosted server, omit LUMIN_HOSTED. Local loopback access is allowed; remote access requires LUMINRAG_API_TOKEN. The backend can use .env and config/llm.yaml, or credentials supplied through the UI. Ollama is available only in self-hosted mode. The local .env is ignored by Git and Docker.

### Self-hosted Docker with audio/video

```powershell
# Copy .env.example to .env only when .env does not already exist.
# Set a long random LUMINRAG_API_TOKEN before starting Compose.
docker compose up --build
```

The frontend is at http://localhost and the backend port binds to loopback. The self-hosted Dockerfile installs the hashed `requirements.lock` plus ffmpeg. Optional NER is in `requirements-ner.lock`. The public image excludes torch, FAISS, Whisper, OpenCV, scikit-image, matplotlib, GLiNER, spaCy and AWS libraries.

Application code, configuration and pinned retrieval weights are root-owned and read-only in both images. Compose also makes the container filesystems read-only, disables privilege escalation and drops capabilities; only data volumes, the optional model cache and temporary directories are writable. A short self-hosted bootstrap retains the four permissions needed to repair older data volumes and switch users, then Python runs as UID 1000 with no capabilities. The frontend runs as UID 101 on internal port 8080, mapped to localhost port 80.
