[← Back to LuminRAG](../README.md)

# Deploy with Docker Compose and Cloudflare Tunnel

Requires Docker Engine and Docker Compose 2.17 or newer. Python, Node and uv
are not required on the server. The default Compose file runs the slim public
BYOK backend, website and a dedicated Cloudflare connector. It publishes no
host ports and does not attach to or restart your other applications.

## One-time Cloudflare setup

Open the [Cloudflare dashboard](https://dash.cloudflare.com/), select the
account containing your domain, and go to **Networking → Tunnels**. Create a
**new dedicated tunnel named LuminRAG**. Choose the Cloudflared connector and
copy only the token from its installation command; Compose runs that connector
for you, so do not also run the dashboard's installation command manually.
Copy its connector token into your local `.env` as described below. Do not
reuse a tunnel serving other apps: its connectors would receive routes they
cannot reach, causing intermittent failures.

In that tunnel add a published application route:

| Setting | Value |
| --- | --- |
| Hostname | `luminrag.karaberke.com` (or your chosen hostname) |
| Service type | HTTP |
| Service URL | `127.0.0.1:8080` |
| HTTP Host Header | Leave unset |

The website, backend and connector share an isolated network namespace. The
website listens on loopback port 8080, the backend on loopback port 8000.
Cloudflare provides external HTTPS. No second API subdomain, Pages project,
manual Docker network, setup script, or source archive is needed.

## Configure and start

```bash
gh repo clone karaberke/luminrag
cd luminrag
cp -n .env.example .env
chmod 600 .env
nano .env
```

Set these two values in `.env`:

```dotenv
LUMIN_HOSTNAME=luminrag.karaberke.com
CLOUDFLARE_TUNNEL_TOKEN=your-dedicated-tunnel-token
```

Use the token only, not the complete Cloudflare installation command. The
other settings in `.env.example` have conservative defaults for the shared
8 GiB server. Never commit `.env` or paste the token into an issue or log.

```bash
sudo docker compose up -d --build --wait --wait-timeout 300
sudo docker compose ps
```

Open your HTTPS hostname and connect Anthropic or OpenRouter in **AI settings**.
Each visitor supplies their own provider key; no shared AI key belongs in the
server `.env`. Provider keys are held in the current tab's memory and temporarily
in server memory during processing; they are never written to browser storage,
server storage, logs or backups. Prompts and relevant source
excerpts go to that selected provider.
The tunnel token is passed only to the connector through its environment,
never to the app containers or image builds.

The default backend ceiling is one CPU and 2 GiB RAM, including temporary
files. It uses one model thread, one concurrent generation request, two cached
workspaces, 64 MiB per workspace, and 10 MiB per uploaded file. The website and
connector have separate small resource limits. `COMPOSE_PARALLEL_LIMIT=1`
reduces concurrent Docker operations during the first build. Limits on running
containers do not cap image-build resource use. These settings are a pilot
starting point, not a measured simultaneous-user capacity; raise them only
after observing representative uploads and queries alongside existing apps.

Originals, graphs and chats are saved in each visitor's browser. Backend
workspaces are temporary caches and restore from browser data after a restart.
Export `.lumin` backups before clearing browser data or changing hostnames.
Large imports can exceed the pilot workspace limit. Cloudflare plan-specific
upload and response-duration limits also apply to unusually large uploads or
long Thorough queries.

## Verify, monitor and update

```bash
curl --fail --show-error https://luminrag.karaberke.com/api/ready
sudo docker compose stats --no-stream
sudo docker compose logs --tail 60 backend cloudflared
```

Verify an upload, Dense/Graph/Hybrid answers, source citations/page navigation,
graph edits/Undo, backup export/import and browser restore after restart before
broad release. Check while Jellyfin or Immich is busy, not only when idle.

To update, export a browser backup, keep `.env` local, then run:

```bash
git pull --ff-only
sudo docker compose up -d --build --force-recreate --wait --wait-timeout 300
```

Recreate all three services together: backend and connector share the website
container's network namespace. `depends_on` coordinates startup and explicit
Compose restarts. No changes to your existing tunnel containers are required.
Stop only this stack with `sudo docker compose stop`.

## GitHub, Google and Apple account setup

### GitHub — required for cloning and updates

The source is [karaberke/luminrag](https://github.com/karaberke/luminrag).
Use your existing GitHub account and `gh repo clone karaberke/luminrag` on the
server. If GitHub CLI requests authentication, run `gh auth login` and choose
GitHub.com with browser authentication. No GitHub OAuth application, deploy
key, Actions secret or client secret is required for this server deployment.
Keep the deployment token only in the server's ignored `.env` file.

### Google — optional future integration

There is no Google sign-in or Google Drive synchronization in this version;
no Google Cloud project, billing account, API key or OAuth client is needed.
Adding Google values to `.env` will not enable those features.

If Google sign-in is implemented later, use the
[Google Auth Platform console](https://console.cloud.google.com/auth/overview):
create a project, configure branding and audience, create a Web application
OAuth client, and register `https://luminrag.karaberke.com` as its JavaScript
origin. Register a redirect URI only after its callback handler is implemented;
this version does not have one. Follow Google's
[official client setup](https://developers.google.com/identity/oauth2/web/guides/get-google-api-clientid).
Drive sync would additionally need the appropriate Drive API/scopes and its
own implementation and consent review.

### Apple — optional future integration

There is no Sign in with Apple integration in this version, so no Apple
Developer membership or keys are needed for launch. For a future implementation,
Apple requires a primary Apple-platform App ID enabled for Sign in with Apple,
a linked Services ID for the website, and registered domains/return URLs.
Configure these only when the application supports the callback. See Apple's
[official web configuration](https://developer.apple.com/help/account/capabilities/configure-sign-in-with-apple-for-the-web/)
and [Developer Program details](https://developer.apple.com/programs/whats-included/).

### AI provider — chosen by each visitor

Visitors can create an [OpenRouter account](https://openrouter.ai/) and use
**Connect OpenRouter**, or obtain an Anthropic key through the
[Anthropic Console](https://console.anthropic.com/). OpenRouter uses the
[PKCE authorization flow](https://openrouter.ai/docs/guides/overview/auth/oauth)
already implemented in LuminRAG; no shared developer client secret is needed.
Provider keys remain in memory during use and are not saved by LuminRAG.

## Persistent self-hosted alternative

For Ollama, optional audio/video or persistent server-side storage, use the
separate older profile and template:

```bash
cp -n .env.selfhost.example .env.selfhost
chmod 600 .env.selfhost
nano .env.selfhost
# Set a long random LUMINRAG_API_TOKEN, plus provider settings if desired.
sudo docker compose --env-file .env.selfhost -f docker-compose.selfhost.yml up -d --build
```

That profile exposes the website only on host loopback port 80 and requires
its own reverse proxy for remote HTTPS. It is heavier than the public BYOK
profile because it includes optional media-processing dependencies.

## Other hosting providers

`Dockerfile.hosted` remains portable to Hugging Face Docker Spaces and Cloud
Run. For Hugging Face, `scripts/prepare_space.py` prepares a clean backend
checkout; set exact `ALLOWED_ORIGINS` and configure the separate frontend's
`VITE_API_URL`. For Cloud Run, `deploy/cloudrun.yaml` uses one instance and
instance-based CPU billing so background ingestion can continue after the
initial response. Check current provider prices before creating paid resources.
