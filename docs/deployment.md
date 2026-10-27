[← Back to LuminRAG](../README.md)

# Deploy: Hugging Face + Cloudflare Pages

Check [Hugging Face pricing](https://huggingface.co/pricing), [Spaces availability](https://huggingface.co/docs/hub/spaces-overview), and [Cloudflare Pages limits](https://developers.cloudflare.com/pages/platform/limits/) before deploying. Users pay their model provider charges separately.

### Backend

1. Create a **Docker Space**, choose **CPU Basic**, and keep one application process.
2. From the application checkout, prepare its clean deployment folder:

   ```powershell
   .venv/Scripts/python.exe scripts/prepare_space.py --output .tmp/hf-space
   ```

3. Initialize/push that prepared folder to your Space repository. It contains the slim public Dockerfile under HF's required `Dockerfile` name. Do not push the application root's self-hosted Dockerfile to the Space. The prepared folder excludes .env, local data, originals, tests and dependencies.
4. In Space Variables set `ALLOWED_ORIGINS=https://YOUR-PROJECT.pages.dev` (add the exact custom domain if needed). No provider keys or AWS secrets are needed.
5. Wait for `/api/ready` to return 200. The README's `sdk: docker` and `app_port: 8000` match the image.

### Frontend

Create a Cloudflare Pages project connected to this repository:

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Build command | `pnpm install --frozen-lockfile && pnpm build` |
| Output directory | `dist` |
| Environment | `VITE_API_URL=https://YOUR-SPACE.hf.space` |

Use HTTPS before entering any key. The build adds the exact `VITE_API_URL` origin to the included browser security headers; rebuild when changing that origin. A source link is provided in settings; publish the corresponding deployed source under AGPL-3.0.

### Cloud Run alternative

`Dockerfile.hosted` also works on Cloud Run. Build/push it to Artifact Registry, substitute the image and frontend origin in deploy/cloudrun.yaml, and deploy it with public invocation enabled. The configuration uses **at most one instance**, two GiB RAM, at most two cached workspaces and **instance-based CPU billing**. Background ingestion must continue after its initial 202 response, so request-only CPU billing is unsuitable for this version. Keep minimum instances 0. Set a Cloud Billing spend cap appropriate to your account and review [Cloud Run pricing](https://cloud.google.com/run/pricing); free-tier use is possible, not guaranteed.

No S3 upload flow or hosted source-file server is required. Retire old AWS resources only after exporting any data you need and confirming the new deployment works.
