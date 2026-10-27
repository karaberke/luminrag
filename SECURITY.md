# Security policy

## Reporting a vulnerability

Use [GitHub private vulnerability reporting](https://github.com/karaberke/luminrag/security/advisories/new) to report a suspected vulnerability. Include the affected version or commit, reproduction steps, expected behavior and impact. Use synthetic data and redact credentials, workspace IDs and private source material.

Do not post live secrets or private documents in public issues. If a credential has been exposed, revoke or rotate it through its provider; removing a file from Git does not invalidate the credential.

## Supported code

Security fixes target the current `main` branch. There is no separate long-term support release series.

## Deployment and credentials

- Use HTTPS for remote deployments. Restrict `ALLOWED_ORIGINS` to the exact frontend origins you operate.
- Configure `LUMINRAG_API_TOKEN` before exposing self-hosted access beyond loopback.
- Hosted workspace IDs are bearer capabilities. Keep them out of URLs, screenshots, issue reports and logs.
- Knowledge bases and original documents persist in browser storage; keep `.lumin` backups in a location appropriate for their contents.
- Provider keys belong in local environment settings or the application's AI settings. Never commit them. Frontend build variables are public and must not contain secrets.
- The `.env.example` files are intentionally empty templates. Real environment files, private keys, local data and assistant configuration are ignored by Git and excluded from container builds.
- Parser limits reduce resource abuse; they are not an operating-system isolation boundary against unknown native-parser exploits.

See [deployment](docs/deployment.md) and [configuration](docs/reference.md) for the implemented limits and operational requirements.
