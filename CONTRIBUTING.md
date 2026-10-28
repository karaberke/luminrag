# Contributing to LuminRAG

Bug reports, documentation improvements and focused pull requests are welcome.
Start with the [local setup](docs/getting-started.md) and [architecture](code_logic.md).

## Development workflow

1. Create a branch for one clear change.
2. Keep credentials in your local `.env` or the application's AI settings. Never include keys, private course material, workspace exports or local agent configuration in a commit.
3. Run the relevant checks below and describe what you verified in the pull request.
4. Use a short, descriptive commit subject such as `fix: preserve citations when replacing a source`.

The reviewed environment templates (`.env.example`, `.env.selfhost.example`, and `frontend/.env.example`) contain example settings and no credentials. Real environment files stay local. Git ignore rules do not remove files that were already tracked: inspect both `git status --short` and `git diff --cached` before publishing.

Report vulnerabilities using the [security policy](SECURITY.md), without including live credentials in a public issue.

## Verify

The maintainer keeps the regression and security test suites locally; they are
excluded from the published repository and deployment images. In a checkout
with those local suites available, run:

```powershell
.venv/Scripts/python.exe -m pytest backend/tests -q
pnpm --dir frontend test
pnpm --dir frontend lint
pnpm --dir frontend build
```

On restricted Windows environments use an explicit workspace `--basetemp` and allow Python's internal loopback sockets for async tests. Provider tests use mocked SDK HTTP transports. Native ONNX tests require a model cache; set FASTEMBED_CACHE_PATH if already downloaded. Local browser smoke checks should cover onboarding, file upload, both query modes, graph editing/undo, workspace switching, backend restart hydration and `.lumin` import in a fresh profile. Real provider billing and cloud deployment are checked only with the user's supplied account settings.


Contributions are distributed under the repository's [AGPL-3.0 license](LICENSE).
