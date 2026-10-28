"""Portable development, Docker, and serverless entry point."""
import os
from pathlib import Path

import uvicorn

from backend.utils.environment import hosted_mode


def main() -> None:
    env_file = Path(__file__).parent.parent / ".env"
    if not hosted_mode() and os.getenv("LUMINRAG_SKIP_DOTENV") != "1" and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
    hosted = hosted_mode()
    uvicorn.run(
        "backend.main:app",
        host=os.getenv("HOST", "0.0.0.0" if hosted else "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("LUMIN_RELOAD") == "1" and not hosted,
        # Operators may supply exact reverse-proxy IPs/CIDRs. Never trust
        # arbitrary forwarding headers by default on a public listener.
        proxy_headers=bool(os.getenv("FORWARDED_ALLOW_IPS")),
        forwarded_allow_ips=os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        access_log=False,
    )


if __name__ == "__main__":
    main()
