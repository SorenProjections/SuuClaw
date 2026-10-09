"""Allow ``python -m suuclaw.feishu`` to start the gateway."""

from .app import main

if __name__ == "__main__":
    raise SystemExit(main())
