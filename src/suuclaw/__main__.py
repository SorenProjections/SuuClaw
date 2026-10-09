"""Run ``python -m suuclaw`` using the public command dispatcher."""

from .entrypoints import main

if __name__ == "__main__":
    raise SystemExit(main())
