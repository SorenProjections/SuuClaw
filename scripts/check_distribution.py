"""Verify built packages before sharing or publishing them."""

from __future__ import annotations

import email
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
STATIC_FILES = ("index.html", "app.js", "style.css", "suuclaw-mark.svg")
EXCLUDED_PARTS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".uv-cache", ".test-tmp", ".suuclaw", ".workbuddy", "workspace", "sandbox",
}


def check_paths(paths: list[str]) -> None:
    for name in paths:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid package path: {name}")
        if set(path.parts) & EXCLUDED_PARTS:
            raise ValueError(f"Local runtime or cache included: {name}")
        if path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        ):
            raise ValueError(f"Private configuration included: {name}")
        if path.suffix in {".pyc", ".pyo", ".pem", ".p12", ".pfx", ".key"}:
            raise ValueError(f"Generated or private file included: {name}")


def check_wheel(path: Path, license_text: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        paths = archive.namelist()
        check_paths(paths)
        metadata_paths = [name for name in paths if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError(f"Expected one package metadata file: {path.name}")
        metadata_path = metadata_paths[0]
        metadata = email.message_from_bytes(archive.read(metadata_path))
        if metadata.get("Name") != "suuclaw" or metadata.get("License-Expression") != "MIT":
            raise ValueError(f"Missing SuuClaw / MIT package metadata: {path.name}")
        if "LICENSE" not in metadata.get_all("License-File", []):
            raise ValueError(f"Missing license-file metadata: {path.name}")
        license_path = metadata_path.removesuffix("METADATA") + "licenses/LICENSE"
        if archive.read(license_path) != license_text:
            raise ValueError(f"Packaged license differs from the project license: {path.name}")
        for name in STATIC_FILES:
            if f"suuclaw/dashboard/static/{name}" not in paths:
                raise ValueError(f"Missing browser resource: {name}")


def check_sdist(path: Path, license_text: bytes) -> None:
    with tarfile.open(path, "r:gz") as archive:
        paths = archive.getnames()
        check_paths(paths)
        roots = {PurePosixPath(name).parts[0] for name in paths}
        if len(roots) != 1:
            raise ValueError(f"Expected one source distribution root: {path.name}")
        prefix = roots.pop()
        license_file = archive.extractfile(f"{prefix}/LICENSE")
        if license_file is None or license_file.read() != license_text:
            raise ValueError(f"Missing or mismatched source license: {path.name}")
        for name in STATIC_FILES:
            if f"{prefix}/src/suuclaw/dashboard/static/{name}" not in paths:
                raise ValueError(f"Missing browser source resource: {name}")


def main() -> None:
    license_text = (ROOT / "LICENSE").read_bytes()
    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
    if not wheels or not sdists:
        raise SystemExit("Run uv build first; both a wheel and a source archive are required.")
    for wheel in wheels:
        check_wheel(wheel, license_text)
        print(f"Verified {wheel.name}")
    for sdist in sdists:
        check_sdist(sdist, license_text)
        print(f"Verified {sdist.name}")


if __name__ == "__main__":
    main()
