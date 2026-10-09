from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from suuclaw.feishu.artifacts import ArtifactTracker, build_artifact_archive


def _call(call_id: str, path: str, *, name: str = "file_write") -> dict:
    return {
        "type": "tool_call",
        "id": call_id,
        "name": name,
        "args": {"file_path": path},
    }


def _result(call_id: str, *, name: str = "file_write", ok: bool = True) -> dict:
    return {"type": "tool_result", "id": call_id, "name": name, "ok": ok}


class ArtifactTrackerTests(unittest.TestCase):
    def test_collects_only_successful_non_sensitive_workspace_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            good = root / "result.html"
            failed = root / "failed.txt"
            secret = root / "api_token.txt"
            hidden = root / ".suuclaw" / "trace.json"
            canonical_hidden = root / ".suuclaw" / "trace.json"
            for path in (good, failed, secret, hidden, canonical_hidden):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data", encoding="utf-8")

            tracker = ArtifactTracker(root)
            for call_id, path, ok in (
                ("good", "result.html", True),
                ("failed", "failed.txt", False),
                ("secret", "api_token.txt", True),
                ("hidden", ".suuclaw/trace.json", True),
                ("canonical-hidden", ".suuclaw/trace.json", True),
            ):
                tracker.observe(_call(call_id, path))
                tracker.observe(_result(call_id, ok=ok))

            self.assertEqual(tracker.collect(), (good,))

    def test_rejects_absolute_path_outside_workspace_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as other:
            root = Path(tmp).resolve()
            inside = root / "report.pdf"
            outside = Path(other).resolve() / "outside.pdf"
            inside.write_bytes(b"pdf")
            outside.write_bytes(b"pdf")

            tracker = ArtifactTracker(root)
            tracker.observe(_call("inside-1", str(inside)))
            tracker.observe(_result("inside-1"))
            tracker.observe(_call("inside-2", str(inside), name="file_edit"))
            tracker.observe(_result("inside-2", name="file_edit"))
            tracker.observe(_call("outside", str(outside)))
            tracker.observe(_result("outside"))

            self.assertEqual(tracker.collect(), (inside,))

    def test_builds_bounded_web_archive_in_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            html = root / "site" / "index.html"
            css = root / "site" / "style.css"
            html.parent.mkdir()
            html.write_text('<link rel="stylesheet" href="style.css">', encoding="utf-8")
            css.write_text("body { color: navy; }", encoding="utf-8")

            data = build_artifact_archive(
                (html, css),
                workspace=root,
                max_bytes=1024 * 1024,
            )

            self.assertIsNotNone(data)
            with zipfile.ZipFile(io.BytesIO(data or b"")) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["site/index.html", "site/style.css"],
                )
            self.assertIsNone(build_artifact_archive((html, css), workspace=root, max_bytes=1))

    def test_snapshots_are_outside_workspace_and_preserve_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "workdir"
            source = workspace / "site" / "index.html"
            source.parent.mkdir(parents=True)
            source.write_text("first", encoding="utf-8")
            tracker = ArtifactTracker(workspace)
            tracker.observe(_call("write", "site/index.html"))
            tracker.observe(_result("write"))

            snapshot = tracker.snapshot(
                root / "artifacts" / "turn-1",
                max_files=8,
                max_bytes=1024,
            )
            source.write_text("changed later", encoding="utf-8")

            self.assertEqual(snapshot.paths, (snapshot.root / "site" / "index.html",))
            self.assertEqual(snapshot.paths[0].read_text(encoding="utf-8"), "first")
            self.assertEqual(snapshot.failures, ())
            self.assertEqual(snapshot.omitted, 0)

    def test_snapshot_reports_oversized_file_without_copying_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "workdir"
            source = workspace / "large.bin"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"too large")
            tracker = ArtifactTracker(workspace)
            tracker.observe(_call("write", "large.bin"))
            tracker.observe(_result("write"))

            snapshot = tracker.snapshot(
                root / "artifacts" / "turn-1",
                max_files=8,
                max_bytes=1,
            )

            self.assertEqual(snapshot.paths, ())
            self.assertEqual(snapshot.failures, ("large.bin",))


if __name__ == "__main__":
    unittest.main()
