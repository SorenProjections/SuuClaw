from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from suuclaw.feishu.isolation import (
    activate_conversation_generation,
    active_conversation_generation,
    active_conversation_root,
    new_conversation_generation,
    prepare_conversation_workspace,
)


class ConversationIsolationTests(unittest.TestCase):
    def test_seeds_once_without_runtime_or_secret_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            target = root / "conversation" / "workdir"
            (template / "src").mkdir(parents=True)
            (template / "src" / "app.py").write_text("v1", encoding="utf-8")
            (template / ".env").write_text("SECRET=value", encoding="utf-8")
            (template / "notepad.md").write_text("old shared memory", encoding="utf-8")
            (template / ".suuclaw").mkdir()
            (template / ".suuclaw" / "checkpoint.json").write_text("old", encoding="utf-8")

            prepare_conversation_workspace(template, target)
            (target / "src" / "app.py").write_text("conversation", encoding="utf-8")
            (template / "src" / "app.py").write_text("v2", encoding="utf-8")
            prepare_conversation_workspace(template, target)

            self.assertEqual(
                (target / "src" / "app.py").read_text(encoding="utf-8"), "conversation"
            )
            self.assertFalse((target / ".env").exists())
            self.assertFalse((target / "notepad.md").exists())
            self.assertFalse((target / ".suuclaw" / "checkpoint.json").exists())
            self.assertTrue((target / ".suuclaw" / "feishu-workspace-seeded").is_file())

    def test_copies_existing_suuclaw_session_into_conversation_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            template.mkdir()
            legacy = root / "legacy"
            old_session = legacy / ".suuclaw" / "session"
            old_session.mkdir(parents=True)
            (old_session / "session.json").write_text('{"turn_index": 7}', encoding="utf-8")
            target = root / "conversation" / "workdir"

            prepare_conversation_workspace(template, target, legacy_workspace=legacy)

            migrated = target / ".suuclaw" / "session" / "session.json"
            self.assertEqual(migrated.read_text(encoding="utf-8"), '{"turn_index": 7}')
            self.assertTrue((legacy / ".suuclaw" / "session" / "session.json").is_file())
            self.assertTrue(old_session.exists())

    def test_active_generation_switch_is_persistent_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            namespace = Path(tmp) / "conversation"
            generation, root = new_conversation_generation(namespace)
            root.mkdir(parents=True)

            selected = activate_conversation_generation(namespace, generation)

            self.assertEqual(selected, root.resolve())
            self.assertEqual(active_conversation_generation(namespace), generation)
            self.assertEqual(active_conversation_root(namespace), root)

            (namespace / "active-session.json").write_text(
                '{"generation": "../escape"}', encoding="utf-8"
            )
            self.assertEqual(active_conversation_generation(namespace), "")
            self.assertEqual(active_conversation_root(namespace), namespace)


if __name__ == "__main__":
    unittest.main()
