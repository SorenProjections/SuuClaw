from __future__ import annotations

import threading
import unittest

from suuclaw.core.agent import _invoke_with_timeout


class EntryTimeoutTests(unittest.TestCase):
    def test_returns_fast_entry_result(self):
        self.assertEqual(_invoke_with_timeout(lambda: {"route": "chat"}, 1.0), {"route": "chat"})

    def test_times_out_without_waiting_for_blocked_entry_thread(self):
        release = threading.Event()
        try:
            with self.assertRaisesRegex(TimeoutError, "entry route exceeded"):
                _invoke_with_timeout(lambda: release.wait(5.0), 0.01)
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
