from __future__ import annotations

import tkinter as tk
import unittest
from unittest.mock import patch

from app import ObserverApp


class ObserverAppDefaultsTests(unittest.TestCase):
    def test_automatic_refresh_starts_unchecked(self) -> None:
        root = tk.Tk()
        root.withdraw()
        try:
            app = ObserverApp(root)
            self.assertFalse(app.auto_refresh.get())
        finally:
            root.destroy()


class F8ToggleTests(unittest.TestCase):
    def _app(self):
        root = tk.Tk()
        root.withdraw()
        return root, ObserverApp(root)

    def test_f8_press_starts_automatic_run(self) -> None:
        root, app = self._app()
        try:
            with patch("app.emergency_pressed", return_value=True):
                app._poll_emergency()
            self.assertEqual(app.mode_text.get(), "1 秒后启动")
            self.assertEqual(str(app.start_button.cget("state")), "disabled")
        finally:
            app.stop_event.set()
            root.destroy()

    def test_held_f8_only_triggers_once(self) -> None:
        root, app = self._app()
        try:
            with patch("app.emergency_pressed", return_value=True):
                app._poll_emergency()
                first_mode = app.mode_text.get()
                app._poll_emergency()
                app._poll_emergency()
            self.assertEqual(app.mode_text.get(), first_mode)
            self.assertEqual(first_mode, "1 秒后启动")
        finally:
            app.stop_event.set()
            root.destroy()

    def test_f8_press_during_run_cancels(self) -> None:
        root, app = self._app()
        try:
            with patch("app.emergency_pressed", return_value=True):
                app._poll_emergency()
            with patch("app.emergency_pressed", return_value=False):
                app._poll_emergency()
            with patch("app.emergency_pressed", return_value=True):
                app._poll_emergency()
            self.assertEqual(app.mode_text.get(), "观察")
            self.assertIn("紧急停止", app.status.get())
            self.assertEqual(str(app.start_button.cget("state")), "normal")
        finally:
            app.stop_event.set()
            root.destroy()


if __name__ == "__main__":
    unittest.main()
