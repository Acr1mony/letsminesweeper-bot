from __future__ import annotations

import tkinter as tk
import unittest

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


if __name__ == "__main__":
    unittest.main()
