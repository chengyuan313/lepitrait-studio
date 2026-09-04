from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_five_english_workflows_render(self):
        app = AppTest.from_file("../app.py")
        app.run(timeout=15)
        self.assertEqual(
            app.radio[0].options,
            [
                "Train Model",
                "Single Identification",
                "Batch Identification",
                "Single Trait Extraction",
                "Batch Trait Extraction",
            ],
        )
        for page in app.radio[0].options:
            app.radio[0].set_value(page)
            app.run(timeout=15)
            self.assertEqual(app.title[0].value, page)
            self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
