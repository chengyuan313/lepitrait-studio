from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class AppSmokeTests(unittest.TestCase):
    def test_three_english_workflows_render(self):
        app = AppTest.from_file("../app.py")
        app.run(timeout=15)
        self.assertEqual(
            app.radio[0].options,
            ["Train Model", "Single Identification", "Batch Identification"],
        )
        self.assertEqual(app.title[0].value, "Train Model")
        self.assertFalse(app.exception)

        app.radio[0].set_value("Single Identification")
        app.run(timeout=15)
        self.assertEqual(app.title[0].value, "Single Identification")
        self.assertFalse(app.exception)

        app.radio[0].set_value("Batch Identification")
        app.run(timeout=15)
        self.assertEqual(app.title[0].value, "Batch Identification")
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
