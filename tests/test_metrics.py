import unittest

import numpy as np

from eurolepi.metrics import macro_f1, should_reject, topk_accuracy


class MetricTests(unittest.TestCase):
    def test_topk_accuracy(self):
        logits = np.array([[0.9, 0.1, 0.0], [0.4, 0.5, 0.8]])
        targets = np.array([0, 1])
        self.assertEqual(topk_accuracy(logits, targets, 1), 0.5)
        self.assertEqual(topk_accuracy(logits, targets, 2), 1.0)

    def test_macro_f1(self):
        predictions = np.array([0, 0, 1, 1])
        targets = np.array([0, 0, 1, 1])
        self.assertEqual(macro_f1(predictions, targets, 2), 1.0)

    def test_rejection(self):
        self.assertTrue(should_reject(np.array([0.55, 0.45]), 0.65))
        self.assertFalse(should_reject(np.array([0.8, 0.2]), 0.65))


if __name__ == "__main__":
    unittest.main()

