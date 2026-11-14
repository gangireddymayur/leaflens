import unittest

import numpy as np

from plantdx.metrics import accuracy, confusion_matrix, macro_f1, per_class_f1


class MetricTests(unittest.TestCase):
    def test_confusion_matrix(self):
        cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 0], 3)
        self.assertEqual(cm.tolist(), [[1, 1, 0], [0, 2, 0], [1, 0, 0]])

    def test_accuracy(self):
        self.assertAlmostEqual(accuracy([0, 1, 1, 0], [0, 1, 0, 0]), 0.75)

    def test_perfect_predictions_give_f1_one(self):
        cm = confusion_matrix([0, 1, 2, 2], [0, 1, 2, 2], 3)
        self.assertTrue(np.allclose(per_class_f1(cm), 1.0))
        self.assertAlmostEqual(macro_f1(cm), 1.0)

    def test_macro_f1_hand_computed(self):
        # class 0: tp=1 fp=1 fn=1 -> P=R=0.5 F1=0.5; class 1: tp=2 fp=1 fn=1 -> P=R=2/3 F1=2/3
        cm = np.array([[1, 1], [1, 2]])
        self.assertAlmostEqual(macro_f1(cm), (0.5 + 2 / 3) / 2)

    def test_absent_class_is_ignored(self):
        cm = confusion_matrix([0, 0], [0, 0], 3)
        self.assertAlmostEqual(macro_f1(cm), 1.0)


if __name__ == "__main__":
    unittest.main()
