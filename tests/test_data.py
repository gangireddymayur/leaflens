import tempfile
import unittest
from pathlib import Path

from plantdx.data import find_data_root, scan_folder, stratified_split


class ScanTests(unittest.TestCase):
    def test_scan_folder_and_color_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls, n in (("healthy", 3), ("blight", 2)):
                (root / "color" / cls).mkdir(parents=True)
                for i in range(n):
                    (root / "color" / cls / f"{i}.jpg").write_bytes(b"x")
            (root / "color" / "blight" / "notes.txt").write_text("skip me")
            data_root = find_data_root(root)
            self.assertEqual(data_root.name, "color")
            classes, items = scan_folder(data_root)
            self.assertEqual(classes, ["blight", "healthy"])
            self.assertEqual(len(items), 5)

    def test_empty_folder_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                scan_folder(tmp)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.labels = [0] * 100 + [1] * 40 + [2] * 5

    def test_splits_are_disjoint_and_complete(self):
        tr, va, te = stratified_split(self.labels, seed=1)
        self.assertEqual(len(set(tr) | set(va) | set(te)), len(self.labels))
        self.assertFalse(set(tr) & set(va))
        self.assertFalse(set(tr) & set(te))
        self.assertFalse(set(va) & set(te))

    def test_every_class_in_every_split(self):
        tr, va, te = stratified_split(self.labels, seed=1)
        for split in (tr, va, te):
            self.assertEqual({self.labels[i] for i in split}, {0, 1, 2})

    def test_is_deterministic_for_a_seed(self):
        self.assertEqual(stratified_split(self.labels, seed=3), stratified_split(self.labels, seed=3))
        self.assertNotEqual(stratified_split(self.labels, seed=3), stratified_split(self.labels, seed=4))

    def test_proportions_roughly_hold(self):
        tr, va, te = stratified_split([0] * 1000, seed=0)
        self.assertAlmostEqual(len(tr) / 1000, 0.7, delta=0.01)
        self.assertAlmostEqual(len(va) / 1000, 0.15, delta=0.01)


if __name__ == "__main__":
    unittest.main()
