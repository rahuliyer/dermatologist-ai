import csv
import os
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from cancer_detector import (
    get_eval_transforms,
    get_model,
    get_train_transforms,
    write_results_csv,
)
from experiment_runner import BinaryLesionDataset, ExperimentRunner, normalize_image_id


def _save_rgb(path, seed):
    rng = np.random.RandomState(seed)
    array = rng.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _build_split(root, n_per_class, seed_offset):
    classes = ("melanoma", "nevus", "seborrheic_keratosis")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir)
        for i in range(n_per_class):
            _save_rgb(os.path.join(class_dir, "{}_{}.jpg".format(class_name, i)), seed_offset + class_idx * 10 + i)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="derm-ai-")
        self.data_root = os.path.join(self.tmpdir, "data")
        _build_split(os.path.join(self.data_root, "train"), 4, 0)
        _build_split(os.path.join(self.data_root, "valid"), 2, 100)
        _build_split(os.path.join(self.data_root, "test"), 2, 200)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_classifier_head_has_nonlinearities(self):
        model = get_model(arch="resnet18", pretrained=False, device=torch.device("cpu"))
        relu_count = sum(1 for module in model.fc.modules() if isinstance(module, nn.ReLU))
        linear_count = sum(1 for module in model.fc.modules() if isinstance(module, nn.Linear))
        dropout_count = sum(1 for module in model.fc.modules() if isinstance(module, nn.Dropout))
        self.assertGreaterEqual(relu_count, 1)
        self.assertGreaterEqual(linear_count, 2)
        self.assertGreaterEqual(dropout_count, 1)
        self.assertFalse(any(isinstance(module, nn.Sigmoid) for module in model.fc.modules()))

    def test_forward_handles_batch_size_one(self):
        model = get_model(arch="resnet18", pretrained=False, device=torch.device("cpu"))
        model.eval()
        logits = model(torch.randn(1, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (1, 1))

    def test_binary_dataset_labels(self):
        dataset = BinaryLesionDataset(
            os.path.join(self.data_root, "train"),
            transform=get_eval_transforms(64),
            positive_classes=("melanoma",),
        )
        labels = dataset.targets
        self.assertEqual(sum(labels), 4)
        self.assertEqual(len(labels) - sum(labels), 8)

        sk_dataset = BinaryLesionDataset(
            os.path.join(self.data_root, "train"),
            transform=None,
            positive_classes=("seborrheic_keratosis",),
        )
        self.assertEqual(sum(sk_dataset.targets), 4)

    def test_train_loader_is_not_cached_tensor_dataset(self):
        runner = ExperimentRunner(
            loss_fn=nn.BCEWithLogitsLoss(),
            train_dir=os.path.join(self.data_root, "train"),
            valid_dir=os.path.join(self.data_root, "valid"),
            test_dir=os.path.join(self.data_root, "test"),
            train_transforms=get_train_transforms(64),
            eval_transforms=get_eval_transforms(64),
            positive_classes=("melanoma",),
            batch_size=4,
            num_workers=0,
            device=torch.device("cpu"),
            use_tta=False,
        )
        loader = runner.get_train_loader()
        self.assertNotIsInstance(loader.dataset, TensorDataset)
        self.assertIsInstance(loader.dataset, BinaryLesionDataset)

    def test_eval_transforms_are_deterministic(self):
        train_repr = repr(get_train_transforms(224))
        eval_repr = repr(get_eval_transforms(224))
        self.assertIn("ColorJitter", train_repr)
        self.assertIn("RandomAffine", train_repr)
        self.assertIn("RandomHorizontalFlip", train_repr)
        self.assertNotIn("ColorJitter", eval_repr)
        self.assertNotIn("RandomHorizontalFlip", eval_repr)

    def test_short_training_and_tta_inference(self):
        device = torch.device("cpu")
        model = get_model(arch="resnet18", pretrained=False, dropout=0.1, device=device)
        runner = ExperimentRunner(
            loss_fn=nn.BCEWithLogitsLoss(),
            train_dir=os.path.join(self.data_root, "train"),
            valid_dir=os.path.join(self.data_root, "valid"),
            test_dir=os.path.join(self.data_root, "test"),
            train_transforms=get_train_transforms(64),
            eval_transforms=get_eval_transforms(64),
            positive_classes=("melanoma",),
            batch_size=4,
            num_workers=0,
            device=device,
            use_tta=True,
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )
        model = runner.train(model, optimizer, num_epochs=1, patience=2)
        paths, labels, probs = runner.test(model)
        self.assertEqual(len(paths), 6)
        self.assertEqual(len(labels), 6)
        self.assertEqual(len(probs), 6)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in probs))
        self.assertTrue(any("data/test/" in path for path in paths))

    def test_write_results_csv_follows_ground_truth_order(self):
        order_csv = os.path.join(self.tmpdir, "ground_truth.csv")
        with open(order_csv, "w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Id", "task_1", "task_2"])
            writer.writerow(["data/test/b.jpg", 0, 1])
            writer.writerow(["data/test/a.jpg", 1, 0])

        out_csv = os.path.join(self.tmpdir, "preds.csv")
        write_results_csv(
            out_csv,
            ["data/test/a.jpg", "data/test/b.jpg"],
            [0.9, 0.1],
            [0.2, 0.8],
            order_csv=order_csv,
        )
        with open(out_csv) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["Id"], "data/test/b.jpg")
        self.assertEqual(rows[1]["Id"], "data/test/a.jpg")
        self.assertAlmostEqual(float(rows[0]["task_1"]), 0.1)

    def test_normalize_image_id(self):
        self.assertEqual(
            normalize_image_id("/tmp/run/data/test/melanoma/ISIC_1.jpg"),
            "data/test/melanoma/ISIC_1.jpg",
        )


class BaselineScoreTests(unittest.TestCase):
    def test_checked_in_predictions_match_published_auc(self):
        import pandas as pd
        from sklearn.metrics import roc_auc_score

        truth = pd.read_csv("ground_truth.csv")
        pred = pd.read_csv("model_results.csv")
        merged = truth.merge(pred, on="Id", suffixes=("_t", "_p"))
        category_1 = roc_auc_score(merged["task_1_t"], merged["task_1_p"])
        category_2 = roc_auc_score(merged["task_2_t"], merged["task_2_p"])
        self.assertAlmostEqual(category_1, 0.827, delta=0.01)
        self.assertAlmostEqual(category_2, 0.922, delta=0.01)


if __name__ == "__main__":
    unittest.main()
