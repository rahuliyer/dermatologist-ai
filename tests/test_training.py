import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from experiment_runner import ExperimentRunner


class ExperimentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        colors = {
            "melanoma": (180, 20, 20),
            "nevus": (20, 180, 20),
            "seborrheic_keratosis": (20, 20, 180),
        }
        for split in ("train", "valid", "test"):
            for class_name, color in colors.items():
                class_dir = self.data_dir / split / class_name
                class_dir.mkdir(parents=True)
                Image.new("RGB", (12, 12), color).save(class_dir / f"{split}.png")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_binary_training_and_path_preserving_inference(self):
        image_transforms = [transforms.Resize((8, 8)), transforms.ToTensor()]
        runner = ExperimentRunner(
            nn.BCEWithLogitsLoss(),
            self.data_dir,
            self.data_dir,
            image_transforms,
            image_transforms,
            positive_class="melanoma",
            batch_size=2,
            workers=0,
            device=torch.device("cpu"),
            amp=False,
        )

        train_loader = runner.get_train_loader()
        self.assertEqual(sorted(set(train_loader.dataset.binary_targets)), [0, 1])

        model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        runner.train_loader = train_loader
        runner.train(model, optimizer, num_epochs=1, patience=0)
        paths, labels, probabilities = runner.test(model)

        self.assertEqual(len(paths), 3)
        self.assertEqual(sum(labels), 1)
        self.assertEqual(len(probabilities), 3)
        self.assertTrue(all(Path(path).is_file() for path in paths))
        self.assertTrue(all(0.0 <= probability <= 1.0 for probability in probabilities))


if __name__ == "__main__":
    unittest.main()
