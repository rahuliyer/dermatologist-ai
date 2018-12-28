import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader

import sys
import os

import yaml

class TestDataSet(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.imgs[index]

        img, target = super().__getitem__(index)

        return path, img, target

class BaseModel(nn.Module):
    def __init__(self, dataset_root, config):
        super().__init__()

        self.default_epochs = 100
        self.default_device_id = 0
        self.default_batch_size = 64
        self.default_learning_rate = 0.01

        self.config = config
        self.dataset_root = dataset_root

        self.model_config = self.config.get('model')
        self.model_name = self.model_config.get('name')
        self.preTrained = self.model_config.get('preTrained', True)
        self.fineTune = self.model_config.get('fineTune', False)

        self.training_config = self.config.get('training')
        self.num_epochs = self.training_config.get('num_epochs', self.default_epochs)
        self.batch_size = self.training_config.get('batch_size', self.default_batch_size)
        self.learning_rate = self.training_config.get('learning_rate', self.default_learning_rate)
        self.parallel = self.training_config.get('parallel', False)
        self.device_id = self.training_config.get('device_id')

        if self.model_name == 'vgg16':
            from torchvision.models import vgg16
            self.model = vgg16(pretrained = self.preTrained)

            if self.fineTune == False:
                for param in self.model.parameters():
                    param.requires_grad = False

            self.model.classifier[6] = nn.Linear(self.model.classifier[6].in_features, 3)
        elif self.model_name == 'resnet152':
            from torchvision.models import resnet152
            self.model = resnet152(pretrained = self.preTrained)

            if self.fineTune == False:
                for param in self.model.parameters():
                    param.requires_grad = False

            self.model.fc = nn.Linear(self.model.fc.in_features, 3)

        if self.parallel == True and self.device_id != None:
            raise Exception("Both parallel and device specified")

        if self.parallel:
            print("Parallel set. Using {} GPUs".format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
        else:
            if self.device_id != None:
                print("Using device cuda:{}".format(self.device_id))
            else:
                print("No device set. Using {}".format(self.default_device))
                self.device_id = self.default_device_id

            torch.cuda.set_device(self.device_id)

        self.model.cuda()

        self.data_config = self.config['data']
        self.transforms = []
        transform_list = self.data_config.get('transforms', [])
        if 'resize' in transform_list:
            self.transforms.append(
                transforms.Resize(
                    (transform_list['resize']['width'], transform_list['resize']['width'])
                )
            )

        self.transforms.append(transforms.ToTensor())
        self.transforms.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )

        self.train_set = datasets.ImageFolder(
                                root = self.dataset_root + '/train',
                                transform = transforms.Compose(
                                    self.transforms
                                )
                            )

        self.valid_set = datasets.ImageFolder(
                                root = self.dataset_root + '/valid',
                                transform = transforms.Compose(
                                    self.transforms
                                )
                            )

        self.test_set = TestDataSet(
                                root = self.dataset_root + '/test',
                                transform = transforms.Compose(
                                    self.transforms
                                )
                            )

        self.train_loader = DataLoader(self.train_set,
                                       self.batch_size,
                                       shuffle=True)

        self.valid_loader = DataLoader(self.valid_set,
                                       self.batch_size,
                                       shuffle=True)

        self.test_loader = DataLoader(self.test_set,
                                       self.batch_size,
                                       shuffle=True)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr = self.learning_rate,
                                   momentum=0.9)

    def train(self):
        print_every = 1
        for i in range(self.num_epochs):
            self.model.train()

            total_train_loss = 0.0
            num_train_batches = 1
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()

                self.optimizer.zero_grad()

                preds = self.model(inputs)

                loss = self.loss_fn(preds, targets)
                loss.backward()

                self.optimizer.step()

                total_train_loss += loss.item()
                num_train_batches += 1

                if num_train_batches % print_every == 0:
                    self.model.eval()
                    total_valid_loss = 0.0
                    num_valid_batches = 1
                    for inputs, targets in self.valid_loader:
                        inputs, targets = inputs.cuda(), targets.cuda()
                        preds = self.model(inputs)

                        loss = self.loss_fn(preds, targets)

                        total_valid_loss += loss.item()
                        num_valid_batches += 1

                    print(
                        "Epoch {}: Average train loss = {}; validation loss: {}".format(
                            i + 1,
                            total_train_loss / num_train_batches,
                            total_valid_loss / num_valid_batches)
                        )

                    num_train_batches = 1
                    total_train_loss = 0.0

    def test(self):
        self.model.eval()

        test_paths = []
        melanoma_probs = []
        sk_probs = []
        for paths, inputs, targets in self.test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            preds = F.softmax(self.model(inputs), dim=1)
            preds = preds.detach().cpu()

            test_paths.extend(paths)
            melanoma_probs.extend(preds[:, 0].numpy())
            sk_probs.extend(preds[:,2].numpy())

        return test_paths, melanoma_probs, sk_probs

    def write_results_csv(self, fname):
        import csv

        paths, mp, skp = self.test()

        with open(fname, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)

            csvwriter.writerow(['Id', 'task_1', 'task_2'])

            for i in range(len(paths)):
                csvwriter.writerow([paths[i], mp[i], skp[i]])

    @staticmethod
    def getResultFileName(experiment_file):
        filename = os.path.split(experiment_file)[-1]

        output_filename = filename.split('.')[0] + ".csv"

        return os.path.join('results', output_filename)

    @staticmethod
    def fromConfig(dataset, config_file):
        f = open(config_file, "r")

        config = yaml.load(f.read())

        return BaseModel(dataset, config)

def usage(cmd):
    print("Usage: {} <dataset folder> <experiment config>".format(cmd))
    sys.exit(-1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage(sys.argv[0])

model = BaseModel.fromConfig(sys.argv[1], sys.argv[2])
model.train()
#model.test()
model.write_results_csv(BaseModel.getResultFileName(sys.argv[2]))

