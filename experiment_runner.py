import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset

import sys
import os

import yaml

import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score

from copy import deepcopy

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.imgs[index]

        img, target = super().__getitem__(index)

        return path, img, target

class TestDataset(Dataset):
    def __init__(self, paths, inputs, labels):
        super().__init__()

        self.paths = paths
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return (self.paths[index], self.inputs[index], self.labels[index])

class ExperimentRunner():
    def __init__(self,
            loss_fn,
            train_dataset_root,
            test_dataset_root,
            train_transforms,
            test_transforms,
            batch_size=64,
            savefile='best_model.pt'):
        self.loss_fn = loss_fn
        self.train_dataset_root = train_dataset_root
        self.test_dataset_root = test_dataset_root
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size

        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

        self.best_valid_loss = np.Inf

        self.savefile = savefile

    def getDataLoader(self, path, transform):
        ds = datasets.ImageFolder(
                 root = path,
                 transform = transforms.Compose(
                     transform
                 )
             )

        dl = DataLoader(ds, self.batch_size)

        inputs = torch.tensor([])
        labels = torch.LongTensor([])

        for i, l in dl:
            inputs = torch.cat((inputs, i))
            labels = torch.cat((labels, l))

        tds = TensorDataset(inputs, labels)

        return DataLoader(tds,
                          self.batch_size,
                          shuffle=True)

    def getTrainLoader(self):
        print("Setting up train dataloader")
        return self.getDataLoader(self.train_dataset_root + '/train', self.train_transforms)

    def getValidLoader(self):
        print("Setting up valid dataloader")
        return self.getDataLoader(self.train_dataset_root + '/valid', self.train_transforms)

    def getTestLoader(self):
        print("Setting up test dataloader")
        ds = ImageFolderWithPaths(
                root = self.test_dataset_root + '/test',
                transform = transforms.Compose(
                    self.test_transforms
                )
            )

        dl = DataLoader(ds, self.batch_size)

        inputs = torch.tensor([])
        labels = torch.LongTensor([])
        paths = []
        for p, i, l in dl:
            paths.extend(p)
            inputs = torch.cat((inputs, i))
            labels = torch.cat((labels, l))

        tds = TestDataset(paths, inputs, labels)

        return DataLoader(tds,
                          self.batch_size,
                          shuffle=True)

    def train(self, model, optimizer, num_epochs, print_every=10):
        cur_best_valid_loss = np.Inf
        cur_best_model = None

        if self.train_loader == None:
            self.train_loader = self.getTrainLoader()

        if self.valid_loader == None:
            self.valid_loader = self.getValidLoader()

        print("Starting training...")
        model.train()
        for epoch_nr in range(num_epochs):
            train_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                optimizer.zero_grad()

                inputs, labels = inputs.cuda(), labels.cuda()
                preds = model(inputs)

                loss = self.loss_fn(preds.squeeze(), labels.float())

                loss.backward()
                optimizer.step()

                train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))

                if batch_idx % print_every == 0:
                    valid_loss = 0.0
                    model.eval()
                    for valid_batch_idx, (v_inputs, v_labels) in enumerate(self.valid_loader):
                        v_inputs, v_labels = v_inputs.cuda(), v_labels.cuda()

                        preds = model(v_inputs)

                        loss = self.loss_fn(preds.squeeze(), v_labels.float())

                        valid_loss = valid_loss + ((1 / (valid_batch_idx + 1)) * (loss.data - valid_loss))

                    print("Epoch: {}, batch: {} - Train loss: {}, validation loss: {}".format(epoch_nr, batch_idx, train_loss, valid_loss))

                    if valid_loss < cur_best_valid_loss:
                        cur_best_valid_loss = valid_loss

                        print("Lowest validation score seen so far. Saving model...")
                        cur_best_model = deepcopy(model.state_dict())

                    model.train()

        if cur_best_valid_loss < self.best_valid_loss:
            self.best_valid_loss = cur_best_valid_loss
            print("Best model so far; saving...")
            torch.save(cur_best_model, self.savefile)

        model.load_state_dict(cur_best_model)

        return model

    def test(self, model):
#        self.model.load_state_dict(torch.load('best_model.pt'))

        model.eval()

        test_paths = []
        labels = []
        probs = []

        if self.test_loader == None:
            self.test_loader = self.getTestLoader()

        for paths, inputs, targets in self.test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()

            preds = model(inputs)

            preds = preds.detach().cpu()

            test_paths.extend(paths)
            probs.extend(preds.numpy().squeeze())
            labels.extend(targets.cpu().numpy())

        return test_paths, labels, probs

    def accuracy(self, model):
        paths, labels, preds = self.test(model)

        return accuracy_score(labels, np.round(preds))

    def confusion_matrix(self, model):
        paths, labels, preds = self.test(model)

        return confusion_matrix(labels, np.round(preds))

    def recall_score(self, model):
        paths, labels, preds = self.test(model)

        return recall_score(labels, np.round(preds))
