import torch

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torchvision import datasets
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from torch.utils.data import DataLoader

import sys
import os

import yaml

import numpy as np

from sklearn.metrics import accuracy_score

from experiment_runner import ExperimentRunner

from torchvision.models import resnet50

import PIL

def get_model(num_layers_to_train):
    model = resnet50(pretrained=True)
#    model.fc.in_features = 204800
    model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
            nn.Sigmoid()
    )

    for l in model.fc.children():
        if l.__class__.__name__ == 'Linear':
            nn.init.normal_(l.weight, mean=0, std=(1.0 / np.sqrt(l.in_features)))

    layers = [layer for layer in model.children()]
    num_untrained_layers = len(layers) - num_layers_to_train

    for i, layer in enumerate(layers):
        if i < num_untrained_layers:
            for param in layer.parameters():
                param.requires_grad = False
        else:
            for param in layer.parameters():
                param.requires_grad = True

    model = nn.DataParallel(model)
    model.cuda()

    return model

def load_image(path, transform):
    from PIL import Image

    img = Image.open(path)
    tf = transforms.Compose(transform)

    tensor = tf(img)

    return tensor

if __name__ == "__main__":
    loss_fn = nn.BCELoss()

    train_transforms = []
    train_transforms.append(transforms.Resize((1024, 768)))
    train_transforms.append(transforms.CenterCrop((512, 512)))
    train_transforms.append(transforms.Resize((224, 224)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomVerticalFlip())
    train_transforms.append(transforms.RandomRotation(20, resample=PIL.Image.BILINEAR))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    test_transforms = []
    test_transforms.append(transforms.Resize((1024, 768)))
    test_transforms.append(transforms.CenterCrop((512, 512)))
    test_transforms.append(transforms.Resize((224, 224)))
    test_transforms.append(transforms.ToTensor())
    test_transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    runner = ExperimentRunner( 
            loss_fn,
            'melanoma_dataset', 
            train_transforms,
            test_transforms,
            batch_size=64)

    ensemble_preds = np.array([])
    num_ensembles = 1
    for e in range(num_ensembles):
        for i in [9]:#[num_layers for num_layers in range(2, 10)]:
            for lr in [0.0001]:
                for j in [20]:
                    print("Model #{}: Training {} layers for {} epochs with lr={}...".format(
                            e,
                            i,
                            j,
                            lr
                        )
                    )

                    model = get_model(i)
                    optimizer = optim.SGD(model.module.parameters(), lr=lr, momentum=0.9)

                    model = runner.train(model, optimizer, j)
                    #print("Accuracy: {}".format(runner.test(model)))
                    paths, labels, probs = runner.test(model)

                    from sklearn.metrics import confusion_matrix
                    print("Confusion matrix:\n {}".format(confusion_matrix(labels, np.round(probs))))
                    from sklearn.metrics import recall_score
                    print("recall: {}".format(recall_score(labels, np.round(probs))))
                    from sklearn.metrics import precision_score
                    print("precision: {}".format(precision_score(labels, np.round(probs))))

                    from sklearn.metrics import roc_auc_score
                    print("roc auc score: {}".format(roc_auc_score(labels, np.round(probs))))
                    if len(ensemble_preds) == 0:
                        ensemble_preds = np.array(probs)
                    else:
                        ensemble_preds += np.array(probs)
    #                false_positives = torch.tensor([])
    #                false_negatives = torch.tensor([])
    #                for idx in range(len(probs)):
    #                    pred = np.round(probs[idx])
    #
    #                    if pred != labels[idx]:
    #                        img = load_image(paths[idx], transform)
    #                        if pred == 0 and labels[idx] == 1:
    #                            false_negatives = torch.cat((false_negatives, img.unsqueeze(dim=0)))
    #                        elif pred == 1 and labels[idx] == 0:
    #                            false_positives = torch.cat((false_positives, img.unsqueeze(dim=0)))
    #
                            #print("{}: {} {} {} {}".format(idx, paths[idx], np.round(probs[idx]), labels[idx], probs[idx]))
    #                save_image(make_grid(false_positives), 'false_positives.jpg')
    #                save_image(make_grid(false_negatives), 'false_negatives.jpg')
    ensemble_preds = ensemble_preds / num_ensembles

    print("Confusion matrix:\n {}".format(confusion_matrix(labels, np.round(ensemble_preds))))
    print("recall: {}".format(recall_score(labels, np.round(ensemble_preds))))
    print("precision: {}".format(precision_score(labels, np.round(ensemble_preds))))
    print("Ensemble roc: {}".format(roc_auc_score(labels, np.round(ensemble_preds))))
