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

from experiment_runner import ExperimentRunner

from torchvision.models import resnet50

import PIL

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

def get_model(num_layers_to_train):
    model = resnet50(pretrained=True)

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

def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.Resize((1024, 768)))
    train_transforms.append(transforms.CenterCrop((512, 512)))
    train_transforms.append(transforms.Resize((224, 224)))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomVerticalFlip())
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    )

    return train_transforms

def get_test_transforms():
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

    return test_transforms

def train_model(dataset_dir, savefile):
    loss_fn = nn.BCELoss()
    num_layers_to_train = 9

    model = get_model(num_layers_to_train)

    runner = ExperimentRunner(
            loss_fn,
            dataset_dir,
            get_train_transforms(),
            get_test_transforms(),
            batch_size=64,
            savefile=savefile)

    num_models_to_train = 1
    lr = 0.00001
    num_epochs = 2
    for i in range(num_models_to_train):
        print("Model #{}: Training {} layers for {} epochs with lr={}...".format(
                i,
                num_layers_to_train,
                num_epochs,
                lr
            )
        )

        optimizer = optim.SGD(model.module.parameters(), lr=lr, momentum=0.9)

        model = runner.train(model, optimizer, num_epochs)

def get_predictions(test_dataset_dir, model_savefiles):
    runner = ExperimentRunner(
            None,
            test_dataset_dir,
            get_train_transforms(),
            get_test_transforms()
    )

    model = get_model(0)

    probs = {}
    for i, savefile in enumerate(model_savefiles):
        model.load_state_dict(torch.load(savefile))
        paths, labels, probs[i] = runner.test(model)

    return paths, labels, probs

def write_results_csv(fname, paths, m_probs, sk_probs):
    import csv

    with open(fname, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        csvwriter.writerow(['Id', 'task_1', 'task_2'])

        for i in range(len(paths)):
            csvwriter.writerow([paths[i], m_probs[i], sk_probs[i]])

if __name__ == "__main__":
    #train_model('melanoma_dataset', 'best_melanoma_model.pt')
    #train_model('sk_dataset', 'best_sk_model.pt')

    paths, labels, probs = get_predictions(
        'data',
        [
            'best_melanoma_model.pt',
            'best_sk_model.pt'
        ]
    )

write_results_csv('model_results.csv', paths, probs[0], probs[1])

'''
    from sklearn.metrics import accuracy_score
    print("Accuracy: {}".format(runner.test(model)))
    from sklearn.metrics import confusion_matrix
    print("Confusion matrix:\n {}".format(confusion_matrix(labels, np.round(probs))))
    from sklearn.metrics import recall_score
    print("recall: {}".format(recall_score(labels, np.round(probs))))
    from sklearn.metrics import precision_score
    print("precision: {}".format(precision_score(labels, np.round(probs))))

    from sklearn.metrics import roc_auc_score
    print("roc auc score: {}".format(roc_auc_score(labels, np.round(probs))))
'''
