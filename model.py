from dataclasses import dataclass

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelParams:
    input_size: tuple = (96, 96)  # Visual Wake Words standard size
    num_classes: int = 2  # binary classification: person/no person
    learning_rate: float = 1e-3


class SimpleCNN(pl.LightningModule):
    """
    Simple CNN for binary image classification on Visual Wake Words dataset.

    Architecture:
    - 3 convolutional layers with batch norm and ReLU
    - Global average pooling
    - Dense layer for classification
    """

    def __init__(self, params=ModelParams()):
        super(SimpleCNN, self).__init__()
        self.params = params
        self.save_hyperparameters()

        # CNN layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.classifier = nn.Linear(128, params.num_classes)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # Classification
        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        self.log("test_loss", loss)
        self.log("test_acc", acc)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
