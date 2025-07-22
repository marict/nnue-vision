from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import wandb
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, Precision, Recall


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

        # Metrics for comprehensive logging
        self.train_acc = Accuracy(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
        )
        self.val_acc = Accuracy(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
        )
        self.test_acc = Accuracy(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
        )

        self.val_precision = Precision(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
            average="macro",
        )
        self.val_recall = Recall(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
            average="macro",
        )
        self.val_f1 = F1Score(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
            average="macro",
        )

        self.test_precision = Precision(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
            average="macro",
        )
        self.test_recall = Recall(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
            average="macro",
        )
        self.test_f1 = F1Score(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
            average="macro",
        )

        self.val_confusion_matrix = ConfusionMatrix(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
        )
        self.test_confusion_matrix = ConfusionMatrix(
            task="binary" if params.num_classes == 2 else "multiclass",
            num_classes=params.num_classes,
        )

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
        acc = self.train_acc(preds, labels)

        # Log basic metrics
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        # Log additional training insights
        self.log("train_step_loss", loss, on_step=True, on_epoch=False)

        # Log class distribution in batch (helpful for debugging data imbalance)
        class_counts = torch.bincount(labels, minlength=self.params.num_classes)
        for i, count in enumerate(class_counts):
            self.log(
                f"train_batch/class_{i}_count",
                count.float(),
                on_step=True,
                on_epoch=False,
            )

        # Log prediction confidence statistics
        probs = torch.softmax(logits, dim=1)
        max_probs = torch.max(probs, dim=1)[0]
        self.log(
            "train_batch/avg_confidence", max_probs.mean(), on_step=True, on_epoch=False
        )
        self.log(
            "train_batch/min_confidence", max_probs.min(), on_step=True, on_epoch=False
        )
        self.log(
            "train_batch/max_confidence", max_probs.max(), on_step=True, on_epoch=False
        )

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Calculate predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_acc(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        self.val_f1(preds, labels)
        self.val_confusion_matrix(preds, labels)

        # Log basic metrics
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        # Log additional validation metrics
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Log confusion matrix and per-class metrics at end of validation epoch."""
        # Get confusion matrix
        cm = self.val_confusion_matrix.compute()

        # Log confusion matrix as wandb image
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm.cpu().numpy(),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Person", "Person"],
                yticklabels=["No Person", "Person"],
            )
            plt.title("Validation Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Log to wandb
            if wandb.run is not None:
                wandb.log({"validation/confusion_matrix": wandb.Image(plt)})

            plt.close()
        except Exception as e:
            print(f"Warning: Could not log confusion matrix: {e}")

        # Reset metrics
        self.val_confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)

        # Calculate predictions
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.test_acc(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.test_f1(preds, labels)
        self.test_confusion_matrix(preds, labels)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """Log final test confusion matrix and detailed metrics."""
        # Get confusion matrix
        cm = self.test_confusion_matrix.compute()

        # Log confusion matrix as wandb image
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm.cpu().numpy(),
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Person", "Person"],
                yticklabels=["No Person", "Person"],
            )
            plt.title("Test Confusion Matrix")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")

            # Log to wandb
            if wandb.run is not None:
                wandb.log({"test/confusion_matrix": wandb.Image(plt)})

            plt.close()
        except Exception as e:
            print(f"Warning: Could not log test confusion matrix: {e}")

        # Calculate and log per-class metrics
        cm_np = cm.cpu().numpy()
        if cm_np.shape[0] == 2:  # Binary classification
            tn, fp, fn, tp = cm_np.ravel()

            # Per-class precision and recall
            precision_class_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
            precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

            if wandb.run is not None:
                wandb.log(
                    {
                        "test/precision_no_person": precision_class_0,
                        "test/precision_person": precision_class_1,
                        "test/recall_no_person": recall_class_0,
                        "test/recall_person": recall_class_1,
                        "test/true_negatives": tn,
                        "test/false_positives": fp,
                        "test/false_negatives": fn,
                        "test/true_positives": tp,
                    }
                )

        # Reset metrics
        self.test_confusion_matrix.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss",
            },
        }
