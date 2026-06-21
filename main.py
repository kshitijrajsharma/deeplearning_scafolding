"""UNet segmentation of tall/flat vegetation from RGB+NDVI(+shade) tiles."""

import argparse
import glob
import os
from functools import partial

import kornia.augmentation as K
import lightning as L
import matplotlib
import numpy as np
import segmentation_models_pytorch as smp
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_NAMES = ["background", "tall", "flat"]  # label values 0, 1, 2


class NpyDataset(Dataset):
    """Tiles of shape (6, H, W): R, G, B, NDVI, shade, label."""

    def __init__(self, split_dir, use_shade=True):
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))
        self.channels = 5 if use_shade else 4

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        tile = np.load(self.files[index]).astype(np.float32)
        image = torch.from_numpy(tile[: self.channels] / 255.0)
        label = torch.from_numpy(tile[5]).long()
        return image, label


class SegModel(L.LightningModule):
    def __init__(self, in_channels, num_classes=3, lr=1e-3, use_aug=True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_aug = use_aug
        self.net = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
        self.loss = nn.CrossEntropyLoss()
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            data_keys=["input", "mask"],
        )
        self.train_f1 = MulticlassF1Score(num_classes, average="micro")
        self.val_f1 = MulticlassF1Score(num_classes, average="micro")
        self.val_miou = MulticlassJaccardIndex(num_classes, average="micro")
        self.train_history = []
        self.val_history = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, _):
        image, label = batch
        if self.use_aug:
            image, mask = self.aug(image, label.unsqueeze(1).float())
            label = mask.squeeze(1).long()
        logits = self(image)
        loss = self.loss(logits, label)
        self.train_f1.update(logits.argmax(1), label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_history.append(self.train_f1.compute().item())  # ty: ignore[missing-argument]
        self.train_f1.reset()

    def validation_step(self, batch, _):
        image, label = batch
        logits = self(image)
        self.val_f1.update(logits.argmax(1), label)
        self.val_miou.update(logits.argmax(1), label)
        self.log("val_loss", self.loss(logits, label), prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute().item()  # ty: ignore[missing-argument]
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_miou", self.val_miou.compute().item(), prog_bar=True)  # ty: ignore[missing-argument]
        self.val_history.append(f1)
        self.val_f1.reset()
        self.val_miou.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def loader(split_dir, use_shade, batch_size, num_workers, shuffle):
    dataset = NpyDataset(split_dir, use_shade)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def plot_history(model, path):
    epochs = range(1, len(model.train_history) + 1)
    plt.figure()
    plt.plot(epochs, model.train_history, marker="o", label="train macro F1")
    plt.plot(epochs, model.val_history, marker="o", label="val macro F1")
    plt.xlabel("epoch")
    plt.ylabel("macro F1")
    plt.legend()
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight")


def evaluate(model, test_loader, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    preds, targets = [], []
    with torch.no_grad():
        for image, label in test_loader:
            logits = model(image.to(device))
            preds.append(logits.argmax(1).cpu().numpy().ravel())
            targets.append(label.numpy().ravel())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(
        targets, preds, labels=labels, target_names=CLASS_NAMES, digits=4
    )
    matrix = confusion_matrix(targets, preds, labels=labels)
    per_class_iou = jaccard_score(
        targets, preds, labels=labels, average=None, zero_division=0
    )
    summary = (
        f"{report}\n"
        f"macro F1 : {f1_score(targets, preds, labels=labels, average='macro', zero_division=0):.4f}\n"
        f"mIoU     : {jaccard_score(targets, preds, labels=labels, average='macro', zero_division=0):.4f}\n"
        f"per-class IoU: {dict(zip(CLASS_NAMES, per_class_iou.round(4)))}\n\n"
        f"confusion matrix (rows=true, cols=pred) {CLASS_NAMES}:\n{matrix}\n"
    )
    print(summary)
    with open(os.path.join(out_dir, "test_report.txt"), "w") as handle:
        handle.write(summary)


def plot_predictions(model, test_loader, path, n=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row in range(n):
        rgb = images[row, :3].permute(1, 2, 0).numpy()
        panels = [(rgb, "image"), (labels[row], "mask"), (preds[row], "prediction")]
        for ax, (data, title) in zip(axes[row], panels):
            ax.imshow(data, vmin=0, vmax=len(CLASS_NAMES) - 1)
            ax.set_title(title)
            ax.axis("off")
    fig.savefig(path, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--use-shade", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-aug", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    in_channels = 5 if args.use_shade else 4

    make = partial(
        loader,
        use_shade=args.use_shade,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    train_loader = make(os.path.join(args.data, "train"), shuffle=True)
    val_loader = make(os.path.join(args.data, "val"), shuffle=False)
    test_loader = make(os.path.join(args.data, "test"), shuffle=False)

    model = SegModel(in_channels=in_channels, lr=args.lr, use_aug=args.use_aug)
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_f1", mode="max", dirpath=args.out, filename="best"
    )
    early_stop = L.pytorch.callbacks.EarlyStopping(
        monitor="val_f1", mode="max", patience=10
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[checkpoint, early_stop],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, val_loader)

    plot_history(model, os.path.join(args.out, "f1_curve.png"))
    best = SegModel.load_from_checkpoint(checkpoint.best_model_path)
    evaluate(best, test_loader, args.out)
    plot_predictions(best, test_loader, os.path.join(args.out, "predictions.png"))


if __name__ == "__main__":
    main()
