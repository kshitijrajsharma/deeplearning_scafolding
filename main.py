"""UNet segmentation of tall/flat vegetation from RGB+NDVI(+shade) tiles."""

import argparse
import glob
import os

import geopandas as gpd
import kornia.augmentation as K
import lightning as L
import matplotlib
import numpy as np
import pandas as pd
import rasterio.features
import segmentation_models_pytorch as smp
import torch
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.transform import from_origin
from shapely.geometry import shape
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
PATIENCE = 10  # early-stopping patience on val_f1
RESOLUTION = 0.08
TILE_CRS = "EPSG:25832"  # ETRS89 / UTM 32N;
LABEL_CMAP = ListedColormap(["#2b2b2b", "#2ca02c", "#bcbd22"])  # bg, tall, flat


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


class VegDataModule(L.LightningDataModule):
    """Train/val/test loaders over the split folders; `use_shade` sets input channels."""

    def __init__(self, data_dir, use_shade=True, batch_size=16, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.use_shade = use_shade
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channels = 5 if use_shade else 4

    def _loader(self, split, shuffle):
        dataset = NpyDataset(os.path.join(self.data_dir, split), self.use_shade)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        return self._loader("train", shuffle=True)

    def val_dataloader(self):
        return self._loader("val", shuffle=False)

    def test_dataloader(self):
        return self._loader("test", shuffle=False)


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
        self.train_f1 = MulticlassF1Score(num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes, average="macro")
        self.val_iou = MulticlassJaccardIndex(num_classes, average="macro")
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
        self.val_iou.update(logits.argmax(1), label)
        self.log("val_loss", self.loss(logits, label), prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute().item()  # ty: ignore[missing-argument]
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_iou", self.val_iou.compute().item(), prog_bar=True)  # ty: ignore[missing-argument]
        self.val_history.append(f1)
        self.val_f1.reset()
        self.val_iou.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def plot_distribution(data_dir, path, splits=("train", "val", "test")):
    def proportions(split):
        counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        for file in glob.glob(os.path.join(data_dir, split, "*.npy")):
            label = np.load(file, mmap_mode="r")[5].astype(int).ravel()
            counts += np.bincount(label, minlength=len(CLASS_NAMES))
        return counts / counts.sum()

    dist = pd.DataFrame({s: proportions(s) for s in splits}, index=CLASS_NAMES)
    dist.T.plot.bar(
        title="Class proportion per split", ylabel="pixel proportion", rot=0
    )
    plt.savefig(path, bbox_inches="tight")


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
            ax.imshow(data, cmap=LABEL_CMAP, vmin=0, vmax=len(CLASS_NAMES) - 1)
            ax.set_title(title)
            ax.axis("off")
    handles = [
        Patch(color=LABEL_CMAP(i), label=name) for i, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASS_NAMES))
    fig.savefig(path, bbox_inches="tight")


def vectorize_predictions(model, dataset, out_dir, n_tiles=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    records = []
    for file in dataset.files[:n_tiles]:
        tile = np.load(file).astype(np.float32)
        image = (
            torch.from_numpy(tile[: dataset.channels] / 255.0).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            pred = model(image).argmax(1)[0].cpu().numpy().astype("int32")
        name = os.path.splitext(os.path.basename(file))[0]
        east, north = (int(v) for v in name.split("_")[1:3])
        transform = from_origin(
            east, north + pred.shape[0] * RESOLUTION, RESOLUTION, RESOLUTION
        )
        records += [
            {"geometry": shape(geom), "class": CLASS_NAMES[int(value)], "tile": name}
            for geom, value in rasterio.features.shapes(pred, transform=transform)
            if value != 0
        ]
    gdf = gpd.GeoDataFrame(records, crs=TILE_CRS)
    gdf.to_file(os.path.join(out_dir, "predictions.geojson"), driver="GeoJSON")


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
    plot_distribution(args.data, os.path.join(args.out, "class_distribution.png"))
    print("use shade", args.use_shade, "use aug", args.use_aug)
    data = VegDataModule(args.data, args.use_shade, args.batch_size, args.num_workers)
    model = SegModel(in_channels=data.in_channels, lr=args.lr, use_aug=args.use_aug)
    early_stop = L.pytorch.callbacks.EarlyStopping(
        monitor="val_f1", mode="max", patience=PATIENCE
    )
    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        monitor="val_f1", mode="max", dirpath=args.out, filename="best"
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[early_stop, checkpoint],
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    )
    trainer.fit(model, datamodule=data)
    optimizer = trainer.optimizers[0]
    print(f"optimizer lr: {optimizer.param_groups[0]['lr']}")

    plot_history(model, os.path.join(args.out, "f1_curve.png"))
    best = SegModel.load_from_checkpoint(checkpoint.best_model_path)
    test_loader = data.test_dataloader()
    evaluate(best, test_loader, args.out)
    plot_predictions(best, test_loader, os.path.join(args.out, "predictions.png"))
    vectorize_predictions(best, test_loader.dataset, args.out)


if __name__ == "__main__":
    main()
