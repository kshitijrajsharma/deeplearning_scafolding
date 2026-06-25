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
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    jaccard_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CLASS_NAMES = ["background", "tall", "flat"]  # label values 0, 1, 2
SEED = 42
PATIENCE = 10  # early-stopping patience on val_f1
RESOLUTION = 0.08
TILE_CRS = "EPSG:25832"  # ETRS89 / UTM 32N;
LABEL_CMAP = ListedColormap(["#2b2b2b", "#2ca02c", "#bcbd22"])  # bg, tall, flat
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def input_channels(use_ndvi, use_shade):
    """Tile channel indices: RGB always, plus NDVI (3) and shade (4) when enabled."""
    return [0, 1, 2] + ([3] if use_ndvi else []) + ([4] if use_shade else [])


def to_model_input(tile, channels):
    """Select channels, scale to [0, 1], ImageNet-normalise the RGB triplet."""
    image = torch.from_numpy(tile[channels] / 255.0)
    image[:3] = (image[:3] - IMAGENET_MEAN) / IMAGENET_STD
    return image


class NpyDataset(Dataset):
    """Tiles of shape (6, H, W): R, G, B, NDVI, shade, label."""

    def __init__(self, split_dir, use_ndvi=True, use_shade=True):
        self.files = sorted(glob.glob(os.path.join(split_dir, "*.npy")))
        self.channels = input_channels(use_ndvi, use_shade)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        tile = np.load(self.files[index]).astype(np.float32)
        image = to_model_input(tile, self.channels)
        label = torch.from_numpy(tile[5]).long()
        return image, label


class VegDataModule(L.LightningDataModule):
    """Train/val/test loaders over the split folders; `use_shade` sets input channels."""

    def __init__(
        self, data_dir, use_ndvi=True, use_shade=True, batch_size=16, num_workers=4
    ):
        super().__init__()
        self.data_dir = data_dir
        self.use_ndvi = use_ndvi
        self.use_shade = use_shade
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_channels = len(input_channels(use_ndvi, use_shade))

    def _loader(self, split, shuffle):
        dataset = NpyDataset(
            os.path.join(self.data_dir, split), self.use_ndvi, self.use_shade
        )
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
    def __init__(self, in_channels, num_classes=3, lr=3e-4, use_aug=True):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.use_aug = use_aug
        self.net = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
        self.ce = nn.CrossEntropyLoss()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomAffine(degrees=90.0, scale=(0.8, 1.2), translate=(0.1, 0.1), p=0.5),
            data_keys=["input", "mask"],
        )
        self.train_f1 = MulticlassF1Score(num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes, average="macro")
        self.train_acc = MulticlassAccuracy(num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes, average="micro")
        self.val_iou = MulticlassJaccardIndex(num_classes, average="macro")
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_history = []
        self.val_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self.train_loss_history = []
        self.val_loss_history = []

    def forward(self, x):
        return self.net(x)

    def _loss(self, logits, label):
        return self.ce(logits, label) + self.dice(logits, label)

    def training_step(self, batch, _):
        image, label = batch
        if self.use_aug:
            image, mask = self.aug(image, label.unsqueeze(1).float())
            label = mask.squeeze(1).long()
        logits = self(image)
        loss = self._loss(logits, label)
        self.train_f1.update(logits.argmax(1), label)
        self.train_acc.update(logits.argmax(1), label)
        self.train_loss.update(loss)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.train_history.append(self.train_f1.compute().item())  # ty: ignore[missing-argument]
        self.train_acc_history.append(self.train_acc.compute().item())  # ty: ignore[missing-argument]
        self.train_loss_history.append(self.train_loss.compute().item())  # ty: ignore[missing-argument]
        self.train_f1.reset()
        self.train_acc.reset()
        self.train_loss.reset()

    def validation_step(self, batch, _):
        image, label = batch
        logits = self(image)
        loss = self._loss(logits, label)
        self.val_f1.update(logits.argmax(1), label)
        self.val_acc.update(logits.argmax(1), label)
        self.val_iou.update(logits.argmax(1), label)
        self.val_loss.update(loss)
        self.log("val_loss", loss, prog_bar=True)

    def on_validation_epoch_end(self):
        f1 = self.val_f1.compute().item()  # ty: ignore[missing-argument]
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_iou", self.val_iou.compute().item(), prog_bar=True)  # ty: ignore[missing-argument]
        self.val_history.append(f1)
        self.val_acc_history.append(self.val_acc.compute().item())  # ty: ignore[missing-argument]
        self.val_loss_history.append(self.val_loss.compute().item())  # ty: ignore[missing-argument]
        self.val_f1.reset()
        self.val_acc.reset()
        self.val_iou.reset()
        self.val_loss.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }


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


def plot_curve(train_values, val_values, ylabel, path):
    epochs = range(1, len(train_values) + 1)
    plt.figure()
    plt.plot(epochs, train_values, marker="o", label=f"train {ylabel}")
    plt.plot(epochs, val_values, marker="o", label=f"val {ylabel}")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_samples(data_dir, path, n=3):
    files = sorted(glob.glob(os.path.join(data_dir, "train", "*.npy")))[:n]
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))
    for row, file in enumerate(files):
        tile = np.load(file).astype(np.float32)
        rgb = tile[:3].transpose(1, 2, 0) / 255.0
        panels = [
            (rgb, "RGB", {}),
            (tile[3], "NDVI", {"cmap": "viridis"}),
            (tile[4], "shade", {"cmap": "gray"}),
            (
                tile[5],
                "label",
                {"cmap": LABEL_CMAP, "vmin": 0, "vmax": len(CLASS_NAMES) - 1},
            ),
        ]
        for ax, (data, title, kw) in zip(axes[row], panels):
            ax.imshow(data, **kw)
            ax.set_title(title)
            ax.axis("off")
    handles = [
        Patch(color=LABEL_CMAP(i), label=name) for i, name in enumerate(CLASS_NAMES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASS_NAMES))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


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
    ConfusionMatrixDisplay.from_predictions(
        targets, preds, labels=labels, display_labels=CLASS_NAMES, normalize="true"
    )
    plt.title("Test confusion matrix (row-normalised)")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), bbox_inches="tight")
    plt.close()


def plot_predictions(model, test_loader, path, n=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    images, labels = next(iter(test_loader))
    with torch.no_grad():
        preds = model(images.to(device)).argmax(1).cpu()
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    for row in range(n):
        rgb = (images[row, :3] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        rgb = rgb.permute(1, 2, 0).numpy()
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


def adjacent_tiles(files, n_tiles, step=20):
    """Largest spatially contiguous block of up to n_tiles tiles (20 m neighbours)."""
    coords = {}
    for file in files:
        east, north = (
            int(v) for v in os.path.splitext(os.path.basename(file))[0].split("_")[1:3]
        )
        coords[(east, north)] = file
    block = []
    for seed in coords:
        seen, queue = [seed], [seed]
        while queue and len(seen) < n_tiles:
            east, north = queue.pop(0)
            for nb in (
                (east + step, north),
                (east - step, north),
                (east, north + step),
                (east, north - step),
            ):
                if nb in coords and nb not in seen:
                    seen.append(nb)
                    queue.append(nb)
        if len(seen) > len(block):
            block = seen
        if len(block) >= n_tiles:
            break
    return [coords[c] for c in block[:n_tiles]]


def vectorize_predictions(model, files, channels, out_dir, n_tiles=6):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    records = []
    for file in adjacent_tiles(files, n_tiles):
        tile = np.load(file).astype(np.float32)
        image = to_model_input(tile, channels).unsqueeze(0).to(device)
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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-ndvi", action="store_true")
    parser.add_argument("--use-shade", action="store_true")
    parser.add_argument("--use-aug", action="store_true")
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()

    L.seed_everything(SEED, workers=True)
    torch.set_float32_matmul_precision("high")
    os.makedirs(args.out, exist_ok=True)
    plot_distribution(args.data, os.path.join(args.out, "class_distribution.png"))
    plot_samples(args.data, os.path.join(args.out, "samples.png"))
    print(
        "use ndvi", args.use_ndvi, "use shade", args.use_shade, "use aug", args.use_aug
    )
    data = VegDataModule(
        args.data, args.use_ndvi, args.use_shade, args.batch_size, args.num_workers
    )
    print("input channels:", data.in_channels)
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

    plot_curve(
        model.train_history,
        model.val_history,
        "macro F1",
        os.path.join(args.out, "f1_curve.png"),
    )
    plot_curve(
        model.train_acc_history,
        model.val_acc_history,
        "pixel accuracy",
        os.path.join(args.out, "accuracy_curve.png"),
    )
    plot_curve(
        model.train_loss_history,
        model.val_loss_history,
        "loss",
        os.path.join(args.out, "loss_curve.png"),
    )
    best = SegModel.load_from_checkpoint(checkpoint.best_model_path)
    test_loader = data.test_dataloader()
    evaluate(best, test_loader, args.out)
    plot_predictions(best, test_loader, os.path.join(args.out, "predictions.png"))
    all_tiles = sorted(glob.glob(os.path.join(args.data, "*", "*.npy")))
    vectorize_predictions(
        best, all_tiles, input_channels(args.use_ndvi, args.use_shade), args.out
    )


if __name__ == "__main__":
    main()
