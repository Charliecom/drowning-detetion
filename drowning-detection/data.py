import torch
import lightning as L
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from myutils import build_transforms


def idx_to_name(idx):
    dct = {
        0: "human",
        1: "wind/sup-board",
        2: "boat",
        3: "bouy",
        4: "sailboat",
        5: "kayak",
    }
    return dct[idx]


def name_to_idx(name):
    dct = {
        "human": 0,
        "wind/sup-board": 1,
        "boat": 2,
        "bouy": 3,
        "sailboat": 4,
        "kayak": 5,
    }
    return dct[name]


class AFODataset(Dataset):
    """AFO detection dataset loaded from local files."""

    def __init__(
        self,
        dir_path: str | Path,
        split: str,
        height: int = 640,
        width: int = 640,
    ) -> None:
        super().__init__()
        self.transforms = build_transforms(split, height, width)
        pth = Path(dir_path)
        img_path = pth / "images"
        split_file_path = pth / (split + ".txt")
        detections_path = pth / "6categories"
        if (
            not img_path.exists()
            or not split_file_path.exists()
            or not detections_path.exists()
        ):
            raise FileNotFoundError()

        detections_set = dict()
        for file in detections_path.rglob("*.txt"):
            stem = file.stem
            detections_set[stem] = file

        with open(split_file_path, "r", encoding="utf-8") as f:
            list_of_files = f.read().splitlines()

        self.samples = list()
        for file in list_of_files:
            file_path = img_path / file
            if not file_path.exists():
                raise FileNotFoundError(f"img({file_path.stem}) is not found")
            self.samples.append((str(file_path), detections_set[file[:-4]]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, detection_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        boxes = []
        labels = []
        with open(detection_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                boxes.append([x, y, w, h])
                labels.append(cls_id)

        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)

        transformed = self.transforms(
            image=image,
            bboxes=boxes,
            class_labels=labels,
        )

        return {
            "image": transformed["image"],
            "boxes": torch.tensor(transformed["bboxes"], dtype=torch.float32),
            "labels": torch.tensor(transformed["class_labels"], dtype=torch.long),
        }


class AFODataModule(L.LightningDataModule):
    """LightningDataModule wrapping the AFO detection dataset.

    Args:
        data_root: Path to the root data directory (e.g. ``AFO/PART_1/PART_1``).
        batch_size: Samples per batch.
        num_workers: DataLoader worker processes.
        height: Resize height in pixels.
        width: Resize width in pixels.
    """

    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        num_workers: int = 4,
        height: int = 640,
        width: int = 640,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.height = height
        self.width = width
        self.split_files = {
            "train": "train",
            "val": "validation",
            "test": "test",
        }
        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            self.train_ds = AFODataset(
                dir_path=self.data_root,
                split="train",
                height=self.height,
                width=self.width,
            )
            self.val_ds = AFODataset(
                dir_path=self.data_root,
                split="validation",
                height=self.height,
                width=self.width,
            )

        if stage in ("test", None):
            self.test_ds = AFODataset(
                dir_path=self.data_root,
                split="test",
                height=self.height,
                width=self.width,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[dict]) -> dict:
        """Кастомная collate-функция для батчей с разным числом боксов."""
        images = []
        boxes = []
        labels = []

        for sample in batch:
            images.append(sample["image"])
            boxes.append(sample["boxes"])
            labels.append(sample["labels"])

        return {
            "image": torch.stack(images, dim=0),
            "boxes": boxes,
            "labels": labels,
        }
