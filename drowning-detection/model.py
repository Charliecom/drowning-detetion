import torch
import lightning as L
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils import IterableSimpleNamespace
from torchvision.ops import box_convert, nms
from torchmetrics.detection import MeanAveragePrecision


class MyModel(L.LightningModule):
    def __init__(
        self,
        model_path: str = "/home/cqss0/ML/drowning-detection/data/yolov5nu.pt",
        num_classes: int = 6,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        warmup_epochs: int = 3,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs

        yolo = YOLO(model_path)
        self.model = yolo.model  # <class 'ultralytics.nn.tasks.DetectionModel'>

        old_detect = self.model.model[-1]
        new_detect = Detect(nc=num_classes, ch=(64, 128, 256))

        new_detect.f = old_detect.f
        new_detect.i = old_detect.i
        new_detect.stride = old_detect.stride

        self.model.model[-1] = new_detect
        new_detect.bias_init()

        self.model.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

        self.criterion = None

        self.val_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")

    def training_step(self, batch, batch_idx=0):
        if self.criterion is None:
            self.criterion = v8DetectionLoss(self.model)

        imgs = batch["image"]
        list_boxes = batch["boxes"]
        list_labels = batch["labels"]

        batch_idx_list = []
        for i, b in enumerate(list_boxes):
            batch_idx_list.append(
                torch.full((b.shape[0],), i, device=self.device, dtype=torch.float32)
            )

        yolo_batch = {
            "batch_idx": torch.cat(batch_idx_list).to(device="cuda:0"),
            "cls": torch.cat(list_labels).to(torch.float32).to(device="cuda:0"),
            "bboxes": torch.cat(list_boxes).to(torch.float32).to(device="cuda:0"),
            "img": imgs.to(device="cuda:0"),
        }

        preds = self.model(imgs)
        loss, loss_items = self.criterion(preds, yolo_batch)

        self.log(
            "train/box_loss",
            loss[0],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=imgs.shape[0],
        )
        self.log(
            "train/cls_loss",
            loss[1],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=imgs.shape[0],
        )
        self.log(
            "train/dfl_loss",
            loss[2],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=imgs.shape[0],
        )
        self.log(
            "train/loss",
            loss.mean(),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=imgs.shape[0],
        )
        return loss.mean()

    def _postprocess_preds(self, raw_preds, img_shape, conf_thres=0.001, iou_thres=0.6):
        """Декодирует координаты и применяет Batched NMS к выходу YOLO."""
        preds = raw_preds[0]

        batch_size = preds.shape[0]
        processed_preds = []

        for i in range(batch_size):
            p = preds[i].permute(1, 0)

            bboxes = p[:, :4]
            scores = p[:, 4:]

            max_scores, class_ids = torch.max(scores, dim=1)

            keep_idx = max_scores > conf_thres
            if not keep_idx.any():
                processed_preds.append(
                    {
                        "boxes": torch.empty((0, 4), device=self.device),
                        "scores": torch.empty((0,), device=self.device),
                        "labels": torch.empty(
                            (0,), dtype=torch.int64, device=self.device
                        ),
                    }
                )
                continue

            bboxes = bboxes[keep_idx]
            max_scores = max_scores[keep_idx]
            class_ids = class_ids[keep_idx]

            bboxes = box_convert(bboxes, in_fmt="cxcywh", out_fmt="xyxy")

            _, _, h, w = img_shape
            bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, w)
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, h)

            offsets = class_ids.to(bboxes.dtype) * (max(h, w) + 1)
            nms_idx = nms(bboxes + offsets.unsqueeze(1), max_scores, iou_thres)

            processed_preds.append(
                {
                    "boxes": bboxes[nms_idx],
                    "scores": max_scores[nms_idx],
                    "labels": class_ids[nms_idx].to(torch.int64),
                }
            )

        return processed_preds

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        imgs = batch["image"]
        list_boxes = batch["boxes"]
        list_labels = batch["labels"]

        self.model.eval()
        with torch.no_grad():
            raw_preds = self.model(imgs)

        predictions = self._postprocess_preds(
            raw_preds, img_shape=imgs.shape, conf_thres=0.001, iou_thres=0.6
        )

        _, _, h, w = imgs.shape

        ground_truths = []
        for boxes, labels in zip(list_boxes, list_labels):
            boxes = boxes.to(self.device)

            boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

            multiplier = torch.tensor(
                [w, h, w, h], dtype=torch.float32, device=self.device
            )
            boxes_absolute = boxes_xyxy * multiplier

            ground_truths.append(
                {
                    "boxes": boxes_absolute,
                    "labels": labels.to(torch.int64).to(self.device),
                }
            )

        self.val_map.update(preds=predictions, target=ground_truths)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_map.compute()

        mAP_50_95 = metrics["map"].item()
        mAP_50 = metrics["map_50"].item()

        self.log(
            "val/mAP_50_95", mAP_50_95, on_epoch=True, prog_bar=True, sync_dist=True
        )
        self.log("val/mAP_50", mAP_50, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_map.reset()

    def predict_step(self, batch: dict, batch_idx: int = 0) -> list[dict]:
        """Шаг инференса. Возвращает NMS-предсказания с абсолютными координатами боксов."""
        imgs = batch["image"]

        raw_preds = self.model(imgs)

        predictions = self._postprocess_preds(
            raw_preds, img_shape=imgs.shape, conf_thres=0.25, iou_thres=0.45
        )

        return predictions

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
