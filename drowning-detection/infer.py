import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from tqdm import tqdm
from myutils import build_transforms, pull_dvc_data

from model import MyModel

log = logging.getLogger(__name__)

_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def draw_predictions_on_image(
    image: Image.Image, prediction: dict, scale_w: float, scale_h: float
) -> None:
    """Принимает PIL-картинку, словарь предсказания и коэффициенты масштабирования.

    Рисует bboxes, скорректированные под оригинальный размер картинки.
    """
    draw = ImageDraw.Draw(image)

    boxes = prediction["boxes"]

    # Переводим в numpy для удобных математических операций
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = box

        # Масштабируем координаты обратно к оригинальному разрешению
        orig_x1 = x1 * scale_w
        orig_y1 = y1 * scale_h
        orig_x2 = x2 * scale_w
        orig_y2 = y2 * scale_h

        # Рисуем прямоугольник (красный цвет, толщина линии 3 пикселя)
        draw.rectangle([orig_x1, orig_y1, orig_x2, orig_y2], outline="red", width=3)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    pull_dvc_data()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.inference.device == "auto"
        else torch.device(cfg.inference.device)
    )
    log.info("Using device: %s", device)
    log.info("Loading checkpoint: %s", cfg.inference.checkpoint)

    # Загружаем вашу модель из чекпоинта
    model = MyModel.load_from_checkpoint(cfg.inference.checkpoint, map_location=device)
    model.eval()

    # Сборка трансформаций для предобработки (из вашего пайплайна)
    transform = build_transforms("val", cfg.data.height, cfg.data.width)

    input_path = Path(cfg.inference.input)
    if input_path.is_file():
        img_paths = [input_path]
    elif input_path.is_dir():
        img_paths = [
            p
            for p in sorted(input_path.rglob("*"))
            if p.suffix.lower() in _IMG_EXTENSIONS
        ]
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")

    if not img_paths:
        log.warning("No images found at %s", input_path)
        return

    out_root = Path(cfg.inference.output_dir) if cfg.inference.output_dir else None
    if out_root:
        out_root.mkdir(parents=True, exist_ok=True)

    log.info("Running inference on %d image(s)", len(img_paths))
    for img_path in tqdm(img_paths, desc="Inference", unit="img"):
        pil_img = Image.open(img_path).convert("RGB")

        # Получаем оригинальные размеры до изменения размера в Albumentations
        orig_w, orig_h = pil_img.size

        # Рассчитываем коэффициенты масштабирования: оригинальный размер / размер входа модели
        scale_w = orig_w / cfg.data.width
        scale_h = orig_h / cfg.data.height

        image_np = np.array(pil_img, dtype=np.uint8)
        transformed = transform(image=image_np, bboxes=[], class_labels=[])
        tensor = transformed["image"].unsqueeze(0).float().to(device)

        batch = {"image": tensor}

        with torch.no_grad():
            predictions = model.predict_step(batch, batch_idx=0)
            pred = predictions[0]

        # Передаем коэффициенты масштабирования в функцию отрисовки
        draw_predictions_on_image(pil_img, pred, scale_w, scale_h)

        out_path = (
            out_root / (img_path.stem + "_pred.png")
            if out_root
            else img_path.with_name(img_path.stem + "_pred.png")
        )
        pil_img.save(out_path)

    log.info("Done. %d image(s) processed.", len(img_paths))


if __name__ == "__main__":
    run()
