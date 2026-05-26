import torch
import numpy as np
import onnx
import onnxruntime as ort
from pathlib import Path
import hydra
from omegaconf import DictConfig
from model import MyModel


def convert_and_validate_yolo(
    checkpoint_path: str,
    output_onnx_path: str = "yolov5_custom.onnx",
    img_size: int = 640,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[Path, int, float]:

    onnx_path = Path(output_onnx_path)
    lightning_model = MyModel.load_from_checkpoint(checkpoint_path)

    pure_pytorch_model = lightning_model.model
    pure_pytorch_model.eval()

    device = torch.device("cpu")
    pure_pytorch_model.to(device)

    batch = torch.rand(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        torch_output_raw = pure_pytorch_model(batch)

    if isinstance(torch_output_raw, tuple):
        torch_output = torch_output_raw[0]
    else:
        torch_output = torch_output_raw

    dynamic_axes = {"images": {0: "batch_size"}, "output0": {0: "batch_size"}}

    torch.onnx.export(
        pure_pytorch_model,
        batch,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=dynamic_axes,
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    ort_session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    ort_outputs = ort_session.run(None, {"images": batch.detach().cpu().numpy()})

    ort_output = ort_outputs[0]

    torch_output_np = torch_output.detach().cpu().numpy()
    max_abs_diff = float(np.max(np.abs(torch_output_np - ort_output)))
    print(f"ℹ Максимальное абсолютное расхождение (Max ABS Diff): {max_abs_diff:.2e}")

    try:
        torch.testing.assert_close(
            torch.from_numpy(ort_output),
            torch.from_numpy(torch_output_np),
            atol=atol,
            rtol=rtol,
        )
        print("Результаты PyTorch и ONNX идентичны")
    except AssertionError as e:
        print(f"Расхождение выше допустимого rtol/atol:\n{e}")

    return onnx_path, len(onnx_model.graph.node), max_abs_diff


@hydra.main(config_name="config.yaml", config_path="../conf", version_base=None)
def main(cfg: DictConfig):
    convert_and_validate_yolo(
        checkpoint_path=cfg.onnx.checkpoint_path,
        output_onnx_path=cfg.onnx.onnx_path,
        img_size=cfg.onnx.img_size,
        atol=cfg.onnx.atol,
        rtol=cfg.onnx.rtol,
    )


if __name__ == "__main__":
    main()
