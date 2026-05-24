import albumentations as A
from albumentations.pytorch import ToTensorV2
from dvc.repo import Repo as DvcRepo


def pull_dvc_data(
    path: str | None = None,
    dvc_root: str = ".",
    remote: str | None = None,
    allow_missing: bool = False,
    force: bool = False,
) -> None:
    repo = DvcRepo(dvc_root)

    targets = [path] if path else []

    repo.pull(
        targets=targets,
        remote=remote,
        allow_missing=allow_missing,
        force=force,
    )

    print(f"DVC data pulled from remote='{remote}' for path='{path or '.'}'")


def build_transforms(split: str, height: int, width: int) -> A.Compose:
    transforms_list = [
        A.Resize(height=height, width=width),
        A.Normalize(
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]

    if split == "train":
        transforms_list = [
            A.HorizontalFlip(p=0.5),
        ] + transforms_list

    return A.Compose(
        transforms_list,  # type: ignore
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0,
            min_area=0.0,
            clip=True,
        ),
    )
