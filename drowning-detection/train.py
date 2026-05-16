import mlflow  # noqa: F401
import hydra
from model import MyModel
from data import AFODataModule
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def train(cfg: DictConfig):
    L.seed_everything(cfg.train.seed)

    dt = AFODataModule(
        data_root=cfg.data.dir_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        height=cfg.data.height,
        width=cfg.data.width,
    )

    model = MyModel(
        model_path=cfg.model.model_path,
        num_classes=cfg.train.num_classes,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        warmup_epochs=cfg.train.warmup_epochs,
        max_epochs=cfg.train.max_epochs,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.train.ckpt_path,
            filename="best",
            monitor="val/mAP_50",
            save_top_k=1,
            mode="max",
        ),
        ModelCheckpoint(
            dirpath=cfg.train.ckpt_path,
            filename="last",
            save_top_k=1,
            every_n_epochs=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = MLFlowLogger(
        experiment_name="AFO detection",
        tracking_uri="http://localhost:8080",
        run_name=None,
    )

    trainer = L.Trainer(
        max_epochs=cfg.train.max_epochs,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=5,
    )

    trainer.fit(
        model,
        datamodule=dt,
    )


if __name__ == "__main__":
    train()
