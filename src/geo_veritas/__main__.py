from geo_veritas import MNISTDataModule
from geo_veritas import GAN
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == '__main__':
    dm = MNISTDataModule()
    model = GAN(*dm.dims)
    logger = TensorBoardLogger("tb_logs", name="gan")
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=5,
        logger=logger,
    )
    trainer.fit(model, dm)
