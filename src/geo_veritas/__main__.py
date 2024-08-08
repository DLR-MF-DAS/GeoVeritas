from geo_veritas import MNISTDataModule
from geo_veritas import GAN
import lightning as L

if __name__ == '__main__':
    dm = MNISTDataModule()
    model = GAN(*dm.dims)
    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=5,
    )
    trainer.fit(model, dm)
