from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAccuracy

from datasets import create_dataloaders
from losses import SiameseLoss
from models import SiameseModel
from trainers import SiameseTrainer


def main():
    hyperparams = {
        "dataset": "SIAMESE",
        "batch_size": 32,
        "n_classes": 2,
        "learning_rate": 0.003,
        "n_epochs": 30,
        "device": "mps",
        "test_size": 0.2,
        "random_state": 42,
        "n_mfcc": 128,
        "max_pad": 60,
        "n_out": 1,
    }
    print(hyperparams)

    train_dataloader, test_dataloader = create_dataloaders(hyperparams, mode="SIAMESE")

    model = SiameseModel(hyperparams)
    device = hyperparams["device"]
    model.to(device)
    print(model)

    optimizer = Adam(model.parameters(), lr=hyperparams["learning_rate"])
    scheduler = StepLR(
        optimizer=optimizer,
        step_size=1,
        gamma=0.9,
        verbose=False,
    )
    criterion = SiameseLoss()
    metric = BinaryAccuracy(threshold=0.5)

    trainer = SiameseTrainer(
        hyperparams,
        model,
        train_dataloader,
        test_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        metric,
        scheduler,
        device,
    )
    trainer.run()


if __name__ == "__main__":
    main()
