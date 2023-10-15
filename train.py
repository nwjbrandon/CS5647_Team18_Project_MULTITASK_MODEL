import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from datasets import create_dataloader
from losses import ClassificationLoss
from models import Model
from trainers import ClassificationTrainer


def get_hyperparams(mode):
    if mode == "TONES":
        hyperparams = {
            "dataset": "TONES",
            "batch_size": 32,
            "n_classes": 4,
            "learning_rate": 0.03,
            "n_epochs": 30,
            "device": "mps",
            "test_size": 0.2,
            "random_state": 42,
            "n_mfcc": 128,
            "max_pad": 60,
        }
    elif mode == "PINYINS":
        hyperparams = {
            "dataset": "PINYINS",
            "batch_size": 32,
            "n_classes": 410,
            "learning_rate": 0.03,
            "n_epochs": 30,
            "device": "mps",
            "test_size": 0.2,
            "random_state": 42,
            "n_mfcc": 128,
            "max_pad": 60,
        }
    elif mode == "LABELS":
        hyperparams = {
            "dataset": "LABELS",
            "batch_size": 32,
            "n_classes": 1640,
            "learning_rate": 0.03,
            "n_epochs": 30,
            "device": "mps",
            "test_size": 0.2,
            "random_state": 42,
            "n_mfcc": 128,
            "max_pad": 60,
        }
    else:
        raise "Invalid Dataset"
    return hyperparams


def main():
    hyperparams = get_hyperparams("TONES")
    print(hyperparams)

    train_dataloader, test_dataloader = create_dataloader(hyperparams)

    model = Model(hyperparams)
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
    criterion = ClassificationLoss()
    metric = torchmetrics.Accuracy(task="multiclass", num_classes=hyperparams["n_classes"])

    trainer = ClassificationTrainer(
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
