import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from datasets import create_dataloaders
from losses import ClassificationLoss
from models import ClassificationModel
from trainers import ClassificationTrainer


def get_hyperparams(mode):
    if mode == "TONES":
        hyperparams = {
            "dataset": "TONES",
            "batch_size": 32,
            "learning_rate": 0.003,
            "n_epochs": 30,
            "device": "mps",
            "test_size": 0.2,
            "random_state": 42,
            "preprocess_type": "mfcc",
            "n_mfcc": 128,
            "n_mels": 128,
            "sampling_rate": 16000,
            "max_length": 16000,
            "max_pad": 60,
            "n_tones": 4,
            "n_pinyins": 410,
            "n_classes": 4,
        }
    elif mode == "PINYINS":
        hyperparams = {
            "dataset": "PINYINS",
            "batch_size": 32,
            "learning_rate": 0.003,
            "n_epochs": 30,
            "device": "mps",
            "test_size": 0.2,
            "random_state": 42,
            "preprocess_type": "mfcc",
            "n_mfcc": 128,
            "n_mels": 128,
            "sampling_rate": 16000,
            "max_length": 16000,
            "max_pad": 60,
            "n_tones": 4,
            "n_pinyins": 410,
            "n_classes": 410,
        }
    elif mode == "LABELS":
        hyperparams = {
            "dataset": "LABELS",
            "batch_size": 32,
            "learning_rate": 0.003,
            "n_epochs": 30,
            "device": "mps",
            "test_size": 0.2,
            "random_state": 42,
            "preprocess_type": "mfcc",
            "n_mfcc": 128,
            "n_mels": 128,
            "sampling_rate": 16000,
            "max_length": 16000,
            "max_pad": 60,
            "n_tones": 4,
            "n_pinyins": 410,
            "n_classes": 1640,
        }
    else:
        raise "Invalid Dataset"
    return hyperparams


def main():
    hyperparams = get_hyperparams("TONES")
    print(hyperparams)

    train_dataloader, test_dataloader = create_dataloaders(hyperparams)

    model = ClassificationModel(hyperparams)
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
