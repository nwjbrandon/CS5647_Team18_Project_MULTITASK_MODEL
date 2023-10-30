import torchmetrics
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from datasets import create_dataloaders
from losses import MultiTaskClassificationLoss
from models import Wav2LetterModel
from trainers import MultiTaskClassificationTrainer


def main():
    hyperparams = {
        "dataset": "MULTITASK",
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
    }
    print(hyperparams)

    train_dataloader, test_dataloader = create_dataloaders(hyperparams, mode="MULTITASK")

    model = Wav2LetterModel(hyperparams)
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
    criterion = MultiTaskClassificationLoss()
    tone_metric = torchmetrics.Accuracy(task="multiclass", num_classes=hyperparams["n_tones"])
    pinyin_metric = torchmetrics.Accuracy(task="multiclass", num_classes=hyperparams["n_pinyins"])

    trainer = MultiTaskClassificationTrainer(
        hyperparams,
        model,
        train_dataloader,
        test_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        tone_metric,
        pinyin_metric,
        scheduler,
        device,
    )
    trainer.run()


if __name__ == "__main__":
    main()
