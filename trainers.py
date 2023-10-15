import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class ClassificationTrainer:
    def __init__(
        self,
        hyperparams,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        metric,
        scheduler,
        device,
    ):
        self.hyperparams = hyperparams
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metric = metric
        self.scheduler = scheduler
        self.device = device
        self.pretrained_model = hyperparams.get("pretrained_model", "")

    def train(self):
        self.model.train()
        losses = []
        y_true = []
        y_pred = []
        for img, label in tqdm(self.train_dataloader):
            inp = img.float().to(self.device)
            label = label.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(inp)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(out.data, 1)
            label = label.detach().cpu().numpy().tolist()
            y_true.extend(label)
            predicted = predicted.detach().cpu().numpy().tolist()
            y_pred.extend(predicted)
            losses.append(loss.item())

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        loss = np.sum(losses) / len(losses)
        acc = self.metric(y_pred, y_true).item()
        return acc, loss

    def validate(self):
        self.model.eval()
        losses = []
        y_true = []
        y_pred = []
        with torch.no_grad():
            for img, label in tqdm(self.valid_dataloader):
                inp = img.float().to(self.device)
                label = label.to(self.device)
                out = self.model(inp)
                loss = self.criterion(out, label)

                _, predicted = torch.max(out.data, 1)
                label = label.detach().cpu().numpy().tolist()
                y_true.extend(label)
                predicted = predicted.detach().cpu().numpy().tolist()
                y_pred.extend(predicted)
                losses.append(loss.item())

        y_pred = torch.tensor(y_pred)
        y_true = torch.tensor(y_true)
        loss = np.sum(losses) / len(losses)
        acc = self.metric(y_pred, y_true).item()
        return acc, loss

    def save_checkpoint(self, epoch, train_loss, valid_loss, train_acc, valid_acc):
        if self.pretrained_model == "":
            ckpt_fname = f"ckpts/{self.hyperparams['dataset']}_model_{epoch}.pth"
        else:
            pretrained_model_name = self.pretrained_model.replace(".pth", "")
            ckpt_fname = f"ckpts/{self.hyperparams['dataset']}_{pretrained_model_name}_model_{epoch}.pth"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
            },
            ckpt_fname,
        )

    def print_log(self, epoch, n_epochs, train_loss, valid_loss, train_acc, valid_acc):
        log = "Epoch: {}/{}, Train Acc={}, Val Acc={}, Train Loss={}, Val Loss={}".format(
            epoch + 1,
            n_epochs,
            np.round(train_acc, 4),
            np.round(valid_acc, 4),
            np.round(train_loss, 4),
            np.round(valid_loss, 4),
        )
        print(log)

    def run(self):
        n_epochs = self.hyperparams["n_epochs"]
        records = []
        for epoch in range(n_epochs):
            train_acc, train_loss = self.train()
            valid_acc, valid_loss = self.validate()
            self.scheduler.step()

            self.save_checkpoint(epoch, train_loss, valid_loss, train_acc, valid_acc)
            self.print_log(epoch, n_epochs, train_loss, valid_loss, train_acc, valid_acc)
            records.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_acc": train_acc,
                    "valid_acc": valid_acc,
                }
            )

        if self.pretrained_model == "":
            logs_fname = f"ckpts/{self.hyperparams['dataset']}_logs.csv"
        else:
            pretrained_model_name = self.pretrained_model.replace(".pth", "")
            logs_fname = f"ckpts/{self.hyperparams['dataset']}_{pretrained_model_name}_logs.csv"
        df = pd.DataFrame(records)
        df.to_csv(logs_fname, sep=",", index=False)
