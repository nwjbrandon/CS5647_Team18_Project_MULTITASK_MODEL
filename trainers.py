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
        self.dataset = hyperparams["dataset"]

    def decode_one_hot_encoding(self, out, label):
        _, predicted = torch.max(out.data, 1)
        label = label.detach().cpu().numpy().tolist()
        predicted = predicted.detach().cpu().numpy().tolist()
        return label, predicted

    def decode_binarized_encoding(self, out, label):
        label = label.detach().cpu()
        mask = 2 ** torch.arange(11 - 1, -1, -1)
        label = torch.sum(mask * label.data, -1).numpy().tolist()

        out[out > 0.5] = 1
        out[out <= 0.5] = 0
        out = out.detach().cpu()
        mask = 2 ** torch.arange(11 - 1, -1, -1)
        predicted = torch.sum(mask * out.data, -1).numpy().tolist()
        return label, predicted

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

            if self.dataset in ["TONES", "PINYINS"]:
                label, predicted = self.decode_one_hot_encoding(out, label)
                losses.append(loss.item())
                y_true.extend(label)
                y_pred.extend(predicted)
            else:
                label, predicted = self.decode_binarized_encoding(out, label)
                losses.append(loss.item())
                y_true.extend(label)
                y_pred.extend(predicted)

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

                if self.dataset in ["TONES", "PINYINS"]:
                    label, predicted = self.decode_one_hot_encoding(out, label)
                    losses.append(loss.item())
                    y_true.extend(label)
                    y_pred.extend(predicted)
                else:
                    label, predicted = self.decode_binarized_encoding(out, label)
                    losses.append(loss.item())
                    y_true.extend(label)
                    y_pred.extend(predicted)

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


class MultiTaskClassificationTrainer:
    def __init__(
        self,
        hyperparams,
        model,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        tone_metric,
        pinyin_metric,
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
        self.tone_metric = tone_metric
        self.pinyin_metric = pinyin_metric
        self.scheduler = scheduler
        self.device = device
        self.pretrained_model = hyperparams.get("pretrained_model", "")
        self.dataset = hyperparams["dataset"]

    def decode_one_hot_encoding(self, out, label):
        _, predicted = torch.max(out.data, 1)
        label = label.detach().cpu().numpy().tolist()
        predicted = predicted.detach().cpu().numpy().tolist()
        return label, predicted

    def train(self):
        self.model.train()
        losses = []
        y_true_tone = []
        y_pred_tone = []
        y_true_pinyin = []
        y_pred_pinyin = []
        for img, tone_label, pinyin_label in tqdm(self.train_dataloader):
            inp = img.float().to(self.device)
            tone_label = tone_label.to(self.device)
            pinyin_label = pinyin_label.to(self.device)
            self.optimizer.zero_grad()
            tone_out, pinyin_out = self.model(inp)
            loss = self.criterion(tone_out, tone_label, pinyin_out, pinyin_label)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())
            tone_label, tone_predicted = self.decode_one_hot_encoding(tone_out, tone_label)
            y_true_tone.extend(tone_label)
            y_pred_tone.extend(tone_predicted)
            pinyin_label, pinyin_predicted = self.decode_one_hot_encoding(pinyin_out, pinyin_label)
            y_true_pinyin.extend(pinyin_label)
            y_pred_pinyin.extend(pinyin_predicted)

        y_true_tone = torch.tensor(y_true_tone)
        y_pred_tone = torch.tensor(y_pred_tone)
        y_true_pinyin = torch.tensor(y_true_pinyin)
        y_pred_pinyin = torch.tensor(y_pred_pinyin)
        loss = np.sum(losses) / len(losses)
        tone_acc = self.tone_metric(y_pred_tone, y_true_tone).item()
        pinyin_acc = self.pinyin_metric(y_pred_pinyin, y_true_pinyin).item()
        return tone_acc, pinyin_acc, loss

    def validate(self):
        self.model.eval()
        losses = []
        y_true_tone = []
        y_pred_tone = []
        y_true_pinyin = []
        y_pred_pinyin = []
        with torch.no_grad():
            for img, tone_label, pinyin_label in tqdm(self.valid_dataloader):
                inp = img.float().to(self.device)
                tone_label = tone_label.to(self.device)
                pinyin_label = pinyin_label.to(self.device)
                tone_out, pinyin_out = self.model(inp)
                loss = self.criterion(tone_out, tone_label, pinyin_out, pinyin_label)

                losses.append(loss.item())
                tone_label, tone_predicted = self.decode_one_hot_encoding(tone_out, tone_label)
                y_true_tone.extend(tone_label)
                y_pred_tone.extend(tone_predicted)
                pinyin_label, pinyin_predicted = self.decode_one_hot_encoding(pinyin_out, pinyin_label)
                y_true_pinyin.extend(pinyin_label)
                y_pred_pinyin.extend(pinyin_predicted)

        y_true_tone = torch.tensor(y_true_tone)
        y_pred_tone = torch.tensor(y_pred_tone)
        y_true_pinyin = torch.tensor(y_true_pinyin)
        y_pred_pinyin = torch.tensor(y_pred_pinyin)
        loss = np.sum(losses) / len(losses)
        tone_acc = self.tone_metric(y_pred_tone, y_true_tone).item()
        pinyin_acc = self.pinyin_metric(y_pred_pinyin, y_true_pinyin).item()
        return tone_acc, pinyin_acc, loss

    def save_checkpoint(
        self, epoch, train_loss, valid_loss, tone_train_acc, tone_valid_acc, pinyin_train_acc, pinyin_valid_acc
    ):
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
                "tone_train_acc": tone_train_acc,
                "tone_valid_acc": tone_valid_acc,
                "pinyin_train_acc": pinyin_train_acc,
                "pinyin_valid_acc": pinyin_valid_acc,
            },
            ckpt_fname,
        )

    def print_log(
        self, epoch, n_epochs, train_loss, valid_loss, tone_train_acc, tone_valid_acc, pinyin_train_acc, pinyin_valid_acc
    ):
        log = "Epoch: {}/{}, Tone Train Acc={}, Tone Val Acc={}, Pinyin Train Acc={}, Pinyin Val Acc={}, Train Loss={}, Val Loss={}".format(
            epoch + 1,
            n_epochs,
            np.round(tone_train_acc, 4),
            np.round(tone_valid_acc, 4),
            np.round(pinyin_train_acc, 4),
            np.round(pinyin_valid_acc, 4),
            np.round(train_loss, 4),
            np.round(valid_loss, 4),
        )
        print(log)

    def run(self):
        n_epochs = self.hyperparams["n_epochs"]
        records = []
        for epoch in range(n_epochs):
            tone_train_acc, pinyin_train_acc, train_loss = self.train()
            tone_valid_acc, pinyin_valid_acc, valid_loss = self.validate()
            self.scheduler.step()

            self.save_checkpoint(
                epoch, train_loss, valid_loss, tone_train_acc, tone_valid_acc, pinyin_train_acc, pinyin_valid_acc
            )
            self.print_log(
                epoch, n_epochs, train_loss, valid_loss, tone_train_acc, tone_valid_acc, pinyin_train_acc, pinyin_valid_acc
            )
            records.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "tone_train_acc": tone_train_acc,
                    "tone_valid_acc": tone_valid_acc,
                    "pinyin_train_acc": pinyin_train_acc,
                    "pinyin_valid_acc": pinyin_valid_acc,
                }
            )

        if self.pretrained_model == "":
            logs_fname = f"ckpts/{self.hyperparams['dataset']}_logs.csv"
        else:
            pretrained_model_name = self.pretrained_model.replace(".pth", "")
            logs_fname = f"ckpts/{self.hyperparams['dataset']}_{pretrained_model_name}_logs.csv"
        df = pd.DataFrame(records)
        df.to_csv(logs_fname, sep=",", index=False)
