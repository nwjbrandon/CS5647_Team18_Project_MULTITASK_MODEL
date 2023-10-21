import torch

from models import MultiTaskClassificationModel

if __name__ == "__main__":
    hyperparams = {
        "dataset": "MULTITASK",
        "batch_size": 32,
        "n_classes": 2,
        "learning_rate": 0.003,
        "n_epochs": 30,
        "device": "mps",
        "test_size": 0.2,
        "random_state": 42,
        "n_mfcc": 128,
        "max_pad": 60,
        "n_tones": 4,
        "n_pinyins": 410,
    }

    model = MultiTaskClassificationModel(hyperparams)
    weights = torch.load("MULTITASK_model_25.pth")["model_state_dict"]
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        # Random Input
        inp = torch.rand(2, 1, 128, 60)
        tone_out, pinyin_out = model(inp)
        print(tone_out.shape, pinyin_out.shape)

        # Save only the feature extractor
        feature_extractor = model.feature_extractor
        torch.save(feature_extractor, "feature_extractor.pth")
