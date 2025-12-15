import os
import argparse
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from model_1d import resnet18_1d


FIXED_LENGTH = 66150


class AudioDataset1D(Dataset):
    def __init__(self, directory, sr=22050, fixed_length=FIXED_LENGTH):
        self.directory = directory
        self.sr = sr
        self.fixed_length = fixed_length

        self.filepaths = []
        self.labels = []

        for label in [0, 1]:
            class_dir = os.path.join(self.directory, str(label))
            if not os.path.isdir(class_dir):
                continue
            for fn in os.listdir(class_dir):
                if fn.lower().endswith(".wav"):
                    self.filepaths.append(os.path.join(class_dir, fn))
                    self.labels.append(label)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        label = self.labels[idx]

        y, _ = librosa.load(path, sr=self.sr)
        if len(y) < self.fixed_length:
            y = np.pad(y, (0, self.fixed_length - len(y)), mode="constant")
        else:
            y = y[:self.fixed_length]

        x = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, label


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_y, all_pred = [], []
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Test"):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_y.extend(y.detach().cpu().numpy())
        all_pred.extend(pred.detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    acc = (correct / max(1, total)) * 100.0
    f1 = f1_score(all_y, all_pred, average="macro")
    return acc, avg_loss, f1, all_y, all_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--weights", type=str, required=True, help="예: ./model_weight_1d_best.pth")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_class", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = AudioDataset1D(os.path.join(args.data_root, "test"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = resnet18_1d(n_class=args.n_class).to(device)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)

    acc, loss, f1, y_true, y_pred = evaluate(model, test_loader, device)

    print(f"Test Accuracy: {acc:.2f}% | Avg Loss: {loss:.4f} | F1(macro): {f1:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
