import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, classification_report

from model_1d import resnet18_1d
from model_2d import resnet18_2d
from preprocess_2d import ImageDataset2D, build_image_transform
from preprocess_1d import AudioDataset1D


@torch.no_grad()
def evaluate_logits(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_y, all_pred, all_prob1 = [], [], []

    for x, y in tqdm(loader, desc="Evaluate"):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)

        all_y.extend(y.detach().cpu().numpy())
        all_pred.extend(pred.detach().cpu().numpy())
        all_prob1.extend(probs[:, 1].detach().cpu().numpy())

    avg_loss = total_loss / max(1, len(loader))
    f1 = f1_score(all_y, all_pred, average="macro")

    auc = None
    if len(set(all_y)) == 2:
        auc = roc_auc_score(all_y, all_prob1)

    return avg_loss, f1, auc, all_y, all_pred


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["1d", "2d"])
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_class", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "2d":
        tfm = build_image_transform(img_size=224)
        test_ds = ImageDataset2D(os.path.join(args.data_root, "test_spec"), transform=tfm)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        model = resnet18_2d(n_class=args.n_class)
    else:
        test_ds = AudioDataset1D(os.path.join(args.data_root, "test"))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        model = resnet18_1d(n_class=args.n_class)

    state = torch.load(args.weights, map_location="cpu")
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load weights: {args.weights}\n"
            f"Mode={args.mode}, n_class={args.n_class}\n"
            f"Original error: {e}"
        )

    model = model.to(device)

    loss, f1, auc, y_true, y_pred = evaluate_logits(model, test_loader, device)

    msg = f"Avg Loss: {loss:.4f} | F1(macro): {f1:.4f}"
    msg += f" | AUC: {auc:.4f}" if auc is not None else " | AUC: N/A"
    print(msg)

    print(classification_report(y_true, y_pred, digits=4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
