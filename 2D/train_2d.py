import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ReduceLROnPlateau

from model_2d import resnet18_2d
from preprocess_2d import ImageDataset2D, build_image_transform


def build_scheduler(name, optimizer):
    name = (name or "none").lower()
    if name == "multistep":
        return MultiStepLR(optimizer, milestones=[5, 10], gamma=0.1)
    if name == "steplr":
        return StepLR(optimizer, step_size=5, gamma=0.1)
    if name == "reducelronplateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_y, all_pred = [], []
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.extend(y.detach().cpu().numpy())
        all_pred.extend(pred.detach().cpu().numpy())

    avg_loss = running_loss / max(1, len(loader))
    acc = correct / max(1, total)
    f1 = f1_score(all_y, all_pred, average="weighted")
    return avg_loss, acc, f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_y, all_pred = [], []
    correct, total = 0, 0

    for x, y in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        logits = model(x)
        loss = criterion(logits, y)
        running_loss += loss.item()
        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)
        all_y.extend(y.detach().cpu().numpy())
        all_pred.extend(pred.detach().cpu().numpy())

    avg_loss = running_loss / max(1, len(loader))
    acc = correct / max(1, total)
    f1 = f1_score(all_y, all_pred, average="weighted")
    return avg_loss, acc, f1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="/content/Dataset/2D")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--scheduler", type=str, default="reducelronplateau",
                   choices=["none", "multistep", "steplr", "reducelronplateau"])
    p.add_argument("--n_class", type=int, default=2)
    p.add_argument("--save_path", type=str, default="./best_model_weight_2d.pth")
    p.add_argument("--patience", type=int, default=999999)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tfm = build_image_transform(img_size=224)
    train_ds = ImageDataset2D(os.path.join(args.data_root, "train_spec"), transform=tfm)
    val_ds = ImageDataset2D(os.path.join(args.data_root, "validation_spec"), transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = resnet18_2d(n_class=args.n_class).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = build_scheduler(args.scheduler, optimizer)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_f1 = validate(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
            f"Val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}"
        )

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            torch.save(best_state, args.save_path)
            print(f"  -> Best updated. Saved to {args.save_path}")
        else:
            no_improve += 1

        if scheduler is not None:
            if args.scheduler == "reducelronplateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()

        if no_improve >= args.patience:
            break

    elapsed = time.time() - t0
    print(f"Training finished. Elapsed: {elapsed/60:.1f} min")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
