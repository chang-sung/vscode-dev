import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset


DATASET_DIR = r"D:\25_Project\01_Python\01_Transformer\dataset"
SAVE_DIR = r"D:\25_Project\01_Python\01_Transformer\models"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
VAL_RATIO = 0.2
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# =========================
# Dataset
# =========================
class LinearStepDataset(Dataset):
    def __init__(self, csv_files, window=720, stride=24, label_th=0.3):
        self.samples = []
        self.cls = []   # 👈 stratify용

        for path in csv_files:
            df = pd.read_csv(path)

            x = df["norm"].values.astype(np.float32)
            y_linear = df["y_linear"].values.astype(np.float32)
            y_step   = df["y_step"].values.astype(np.float32)

            N = len(df)
            for s in range(0, N - window + 1, stride):
                e = s + window
                x_win = x[s:e]

                yl = float(y_linear[s:e].mean() > label_th)
                ys = float(y_step[s:e].mean() > label_th)

                self.samples.append((x_win[None, :], np.array([yl, ys], dtype=np.float32)))

                # stratify label
                if ys == 1:
                    self.cls.append(2)
                elif yl == 1:
                    self.cls.append(1)
                else:
                    self.cls.append(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)



class SpikeDataset(Dataset):
    def __init__(self, csv_files, window=24, stride=3):
        self.samples = []
        self.cls = []

        for path in csv_files:
            df = pd.read_csv(path)

            x = df["dx_pos"].values.astype(np.float32)
            y = df["y_spike"].values.astype(np.float32)

            N = len(df)
            for s in range(0, N - window + 1, stride):
                e = s + window
                x_win = x[s:e]
                y_win = float(y[s:e].max())

                self.samples.append((x_win[None, :], np.array([y_win], dtype=np.float32)))
                self.cls.append(int(y_win))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


# =========================
# Models
# =========================
class CNN_LinearStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return torch.sigmoid(self.fc(x))


class CNN_Spike(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = self.net(x).squeeze(-1)
        return torch.sigmoid(self.fc(x))


# =========================
# Train / Eval with Best Save
# =========================
@torch.no_grad()
def eval_epoch(model, dl, criterion):
    model.eval()
    total = 0.0
    for x, y in dl:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        total += loss.item()
    return total / max(1, len(dl))


# Stratified K-Fold
def train_kfold(model_cls, dataset, model_name, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    cls = np.array(dataset.cls)

    best_overall = float("inf")
    best_path = None

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(cls)), cls)):
        print(f"\n===== [{model_name}] Fold {fold+1}/{n_splits} =====")

        train_ds = Subset(dataset, tr_idx)
        val_ds   = Subset(dataset, va_idx)

        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = model_cls().to(DEVICE)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_fold = float("inf")

        for epoch in range(EPOCHS):
            model.train()
            total_train = 0.0

            for x, y in train_dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                total_train += loss.item()

            train_loss = total_train / max(1, len(train_dl))
            val_loss = eval_epoch(model, val_dl, criterion)

            if val_loss < best_fold:
                best_fold = val_loss

            print(f"Fold {fold+1} | Epoch {epoch+1}/{EPOCHS} | "
                  f"train {train_loss:.4f} | val {val_loss:.4f}")

        # fold best 저장
        fold_path = os.path.join(SAVE_DIR, f"{model_name}_fold{fold+1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "val_loss": best_fold,
            "fold": fold + 1
        }, fold_path)

        print(f"✅ Fold {fold+1} best val={best_fold:.4f}")

        # overall best
        if best_fold < best_overall:
            best_overall = best_fold
            best_path = fold_path

    print(f"\n🏆 [{model_name}] Overall BEST: {best_path} (val={best_overall:.4f})")
    return best_path


# =========================
# Run
# =========================
csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found")

# Linear + Step (멀티라벨)
ls_ds = LinearStepDataset(csv_files)
ls_best = train_kfold(CNN_LinearStep, ls_ds, "LinearStep")

# Spike
sp_ds = SpikeDataset(csv_files)
sp_best = train_kfold(CNN_Spike, sp_ds, "Spike")
