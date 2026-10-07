"""Deep models (needs ``pip install engagecam[torch]``). Import this module only when you train them.

The data loader calls the same ``preprocess`` function as inference, with the input spec of the
model, so the size and the scaling always match the backbone.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from engagecam.preprocess import preprocess, spec_for
from engagecam.states import STATES, encode


class TinyCNN(nn.Module):
    def __init__(self, n: int = len(STATES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, n))

    def forward(self, x):
        return self.net(x)


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name == "tiny_cnn":
        return TinyCNN()
    if name == "efficientnet_b0":
        from torchvision.models import efficientnet_b0

        m = efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(STATES))
        return m
    raise KeyError(f"unknown deep model {name!r}")


class FrameDataset(Dataset):
    def __init__(self, frames: pd.DataFrame, source, model: str, flip: bool = False, seed: int = 0):
        self.paths, self.y = list(frames["path"]), encode(frames["state"])
        self.source, self.spec, self.flip, self.seed, self.epoch = source, spec_for(model), flip, seed, 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x = preprocess(self.source.get(self.paths[i]), self.spec)
        if self.flip and np.random.default_rng([self.seed, self.epoch, i]).random() < 0.5:
            x = x[:, :, ::-1].copy()  # horizontal flip, training only
        return torch.from_numpy(x), self.y[i]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    return np.concatenate([torch.softmax(model(x), 1).numpy() for x, _ in loader])


def train(split: pd.DataFrame, source, model_name: str = "tiny_cnn", epochs: int = 5, lr: float = 3e-3,
          batch_size: int = 64, seed: int = 0, pretrained: bool = True, out_dir: str | Path = "runs") -> dict:
    """Class-weighted training on the train split, best epoch by val balanced accuracy, reloaded for test."""
    from engagecam.evaluate import report

    set_seed(seed)
    tr, va, te = (split[split["split"] == s] for s in ("train", "val", "test"))
    ds_tr = FrameDataset(tr, source, model_name, flip=True, seed=seed)
    gen = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, generator=gen)
    dl_va = DataLoader(FrameDataset(va, source, model_name), batch_size=batch_size)
    dl_te = DataLoader(FrameDataset(te, source, model_name), batch_size=batch_size)
    counts = np.bincount(ds_tr.y, minlength=len(STATES)).astype(float)
    w = np.where(counts > 0, counts.sum() / (len(STATES) * np.maximum(counts, 1)), 0.0)
    model = build_model(model_name, pretrained)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ckpt = Path(out_dir) / model_name / f"seed{seed}"
    ckpt.mkdir(parents=True, exist_ok=True)
    best, history = -1.0, []
    y_va = np.asarray(encode(va["state"]))
    for epoch in range(epochs):
        ds_tr.epoch = epoch
        model.train()
        for x, y in dl_tr:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        score = report(y_va, predict(model, dl_va))["balanced_accuracy"]
        history.append(score)
        if score > best:
            best = score
            torch.save(model.state_dict(), ckpt / "best.pt")
    model.load_state_dict(torch.load(ckpt / "best.pt", weights_only=True))
    return {"history": history, "best_val": best, "test": report(np.asarray(encode(te["state"])), predict(model, dl_te))}
