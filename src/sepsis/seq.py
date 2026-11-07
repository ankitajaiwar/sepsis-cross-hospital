"""A recurrent neural network (GRU) that reads each patient's hourly history, with
optional domain-adversarial training (DANN) to adapt to a new hospital without its
labels.

Inputs at each hour: the last known value of each measurement (standardised), a flag
for whether it was measured this hour, and how long ago it was last measured
(missingness is informative in ICU data), plus age, sex, unit and time in ICU.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from .calibration import Platt
from .schema import PHYSIO

MAX_LEN = 336  # 14 days of hourly data, the challenge maximum


def _patients(df: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    """Turn the hourly table into a list of (features[T, D], labels[T]) per patient."""
    df = df.sort_values(["person_id", "hour"])
    vals = df[PHYSIO]
    meas = vals.notna().to_numpy(np.float32)
    z = ((vals - mu) / sd).groupby(df["person_id"], sort=False).ffill().fillna(0).to_numpy(np.float32)
    z = np.clip(z, -6, 6)
    hour = df["hour"].to_numpy(np.float32)
    since = np.empty_like(meas)
    for j, v in enumerate(PHYSIO):
        last = pd.Series(np.where(meas[:, j] > 0, hour, np.nan)).groupby(
            df["person_id"].to_numpy(), sort=False).ffill().to_numpy()
        since[:, j] = np.where(np.isnan(last), 1.0, np.log1p(hour - last) / math.log(337))
    static = np.stack([
        ((df["Age"].fillna(62) - 62) / 16).to_numpy(np.float32),
        df["Gender"].fillna(0.5).to_numpy(np.float32),
        df["Unit1"].fillna(0).to_numpy(np.float32),
        df["Unit2"].fillna(0).to_numpy(np.float32),
        (df["Unit1"].isna()).to_numpy(np.float32),
        np.clip(-df["HospAdmTime"].fillna(0).to_numpy(np.float32) / 100, -5, 5),
        (df["ICULOS"].to_numpy(np.float32) / 100),
    ], axis=1)
    X = np.concatenate([z, meas, since, static], axis=1)
    y = df["SepsisLabel"].to_numpy(np.float32)
    bounds = np.flatnonzero(np.r_[True, df["person_id"].to_numpy()[1:] != df["person_id"].to_numpy()[:-1], True])
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        b = min(b, a + MAX_LEN)
        out.append((torch.from_numpy(X[a:b].copy()), torch.from_numpy(y[a:b].copy())))
    return out, df.index


def _batch(items):
    lens = torch.tensor([len(x) for x, _ in items])
    T = int(lens.max())
    X = torch.zeros(len(items), T, items[0][0].shape[1])
    Y = torch.zeros(len(items), T)
    for i, (x, y) in enumerate(items):
        X[i, : len(x)] = x
        Y[i, : len(y)] = y
    mask = torch.arange(T)[None, :] < lens[:, None]
    return X, Y, mask


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class GRUNet(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        self.domain = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, lam: float = 0.0):
        h, _ = self.gru(self.inp(x))
        logit = self.head(h).squeeze(-1)
        dom = self.domain(_GradReverse.apply(h, lam)).squeeze(-1)
        return logit, dom


class GRUModel:
    def __init__(self, epochs: int = 8, batch: int = 64, lr: float = 1e-3, hidden: int = 64,
                 dann: float = 0.0, seed: int = 0, threads: int | None = None):
        self.epochs, self.batch, self.lr, self.hidden = epochs, batch, lr, hidden
        self.dann, self.seed = dann, seed
        torch.set_num_threads(threads or max(1, (os.cpu_count() or 2) - 1))
        self.platt = None

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, target: pd.DataFrame | None = None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.mu = train[PHYSIO].mean()
        self.sd = train[PHYSIO].std().replace(0, 1).fillna(1)
        tr, _ = _patients(train, self.mu, self.sd)
        va, _ = _patients(valid, self.mu, self.sd)
        tg = _patients(target, self.mu, self.sd)[0] if (target is not None and self.dann > 0) else None
        self.net = GRUNet(tr[0][0].shape[1], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        pos = float(train["SepsisLabel"].mean())
        pw = torch.tensor(min(10.0, math.sqrt((1 - pos) / max(pos, 1e-4))))
        best, best_state = -1.0, None
        steps = math.ceil(len(tr) / self.batch)
        for ep in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(tr))
            for k in range(steps):
                X, Y, M = _batch([tr[i] for i in order[k * self.batch:(k + 1) * self.batch]])
                prog = (ep * steps + k) / (self.epochs * steps)
                lam = self.dann * (2 / (1 + math.exp(-10 * prog)) - 1) if tg else 0.0
                logit, dom = self.net(X, lam)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logit[M], Y[M], pos_weight=pw)
                if tg:
                    Xt, _, Mt = _batch([tg[i] for i in rng.choice(len(tg), self.batch)])
                    _, dom_t = self.net(Xt, lam)
                    d_logits = torch.cat([dom[M], dom_t[Mt]])
                    d_y = torch.cat([torch.zeros(int(M.sum())), torch.ones(int(Mt.sum()))])
                    loss = loss + nn.functional.binary_cross_entropy_with_logits(d_logits, d_y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()
            s = self._logits(va)
            yv = np.concatenate([y.numpy() for _, y in va])
            auc = roc_auc_score(yv, s) if 0 < yv.sum() < len(yv) else 0.5
            if auc > best:
                best, best_state = auc, {k: v.clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        self.best_valid_auroc = best
        return self

    @torch.no_grad()
    def _logits(self, pats) -> np.ndarray:
        self.net.eval()
        out = []
        for k in range(0, len(pats), 256):
            items = pats[k:k + 256]
            X, _, M = _batch(items)
            logit, _ = self.net(X)
            out.append(logit[M].numpy())
        return np.concatenate(out) if out else np.array([])

    def _scores(self, df: pd.DataFrame) -> pd.Series:
        """Logits aligned to df's rows. Hours beyond MAX_LEN get the last logit."""
        df = df.sort_values(["person_id", "hour"])
        pats, idx = _patients(df, self.mu, self.sd)
        logits = self._logits(pats)
        lens = df.groupby("person_id", sort=False).size().to_numpy()
        kept = np.minimum(lens, MAX_LEN)
        full = np.empty(len(df), dtype=np.float32)
        a = b = 0
        for n, k in zip(lens, kept):
            full[a:a + k] = logits[b:b + k]
            if n > k:
                full[a + k:a + n] = logits[b + k - 1]
            a += n
            b += k
        return pd.Series(full, index=idx)

    def calibrate(self, df: pd.DataFrame):
        s = self._scores(df)
        self.platt = Platt().fit(s.to_numpy(), df.loc[s.index, "SepsisLabel"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        s = self._scores(df)
        if self.platt is None:
            return pd.Series(1 / (1 + np.exp(-s.to_numpy())), index=s.index)
        return pd.Series(self.platt(s.to_numpy()), index=s.index)
